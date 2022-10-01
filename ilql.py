import os
import sys
import yaml
from time import time

import torch as th
from torch import tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Config, AutoTokenizer
from accelerate import Accelerator
import numpy as np

import wandb
from tqdm import tqdm, trange
from utils import run_from_ipython, timeit, check_weights
from models import QVModel
from copy import deepcopy
import accelerate
import deepspeed

th.set_printoptions(sci_mode=False)

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
WORLD_RANK = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

def main(**args):
    task = args['task'] if 'task' in args else 'RandomWalks'
    config = yaml.safe_load(open('config.yaml'))[task]
    config.update(args)

    accelerator = Accelerator(log_with='wandb')
    accelerator.print(os.environ)
    device = accelerator.device

    if WORLD_SIZE > 1:
        th.distributed.barrier(device_ids=[LOCAL_RANK])
    else:
        th.random.manual_seed(1000)

    if not config.get('debug', False) and accelerator.is_main_process:
        modelname = config.get('model', '')
        accelerator.init_trackers(project_name='test-ilql', init_kwargs={
            'wandb': {'name': f'ilql-{task}-{modelname}',
                      'mode': 'disabled' if args.get('debug', False) else 'online'}}, config=config)

        config = wandb.config

    if task == 'RandomWalks':
        from randomwalks import RandomWalks
        data = RandomWalks(seed=config['seed'])
        gptconfig = GPT2Config(**config['gptconfig'], vocab_size=data.n_nodes)
        model = QVModel(gptconfig, config)

    elif task == 'Sentiments':
        from sentiments import Sentiments

        with accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(config['model'])
            tokenizer.pad_token_id = tokenizer.eos_token_id
            data = Sentiments(tokenizer, needs_reward_model=accelerator.is_main_process)

        with timeit('init model'):
            model = QVModel(config['model'], config)

    elif task == 'Carps':
        from carps import Carps
        data = Carps(max_length=config['max_length'], diff_reward=config['diff_reward'])
        model = QVModel(config['model'], two_qs=config['two_qs']).to(device)

    elif task == 'Captions':
        from captions import AestheticCaptions

        tokenizer = AutoTokenizer.from_pretrained(config['model'])
        tokenizer.pad_token = tokenizer.eos_token_id
        with accelerator.main_process_first():
            data = AestheticCaptions(tokenizer, batch_size=config['batch_size'], n_samples=16)

        model = QVModel(config['model'], two_qs=config['two_qs']).to(device)
    else:
        raise ValueError(f'nonexistent {task=}')

    if hasattr(model.gpt, 'gpt_neox'):
        gpt_blocks = list(model.gpt.gpt_neox.layers)[:-config['n_layers_unfrozen']]
    else:
        gpt_blocks = list(model.gpt.transformer.h)[:-config['n_layers_unfrozen']]

    for m in gpt_blocks:
        m.requires_grad_(False)

    train_dataloader = DataLoader(data.dataset, batch_size=config['batch_size'])

    eval_batch_size = max(1, len(data.eval_dataset) // WORLD_SIZE)
    eval_dataloader = DataLoader(data.eval_dataset, eval_batch_size)

    opt_cls = (
        th.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else accelerate.utils.DummyOptim
    )
    opt = opt_cls([p for p in model.parameters() if p.requires_grad], lr=config['lr'], betas=config['opt_betas'])

    total_steps = int(config['n_epochs'] * (len(data.dataset) // (config['batch_size'] * WORLD_SIZE)))
    n_opt_steps = 0

    with timeit('prepare'):
        model, opt, train_dataloader, eval_dataloader = accelerator.prepare(
            model, opt, train_dataloader, eval_dataloader
        )

    print(f'{WORLD_RANK=}: {model(**accelerator.unwrap_model(model).dummy_inputs)[0].device}')

    model.train()
    tbar = trange(total_steps, disable=not accelerator.is_local_main_process)

    for iepoch in range(config['n_epochs']):
        for batch in train_dataloader:
            logs = {}

            if n_opt_steps % config['steps_for_eval'] == 0:
                model.eval()
                beta = config['inference_betas'][0]

                all_samples = []
                for tokens in eval_dataloader:
                    tokens = tokens[0].to(device)
                    with th.no_grad():
                        samples, stats = accelerator.unwrap_model(model).sample(
                            tokens,
                            beta=beta,
                            max_length=data.max_length,
                            logit_mask=data.logit_mask
                        )

                    all_samples.append(samples)
                    logs.update(stats)

                samples = accelerator.gather(th.vstack(all_samples))

                if accelerator.is_main_process:
                    reward, stats = data.eval(samples, beta)
                    logs.update(stats)
                    tbar.set_postfix(stats)

                model.train()

            for ix in range(len(batch)):
                batch[ix] = batch[ix].to(device)

            batch_time = time()
            forward_time = time()
            loss, stats = model.loss(batch)
            forward_time = time() - forward_time

            backward_time = time()
            accelerator.backward(loss)
            backward_time = time() - backward_time

            opt.step()
            opt.zero_grad()
            n_opt_steps += 1

            batch_time = time() - batch_time
            tokens_per_sec = batch[0].numel() * WORLD_SIZE / batch_time
            tbar.set_description(f'{tokens_per_sec=:.2f} {batch_time=:.2f}')
            tbar.update()

            if (n_opt_steps + 1) % config['steps_for_target_q_sync'] == 0:
                accelerator.unwrap_model(model).sync_target_q_heads()

            logs.update(stats)
            logs['target_sum'] = check_weights(accelerator.unwrap_model(model).target_q1_head)
            logs['batch_time'] = batch_time

            if not config.get('debug', False):
                accelerator.log(logs)

    return model, data

if __name__ == '__main__':
    if os.environ.get('LOCAL_RANK'):
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']

    if run_from_ipython:
        args = {'debug': True}
    else:
        # poor man's argparse
        args = {a[2:]: eval(v) for a, v in map(lambda s: s.split('='), sys.argv[1:])}

    main(**args)
