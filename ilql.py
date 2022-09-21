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

def main(**args):
    task = args['task'] if 'task' in args else 'RandomWalks'
    config = yaml.safe_load(open('config.yaml'))[task]
    config.update(args)

    accelerator = Accelerator(log_with='wandb')
    accelerator.print(os.environ)

    if not config.get('debug', False) and accelerator.is_main_process:
        prefix = config.get('prefix', '')
        modelname = config.get('model', '')
        accelerator.init_trackers(project_name='test-ilql', init_kwargs={'wandb': {'name': f'{prefix}-ilql-{task}-{modelname}', 'mode': 'disabled' if args.get('debug', False) else 'online'}}, config=config)
        config = wandb.config

    device = accelerator.device

    alpha, gamma, tau = config['alpha'], config['gamma'], config['tau']
    awac_scale, cql_scale = config['awac_scale'], config['cql_scale']

    if task == 'RandomWalks':
        from randomwalks import RandomWalks
        data = RandomWalks(seed=config['seed'])
        gptconfig = GPT2Config(**config['gptconfig'], vocab_size=data.n_nodes)
        model = QVModel(gptconfig, two_qs=config['two_qs'])

    elif task == 'Sentiments':
        from sentiments import Sentiments

        with accelerator.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(config['model'])
            data = Sentiments(tokenizer, needs_pipe=accelerator.is_main_process)

        with timeit('init model'):
            model = QVModel(config['model'], two_qs=config['two_qs'])

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

    gpt_blocks = list(model.gpt.transformer.h)[:-config['n_layers_unfrozen']]
    for m in gpt_blocks:
        for p in m.parameters():
            p.requires_grad = False

    train_dataloader = DataLoader(data.dataset, batch_size=config['batch_size'], shuffle=True)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    eval_batch_size = int(config['batch_size'] if len(data.eval_dataset) / world_size > config['batch_size'] else np.ceil(len(data.eval_dataset) / world_size))

    eval_dataloader = DataLoader(data.eval_dataset, eval_batch_size)
    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], config['lr'], config['opt_betas'])

    total_steps = int(config['n_epochs'] * np.ceil(len(data.dataset) / config['batch_size']) / world_size)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    n_opt_steps = 0

    with timeit('prepare'):
        model, opt, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
            model, opt, scheduler, train_dataloader, eval_dataloader
        )

    print(model(**accelerator.unwrap_model(model).dummy_inputs)[0].device)

    model.train()
    tbar = trange(total_steps, disable=not accelerator.is_local_main_process)

    for iepoch in range(config['n_epochs']):
        for batch in train_dataloader:
            batch_time = time()
            logs = {}
            if n_opt_steps % config['steps_for_eval'] == 0:
                model.eval()
                beta = config['inference_betas'][0]

                with timeit('eval'):
                    all_samples = []
                    for tokens in eval_dataloader:
                        tokens = tokens[0]
                        print(f'{tokens.shape=}')
                        with th.no_grad():
                            samples, stats = accelerator.unwrap_model(model).sample(
                                tokens,
                                beta=beta,
                                max_length=data.max_length,
                                logit_mask=data.logit_mask
                            )

                        all_samples.append(samples)
                        logs.update(stats)

                samples = accelerate.utils.gather(th.vstack(all_samples))
                print(samples)

                if accelerator.is_main_process:
                    print(f'{samples.shape=}')
                    reward, stats = data.eval(samples, beta)
                    logs.update(stats)
                    tbar.set_postfix(stats)

                accelerator.wait_for_everyone()

                if n_opt_steps > 0 and not config.get('debug', False):
                    accelerator.save_state(f'stash/state-{task.lower()}')

                model.train()

            tokens, attn, rewards = batch

            batch_time = time()
            actions = tokens[:, 1:, None]
            isterminal = attn[:, :-1]

            forward_time = time()
            logits, qs, target_qs, vs, _ = model(input_ids=tokens, attention_mask=attn)

            forward_time = time() - forward_time

            bsize, ntokens, dsize = logits.shape

            if config['two_qs']:
                Q1 = qs[0][:, :-1].gather(-1, actions).squeeze(-1)
                Q2 = qs[1][:, :-1].gather(-1, actions).squeeze(-1)

                targetQ1 = target_qs[0][:, :-1].gather(-1, actions).squeeze(-1).detach()
                targetQ2 = target_qs[1][:, :-1].gather(-1, actions).squeeze(-1).detach()
                targetQ = th.minimum(targetQ1, targetQ2)
            else:
                Q = qs[:, :-1].gather(-1, actions).squeeze(-1)
                targetQ = target_qs[:, :-1].gather(-1, actions).squeeze(-1).detach()

            n_nonterminal = max(1, isterminal.sum())
            V = vs[:, 1:].squeeze() * isterminal
            Q_ = rewards + gamma * V

            if config['two_qs']:
                loss_q1 = ((Q1 - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal
                loss_q2 = ((Q2 - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal
                loss_q = loss_q1 + loss_q2
            else:
                loss_q = ((Q - Q_.detach()) * isterminal).pow(2).sum() / n_nonterminal

            loss_v = (((targetQ >= V).int() * tau * (targetQ - V).pow(2) + (targetQ < V).int() * (1 - tau) * (targetQ - V).pow(2)) * isterminal).sum() / n_nonterminal

            if config['two_qs']:
                loss_cql_q1 = (F.cross_entropy(qs[0][:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal
                loss_cql_q2 = (F.cross_entropy(qs[1][:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal
                loss_cql = loss_cql_q1 + loss_cql_q2
            else:
                loss_cql = (F.cross_entropy(qs[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal

            loss_awac = (F.cross_entropy(logits[:, :-1].reshape(-1, dsize), actions.reshape(-1), reduction='none').reshape(bsize, ntokens-1) * isterminal).sum() / n_nonterminal

            loss = loss_q + loss_v + cql_scale * loss_cql + awac_scale * loss_awac

            backward_time = time()
            accelerator.backward(loss)
            backward_time = time() - backward_time

            opt.step()
            scheduler.step()
            opt.zero_grad()
            n_opt_steps += 1

            tokens_per_sec = tokens.numel() * world_size / (time() - batch_time)
            tbar.set_description(f'{loss_q=:.1f} {loss_v=:.2f} {loss_awac=:.1f} {backward_time=:.1f} {tokens_per_sec=:.2f}')
            tbar.update()

            if (n_opt_steps + 1) % config['steps_for_target_q_sync'] == 0:
                accelerator.unwrap_model(model).sync_target_q_heads(alpha)

            logs.update({k: v for k, v in locals().items() if k in ['loss', 'loss_v', 'loss_q', 'loss_cql', 'loss_awac']})
            logs['lr'] = scheduler.get_last_lr()[0]
            logs['forward_time'] = forward_time
            logs['backward_time'] = backward_time
            logs['target_sum'] = check_weights(accelerator.unwrap_model(model).target_q1_head)
            logs['batch_time'] = time() - batch_time

            if not config.get('debug', False):
                accelerator.log(logs)

    return model, data

if __name__ == '__main__':
    if os.environ.get('LOCAL_RANK'):
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']

    if run_from_ipython:
        args = {}
    else:
        # poor man's argparse
        args = {a[2:]: eval(v) for a, v in map(lambda s: s.split('='), sys.argv[1:])}

    main(**args)
