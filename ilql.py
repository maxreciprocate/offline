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

import wandb
from tqdm import tqdm
from utils import run_from_ipython
from models import QVModel, QTargetHeads
from copy import deepcopy
import accelerate

th.set_printoptions(sci_mode=False)
th.manual_seed(1000)

def main(**args):
    task = args['task'] if 'task' in args else 'RandomWalks'
    config = yaml.safe_load(open('config.yaml'))[task]
    config.update(args)

    accelerator = Accelerator(log_with='wandb')
    if not config.get('debug', False) and accelerator.is_main_process:
        prefix = config.get('prefix', '')
        modelname = config.get('model', '')
        accelerator.init_trackers(project_name='test-ilql', init_kwargs={'wandb': {'name': f'{prefix}-ilql-{task}-{modelname}', 'mode': 'disabled' if args.get('debug', False) else 'online'}}, config=config)
        config = wandb.config

    device = accelerator.device

    alpha, gamma, tau = config['alpha'], config['gamma'], config['tau']
    steps_for_eval, steps_for_target_q_sync = config['steps_for_target_q_sync'], config['steps_for_eval']
    awac_scale, cql_scale = config['awac_scale'], config['cql_scale']

    if task == 'RandomWalks':
        from randomwalks import RandomWalks
        data = RandomWalks(seed=config['seed'])
        gptconfig = GPT2Config(**config['gptconfig'], vocab_size=data.n_nodes)
        model = QVModel(gptconfig, two_qs=config['two_qs'])

    elif task == 'Sentiments':
        from sentiments import Sentiments

        tokenizer = AutoTokenizer.from_pretrained(config['model'])
        with accelerator.main_process_first():
            data = Sentiments(tokenizer, batch_size=config['batch_size'], need_pipe=accelerator.is_main_process)

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

    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], config['lr'], config['opt_betas'])
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt, len(data))
    n_opt_steps = 0

    target_q_heads = QTargetHeads(model.n_embd, model.vocab_size, two_qs=config['two_qs'])
    model, opt, dataloader, scheduler = accelerator.prepare(model, opt, dataloader, scheduler)

    print(model(**model.dummy_inputs)[0].device)

    target_q_heads = target_q_heads.to(device).bfloat16()
    model.train()

    tbar = tqdm(dataloader, disable=not accelerator.is_main_process)

    for tokens, attn, rewards in tbar:
        batch_time = time()
        actions = tokens[:, 1:, None]
        isterminal = attn[:, :-1]

        forward_time = time()
        logits, qs, vs, hs, _ = model(input_ids=tokens, attention_mask=attn)
        target_qs = target_q_heads(hs)

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

        loss_v = (((targetQ >= V).int() * tau * (targetQ - V).pow(2) +
                   (targetQ < V).int() * (1 - tau) * (targetQ - V).pow(2)) * isterminal).sum() / n_nonterminal

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

        tbar.set_description(f'{loss_q=:.1f} {loss_v=:.2f} {loss_cql=:.1f} {loss_awac=:.1f} {backward_time=:.1f}')

        if (n_opt_steps + 1) % steps_for_target_q_sync == 0:
            if accelerator.is_main_process:
                model.sync_target_q_heads(target_q_heads, alpha)


        logs = {k: v for k, v in locals().items() if k in ['loss', 'loss_v', 'loss_q', 'loss_cql', 'loss_awac']}
        logs['lr'] = scheduler.get_last_lr()[0]
        logs['batch_time'] = time() - batch_time
        logs['forward_time'] = forward_time
        logs['backward_time'] = backward_time
        batch_time = time()

        if (n_opt_steps + 1) % steps_for_eval == 0:
            model.eval()
            _, stats = data.eval(logs, model, target_q_heads, betas=config['inference_betas'])
            model.train()
            tbar.set_postfix(stats)
            # accelerator.save_state(f'stash/state-{task.lower()}')

        if not config.get('debug', False):
            accelerator.log(logs)

    accelerator.wait_for_everyone()
    model.eval()
    target = data.eval({}, model, target_q_heads, betas=[1])[0]

    if not config.get('debug', False):
        accelerator.log({'target': target})

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
