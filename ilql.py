import sys
import yaml
from time import time

import torch as th
from torch import tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2PreTrainedModel, GPT2Model
from accelerate import Accelerator

import wandb
from tqdm import trange
from utils import run_from_ipython
from models import QVModel
from copy import deepcopy

th.set_printoptions(sci_mode=False)
th.manual_seed(1000)

def main():
    if run_from_ipython:
        args = {}
    else:
        # poor man's argparse
        args = {a[2:]: eval(v) for a, v in map(lambda s: s.split('='), sys.argv[1:])}

    task = args['task'] if 'task' in args else 'RandomWalks'
    config = yaml.safe_load(open('config.yaml'))[task]
    config.update(args)

    accelerator = Accelerator(log_with='wandb')
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name='test-ilql', init_kwargs={'name': f'ilql-{task}'}, config=config)
        config = wandb.config

    device = accelerator.device

    alpha, gamma, tau = config['alpha'], config['gamma'], config['tau']
    steps_for_eval, steps_for_target_q_sync = config['steps_for_target_q_sync'], config['steps_for_eval']
    cql_scale = config['cql_scale']

    if task == 'RandomWalks':
        from randomwalks import RandomWalks
        data = RandomWalks(seed=config['seed'])
        model = QVModel(GPT2Config(**config['gptconfig'], vocab_size=data.n_nodes), two_qs=config['two_qs']).to(device)

    elif task == 'Sentiments':
        from sentiments import Sentiments

        with accelerator.main_process_first():
            data = Sentiments(use_cache=True)

        model = QVModel.from_pretrained(config['model'], two_qs=config['two_qs']).to(device)
        gpt_blocks = list(model.transformer.h)[:-config['n_layers_unfrozen']]
        for m in gpt_blocks:
            for p in m.parameters():
                p.requires_grad = False

    elif task == 'Carps':
        from carps import Carps
        data = Carps(max_length=config['max_length'], diff_reward=config['diff_reward'])

        model = QVModel.from_pretrained(config['model'], two_qs=config['two_qs']).to(device)
        gpt_blocks = list(model.transformer.h)[:-config['n_layers_unfrozen']]
        for m in gpt_blocks:
            for p in m.parameters():
                p.requires_grad = False
    else:
        raise ValueError(f'nonexistent {task=}')

    dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], config['lr'], config['opt_betas'])
    n_opt_steps = 0

    model, opt, dataloader = accelerator.prepare(model, opt, dataloader)
    print(model(input_ids=th.ones(1, 1).long().to(model.device))[-1].device)

    tbar = trange(config['n_epochs'], disable=not accelerator.is_local_main_process)

    if accelerator.is_local_main_process:
        data.eval({}, model, two_qs=config['two_qs'], betas=[0, 1])

    start_time = time()

    for iepoch in tbar:
        for input, attn, rewards in dataloader:
            logits, qs, target_qs, vs = model(input_ids=input, attention_mask=attn)
            bsize, ntokens, dsize = logits.shape

            if config['two_qs']:
                Q1 = qs[0][:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1)
                Q2 = qs[1][:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1)
                targetQ1 = target_qs[0][:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1).detach()
                targetQ2 = target_qs[1][:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1).detach()
                targetQ = th.minimum(targetQ1, targetQ2)
            else:
                Q = qs[:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1)
                targetQ = target_qs[:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1).detach()

            n_nonterminal = max(1, attn[:, :-1].sum())
            V = vs[:, 1:].squeeze() * attn[:, 1:]
            Q_ = rewards + gamma * V

            if config['two_qs']:
                loss_q1 = ((Q1 - Q_.detach()) * attn[:, :-1]).pow(2).sum() / n_nonterminal
                loss_q2 = ((Q2 - Q_.detach()) * attn[:, :-1]).pow(2).sum() / n_nonterminal
                loss_q = loss_q1 + loss_q2
            else:
                loss_q = ((Q - Q_.detach()) * attn[:, :-1]).pow(2).sum() / n_nonterminal

            loss_v = (((targetQ >= V).int() * tau * (targetQ - V).pow(2) +
                       (targetQ < V).int() * (1 - tau) * (targetQ - V).pow(2)) * attn[:, 1:]).sum() / n_nonterminal

            if config['two_qs']:
                loss_cql_q1 = (F.cross_entropy(qs[0][:, :-1, :].reshape(-1, dsize), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal
                loss_cql_q2 = (F.cross_entropy(qs[1][:, :-1, :].reshape(-1, dsize), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal
                loss_cql = loss_cql_q1 + loss_cql_q2
            else:
                loss_cql = (F.cross_entropy(qs[:, :-1, :].reshape(-1, dsize), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal

            loss_awac = (F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal

            loss = loss_q + loss_v + loss_awac + cql_scale * loss_cql

            checksum = model.target_q1_head.state_dict()['2.weight'].sum()

            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            n_opt_steps += 1

            assert model.target_q1_head.state_dict()['2.weight'].sum() == checksum

            tbar.set_description(f'{loss_q=:.1f} {loss_v=:.2f} {loss_cql=:.1f} {loss_awac=:.1f}')

            if (n_opt_steps + 1) % steps_for_target_q_sync == 0:
                accelerator.wait_for_everyone()
                accelerator.unwrap_model(model).sync_target_q(alpha)

            if (n_opt_steps + 1) % steps_for_eval == 0:
                logs = {k: v for k, v in locals().items() if k in ['loss', 'loss_v', 'loss_q', 'loss_cql', 'loss_awac']}

                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    _, stats = data.eval(logs, model, two_qs=config['two_qs'], betas=config['inference_betas'])
                    tbar.set_postfix(stats)

                logs['epoch_time'] = time() - start_time
                accelerator.log(logs)
                start_time = time()

    if accelerator.is_local_main_process:
        accelerator.wait_for_everyone()
        accelerator.log({'target': data.eval({}, model, two_qs=config['two_qs'], betas=[1])[0]})

if __name__ == '__main__':
    main()
