import sys
import yaml
from time import time

import torch as th
from torch import tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
from accelerate import Accelerator

import wandb
from tqdm import trange
from utils import run_from_ipython

th.set_printoptions(sci_mode=False)
th.manual_seed(1000)

if run_from_ipython:
    args = {}
else:
    # poor man's argparse
    args = {a[2:]: eval(v) for a, v in map(lambda s: s.split('='), sys.argv[1:])}

task = args['task'] if 'task' in args else 'RandomWalks'
config = yaml.safe_load(open('config.yaml'))[task]
config.update(args)

wandb.init(name=f'ilql-{task}', project='test-ilql', mode='online', config=config)
config = wandb.config
locals().update(config)

accelerator = Accelerator()
device = accelerator.device

class QVModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ['attn.masked_bias']

    def __init__(self, config: GPT2Config, two_qs=True):
        super().__init__(config)
        self.two_qs = two_qs

        def make_head(out):
            return nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd * 2),
                nn.ReLU(),
                nn.Linear(config.n_embd * 2, out)
            )

        self.transformer = GPT2Model(config)
        self.q1_head = make_head(config.vocab_size)
        self.target_q1_head = make_head(config.vocab_size)

        if two_qs:
            self.q2_head = make_head(config.vocab_size)
            self.target_q2_head = make_head(config.vocab_size)

        self.lm_head = make_head(config.vocab_size)
        self.v_head = make_head(1)

        self.init_weights()

    def forward(self, **x):
        hs = self.transformer(**x)[0]

        if self.two_qs:
            qs = (self.q1_head(hs), self.q2_head(hs))
            target_qs = (self.target_q1_head(hs), self.target_q2_head(hs))
        else:
            qs = self.q1_head(hs)
            target_qs = self.target_q1_head(hs)

        return self.lm_head(hs), qs, target_qs, self.v_head(hs)

    def sync_target_q(self, alpha):
        for target_param, copy_param in zip(self.target_q1_head.parameters(), self.q1_head.parameters()):
            target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

        if self.two_qs:
            for target_param, copy_param in zip(self.target_q2_head.parameters(), self.q2_head.parameters()):
                target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

if task == 'RandomWalks':
    from randomwalks import RandomWalks
    data = RandomWalks(seed=config['seed'])
    model = QVModel(GPT2Config(**config['gptconfig'], vocab_size=data.n_nodes), two_qs=config['two_qs']).to(device)

elif task == 'Sentiments':
    from sentiments import Sentiments
    data = Sentiments()
    model = QVModel.from_pretrained(config['model']).to(device)

elif task == 'Carps':
    from carps import Carps
    data = Carps(max_length=config['max_length'], diff_reward=config['diff_reward'])

    model = QVModel.from_pretrained(config['model']).to(device)
    gpt_blocks = list(model.transformer.h)[:-config['n_layers_unfrozen']]
    for m in gpt_blocks:
        for p in m.parameters():
            p.requires_grad = False

else:
    raise ValueError(f'nonexistent {task=}')

dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
opt = th.optim.Adam(model.parameters(), config['lr'], config['opt_betas'])
n_opt_steps = 0

model, opt, dataloader = accelerator.prepare(model, opt, dataloader)

tbar = trange(config['n_epochs'], disable=not accelerator.is_local_main_process)
data.eval({}, model, betas=[0, 1])

for iepoch in tbar:
    start_time = time()

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
                   (targetQ < V).int() * (1 - tau) * (targetQ - V).pow(2)) * attn[:, :-1]).sum() / n_nonterminal

        if config['two_qs']:
            loss_cql_q1 = (F.cross_entropy(qs[0][:, :-1, :].reshape(-1, dsize), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal
            loss_cql_q2 = (F.cross_entropy(qs[1][:, :-1, :].reshape(-1, dsize), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal
            loss_cql = loss_cql_q1 + loss_cql_q2
        else:
            loss_cql = (F.cross_entropy(qs[:, :-1, :].reshape(-1, dsize), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal

        loss_awac = (F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), input[:, 1:].reshape(-1), reduction='none').reshape(bsize, ntokens-1) * attn[:, :-1]).sum() / n_nonterminal

        loss = loss_q + loss_v + loss_awac + cql_scale * loss_cql

        opt.zero_grad()
        accelerator.backward(loss)
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        n_opt_steps += 1

        tbar.set_description(f'{loss_q=:.1f} {loss_v=:.1f} {loss_cql=:.1f} {loss_awac=:.1f}')

        if (n_opt_steps + 1) % steps_for_target_q_sync == 0:
            model.sync_target_q(alpha)

        if (n_opt_steps + 1) % steps_for_eval == 0:
            logs = {k: v for k, v in locals().items() if k in ['loss', 'loss_v', 'loss_q', 'loss_cql', 'loss_awac']}

            _, stats = data.eval(logs, model, betas=config['inference_betas'])
            tbar.set_postfix(stats)

            if accelerator.is_local_main_process and task != 'RandomWalks':
                th.save(model.state_dict(), 'stash/model.pt')

            logs['epoch_time'] = time() - start_time
            wandb.log(logs)

wandb.log({'target': data.eval({}, model, betas=[1])[0]})
