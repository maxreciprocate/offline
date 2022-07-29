import numpy as np
from numpy.random import randint
from tqdm import tqdm, trange
from matplotlib import pyplot

import torch as th
from torch import tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
from randomwalks import RandomWalks

th.set_printoptions(sci_mode=False)
th.manual_seed(1000)
device = th.device('cpu')

class QVModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.transformer = GPT2Model(gptconfig)

        self.q_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.vocab_size)
        )

        self.v_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, 1)
        )

    def forward(self, **x):
        hs = self.transformer(**x)[0]
        return self.q_head(hs), self.v_head(hs)

class PIModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.transformer = GPT2Model(config)

        self.pi_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.vocab_size)
        )

    def forward(self, **x):
        return self.pi_head(self.transformer(**x)[0])

# walks = RandomWalks(seed=4, walksize=10, nnodes=10, pedge=1, nwalks=100)
# walks = RandomWalks(seed=4, walksize=9, nnodes=3, pedge=1, nwalks=100)
# walks = RandomWalks(seed=100)
walks = RandomWalks()

ds = DataLoader(walks, batch_size=500, shuffle=True)

gptconfig = GPT2Config(
    vocab_size=walks.nnodes+1,
    n_embd=72*2,
    n_layer=4,
    resid_pdrop=0,
    embd_pdrop=0,
    attn_pdrop=0,
)

tau = 0.8
beta = 8

master = QVModel(gptconfig)
puppet = QVModel(gptconfig)
pi_beta = PIModel(gptconfig)

qv_opt = th.optim.Adam(puppet.parameters(), 3e-4)
pi_opt = th.optim.Adam(pi_beta.parameters(), 3e-4)

print(f'Finetuning...')
tbar = trange(10)
for _ in tbar:
    for s, attn, _ in ds:
        logits = pi_beta(input_ids=s, attention_mask=attn)

        logits = logits[:, :-1]
        s = s[:, 1:]

        pi_opt.zero_grad()
        pi_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), s.flatten())
        pi_loss.backward()
        pi_opt.step()

        tbar.set_description(f'{pi_loss=:.2f}')

    with th.no_grad():
        nsolved = 0
        actn = 0

        for start in set(range(1, walks.nnodes)):
            path = [start]
            for _ in range(walks.walksize-1):
                logits = pi_beta(input_ids=tensor(path))[-1]
                logits[:walks.nnodes][np.where(~walks.adj[path[-1]])[0]] = -np.inf
                logits[-1] = -np.inf

                ps = F.softmax(logits, dim=-1)
                # step = th.multinomial(ps, 1)
                step = ps.argmax()

                path.append(step)
                if step == walks.goal:
                    nsolved += 1
                    break

            actn += len(path) / (walks.nnodes-1)

        current = (walks.worstlen - actn)/(walks.worstlen - walks.bestlen)
        average = (walks.worstlen - walks.avglen)/(walks.worstlen - walks.bestlen)

        tbar.set_postfix({
            'arrived':f'{nsolved / (walks.nnodes-1) * 100:.0f}%',
            'optimal': f'{current*100:.0f}% > {average*100:.0f}%'})

print(f'Steering...')
tbar = trange(10)
for _ in tbar:
    for s, attn, rs in ds:
        puppet_qs, vs = puppet(input_ids=s, attention_mask=attn)
        target_qs, _ = master(input_ids=s, attention_mask=attn)

        pQ = puppet_qs[:, :-1, :].gather(-1, s[:, 1:, None]).squeeze(-1)
        tQ = target_qs[:, :-1, :].gather(-1, s[:, 1:, None]).squeeze(-1)
        Q = th.minimum(pQ, tQ)

        V = vs[:, 1:].squeeze()
        Q_ = rs + 0.99 * V * attn[:, 1:]

        qv_opt.zero_grad()
        loss_q = (pQ - Q_.detach()).pow(2).mean()

        tQ = tQ.detach()
        loss_v = (((tQ >= V).int() * tau * (tQ - V).pow(2) +
                   (tQ < V).int() * (1 - tau) * (tQ - V).pow(2)) * attn[:, 1:]).mean()

        loss = loss_q + loss_v
        loss.backward()

        nn.utils.clip_grad_norm_(puppet.parameters(), 1)
        qv_opt.step()

        tbar.set_description(f'{loss_q=:.1f} {loss_v=:.1f}')

    master.load_state_dict(puppet.state_dict())

    with th.no_grad():
        nsolved = 0
        actn = 0

        for start in set(range(1, walks.nnodes)):
            path = [start]
            for _ in range(walks.walksize-1):
                qs, vs = puppet(input_ids=tensor(path))
                logits = pi_beta(input_ids=tensor(path))[-1]

                mod = beta * (qs[-1])

                pi = F.log_softmax(logits, dim=-1)
                modpi = F.log_softmax(pi + mod, dim=-1)

                unreachable = list(
                    np.where(~walks.adj[path[-1]])[0]) + [walks.nnodes]

                pi[unreachable] = -np.inf
                modpi[unreachable] = -np.inf
                qs[-1][unreachable] = -np.inf

                step = modpi.argmax()

                path.append(step.item())
                if step == walks.goal:
                    nsolved += 1
                    break

            actn += len(path) / (walks.nnodes-1)

        current = (walks.worstlen - actn)/(walks.worstlen - walks.bestlen)
        average = (walks.worstlen - walks.avglen)/(walks.worstlen - walks.bestlen)

        tbar.set_postfix({
            'arrived':f'{nsolved / (walks.nnodes-1) * 100:.0f}%',
            'optimal': f'{current*100:.0f}% > {average*100:.0f}%'})
