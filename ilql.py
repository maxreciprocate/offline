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

class Model(nn.Module):
    def __init__(self, vocab_size=20):
        super().__init__()

        self.nhidden = 72*2
        self.transformer = GPT2Model(GPT2Config(
            vocab_size=vocab_size,
            n_layer=4,
            n_embd=self.nhidden,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))

        self.q1_head = nn.Sequential(
            nn.Linear(self.nhidden, self.nhidden * 2),
            nn.ReLU(),
            nn.Linear(self.nhidden * 2, vocab_size)
        )

        self.q2_head = nn.Sequential(
            nn.Linear(self.nhidden, self.nhidden * 2),
            nn.ReLU(),
            nn.Linear(self.nhidden * 2, vocab_size)
        )

        self.v_head = nn.Sequential(
            nn.Linear(self.nhidden, self.nhidden * 2),
            nn.ReLU(),
            nn.Linear(self.nhidden * 2, 1)
        )

    def forward(self, **x):
        hs = self.transformer(**x)[0]
        return self.q1_head(hs), self.q2_head(hs), self.v_head(hs)

# walks = RandomWalks(seed=100, walksize=10, nnodes=3, pedge=1, nwalks=100)
walks = RandomWalks()

ds = DataLoader(walks, batch_size=500, shuffle=True)
master = Model(vocab_size=walks.nnodes+1)
puppet = Model(vocab_size=walks.nnodes+1)

opt = th.optim.Adam(puppet.parameters(), 3e-4)

tbar = trange(10)
for _ in tbar:
    for s, attn, rs in ds:
        *qs, _ = puppet(input_ids=s, attention_mask=attn)
        *_, vs = master(input_ids=s, attention_mask=attn)

        Q1 = qs[0][:, :-1, :].gather(-1, s[:, 1:, None]).squeeze(-1)
        Q2 = qs[1][:, :-1, :].gather(-1, s[:, 1:, None]).squeeze(-1)
        Q = th.minimum(Q1, Q2)

        V = vs[:, 1:].squeeze()
        Q_ = rs + 0.99 * V * attn[:, 1:]

        opt.zero_grad()
        loss_q = (Q - Q_.detach()).pow(2).mean()
        loss_v = (Q - V).pow(2).mean()
        loss = loss_q + loss_v
        loss.backward()

        nn.utils.clip_grad_norm_(puppet.parameters(), 1)
        opt.step()

        tbar.set_description(f'{loss_q=:.1f} {loss_v=:.1f}')

    master.load_state_dict(puppet.state_dict())

    with th.no_grad():
        nsolved = 0
        actn = 0

        for start in set(range(1, walks.nnodes)):
            path = [start]
            for _ in range(walks.walksize-1):
                *qs, _ = puppet(input_ids=tensor(path))
                qs = th.minimum(*qs)

                # cql with hands
                qs[-1, :walks.nnodes][np.where(~walks.adj[path[-1]])[0]] = -np.inf
                qs[-1, -1] = -np.inf

                step = qs[-1].argmax().item()
                path.append(step)
                if step == walks.goal:
                    nsolved += 1
                    break

            actn += len(path) / (walks.nnodes-1)

        current = (walks.worstlen - actn)/(walks.worstlen - walks.bestlen)
        average = (walks.worstlen - walks.avglen)/(walks.worstlen - walks.bestlen)

        tbar.set_postfix({'arrived':f'{nsolved / (walks.nnodes-1) * 100:.0f}%',
                          'optimal': f'{current*100:.0f}% > {average*100:.0f}%'})
