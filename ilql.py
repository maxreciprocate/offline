import sys
import torch as th
from torch import tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
import wandb
from tqdm import trange

th.set_printoptions(sci_mode=False)
th.manual_seed(1000)

dataname = sys.argv[1] if len(sys.argv) > 1 else 'RandomWalks'
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
wandb.init(name=f'ilql-{dataname}', project='ilql', mode='online')

class QVModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ['attn.masked_bias']

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        self.q_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.vocab_size)
        )

        self.target_q_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.vocab_size)
        )

        self.v_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, 1)
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, **x):
        hs = self.transformer(**x)[0]
        return self.lm_head(hs), self.q_head(hs), self.target_q_head(hs), self.v_head(hs)

    def sync_target_q(self, alpha):
        for target_param, copy_param in zip(self.target_q_head.parameters(), self.q_head.parameters()):
            target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

lr = 3e-4
batch_size = 384
tau = 0.5
cql_scale = 0.25
steps_for_target_q_sync = 1000

wandb.config.update({k: v for k, v in locals().items() if k in ['tau', 'lr', 'batch_size', 'cql_scale', 'steps_for_target_q_sync']})

if dataname == 'RandomWalks':
    from randomwalks import RandomWalks
    data = RandomWalks(seed=1000)

    config = GPT2Config(
        vocab_size=data.nnodes,
        n_embd=72,
        n_layer=2,
        n_head=1,
    )

    model = QVModel(config).to(device)

else:
    from sentiments import Sentiments
    data = Sentiments()

    model = QVModel.from_pretrained('gpt2').to(device)

ds = DataLoader(data, batch_size=batch_size, shuffle=True)
opt = th.optim.Adam(model.parameters(), lr)
n_opt_steps = 0

tbar = trange(50)
data.eval({}, tbar, model, beta=1)

for iepoch in tbar:
    for input, attn, rewards in ds:
        input = input.to(device)
        attn = attn.to(device)
        rewards = rewards.to(device)

        logits, qs, target_qs, vs = model(input_ids=input, attention_mask=attn)

        Q = qs[:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1)
        targetQ = target_qs[:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1).detach()

        V = vs[:, 1:].squeeze() * attn[:, 1:]
        Q_ = rewards + V

        loss_q = (Q - Q_.detach()).pow(2).mean()

        loss_v = (((targetQ >= V).int() * tau * (targetQ - V).pow(2) +
                   (targetQ < V).int() * (1 - tau) * (targetQ - V).pow(2)) * attn[:, 1:]).mean()

        loss_cql = cql_scale * F.cross_entropy(qs[:, :-1, :].reshape(-1, qs.size(-1)), input[:, 1:, None].reshape(-1))
        loss_awac = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), input[:, 1:].reshape(-1))

        loss = loss_q + loss_v + loss_cql + loss_awac

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        n_opt_steps += 1

        if n_opt_steps > 0 and n_opt_steps % steps_for_target_q_sync == 0:
            model.sync_target_q(1)

        tbar.set_description(f'{loss_q=:.1f} {loss_v=:.1f} {loss_cql=:.1f} {loss_awac=:.1f}')

    logs = {k: v for k, v in locals().items() if k in ['loss', 'loss_v', 'loss_q', 'loss_cql', 'loss_awac']}

    data.eval(logs, tbar, model, beta=0)
    data.eval(logs, tbar, model, beta=1)
    data.eval(logs, tbar, model, beta=4)
    data.eval(logs, tbar, model, beta=8)

    th.save(model.state_dict(), 'stash/model.pt')
    wandb.log(logs)
