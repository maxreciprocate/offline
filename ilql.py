import torch as th
from torch import tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
import wandb
from tqdm import trange

th.set_printoptions(sci_mode=False)
th.manual_seed(1000)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
wandb.init(name='ilql-sentiment-gpt2', project='ilql')

class QVModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

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

        self.init_weights()

    def forward(self, **x):
        hs = self.transformer(**x)[0]
        return self.q_head(hs), self.v_head(hs)

class PIModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)

        self.pi_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, config.vocab_size)
        )

        self.init_weights()

    def forward(self, **x):
        return self.pi_head(self.transformer(**x)[0])

tau = 0.8
beta = 4
gamma = 1
lr = 5e-5
bsize = 150

pi_beta = PIModel.from_pretrained('gpt2').to(device)
pi_opt = th.optim.Adam(pi_beta.parameters(), lr)

master = QVModel.from_pretrained('gpt2').to(device)
puppet = QVModel.from_pretrained('gpt2').to(device)
qv_opt = th.optim.Adam(puppet.parameters(), lr)

from sentiments import Sentiments
sentiments = Sentiments()
ds = DataLoader(sentiments, batch_size=bsize, shuffle=True)

print(f'Finetuning...')
tbar = trange(20)
for _ in tbar:
    logs = {}
    reward = sentiments.eval(logs, pi_beta)
    tbar.set_postfix({'reward': f'{reward:.2f}'})
    wandb.log(logs)

    for input, attn, _ in ds:
        input = input.to(device)
        attn = attn.to(device)

        logits = pi_beta(input_ids=input, attention_mask=attn)
        logits = logits[:, :-1]
        input = input[:, 1:]

        pi_opt.zero_grad()
        pi_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), input.flatten())
        pi_loss.backward()
        pi_opt.step()

        tbar.set_description(f'{pi_loss=:.2f}')

    th.save(pi_beta.state_dict(), 'stash/pi_model.pt')

print(f'Steering...')
tbar = trange(30)
for iepoch in tbar:
    logs = {}
    reward = sentiments.eval(logs, pi_beta, puppet, beta=beta)
    tbar.set_postfix({'reward': f'{reward:.2f}'})
    wandb.log(logs)

    for input, attn, rewards in ds:
        input = input.to(device)
        attn = attn.to(device)
        rewards = rewards.to(device)

        puppet_qs, vs = puppet(input_ids=input, attention_mask=attn)
        with th.no_grad():
            target_qs, _ = master(input_ids=input, attention_mask=attn)

        pQ = puppet_qs[:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1)
        tQ = target_qs[:, :-1, :].gather(-1, input[:, 1:, None]).squeeze(-1)

        V = vs[:, 1:].squeeze() * attn[:, 1:]
        Q_ = rewards + gamma * V

        qv_opt.zero_grad()
        loss_q = (pQ - Q_.detach()).pow(2).mean()

        loss_v = (((tQ >= V).int() * tau * (tQ - V).pow(2) +
                   (tQ < V).int() * (1 - tau) * (tQ - V).pow(2)) * attn[:, 1:]).mean()

        loss_cql = 0.5 * F.cross_entropy(
            puppet_qs[:, :-1, :].reshape(-1, puppet_qs.size(-1)),
            input[:, 1:, None].flatten())

        loss = loss_q + loss_v + loss_cql
        loss.backward()

        nn.utils.clip_grad_norm_(puppet.parameters(), 1)
        qv_opt.step()

        tbar.set_description(f'{loss_q=:.1f} {loss_v=:.1f} {loss_cql=:.1f}')

    master.load_state_dict(puppet.state_dict())
    th.save(puppet.state_dict(), 'stash/qv_model.pt')

logs = {}
sentiments.eval(logs, pi_beta, puppet, beta=beta)
wandb.log(logs)
