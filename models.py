import torch as th
import numpy as np
from torch import tensor, nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig
from typing import NamedTuple, Tuple, Union
from copy import deepcopy
from collections import defaultdict

def topk_mask(xs: th.FloatTensor, k: int):
    mintop = th.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)

class QVOutput(NamedTuple):
    logits: th.FloatTensor
    qs: th.FloatTensor
    target_qs: th.FloatTensor
    vs: th.FloatTensor
    past_key_values: Tuple[th.FloatTensor]

def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 4),
        nn.GELU(),
        nn.Linear(n_embd * 4, out)
    )

class QVModel(nn.Module):
    def __init__(self, config: Union[PretrainedConfig, str], two_qs=True):
        super().__init__()

        if isinstance(config, PretrainedConfig):
            self.gpt = AutoModelForCausalLM.from_config(config)
        elif config == 'EleutherAI/gpt-j-6B':
            self.gpt = AutoModelForCausalLM.from_pretrained(config, revision='float16', torch_dtype=th.float16)
        else:
            self.gpt = AutoModelForCausalLM.from_pretrained(config)

        self.two_qs = two_qs
        vocab_size = self.gpt.config.vocab_size
        n_embd = self.gpt.config.n_embd

        self.q1_head = make_head(n_embd, vocab_size)
        self.target_q1_head = deepcopy(self.q1_head)

        for params in self.target_q1_head.parameters():
            params.requires_grad = False

        if two_qs:
            self.q2_head = make_head(n_embd, vocab_size)
            self.target_q2_head = deepcopy(self.q2_head)

            for params in self.target_q2_head.parameters():
                params.requires_grad = False

        self.v_head = make_head(n_embd, 1)

    def forward(self, **x):
        out = self.gpt.transformer(**x)
        hs = out.last_hidden_state

        if self.two_qs:
            qs = (self.q1_head(hs), self.q2_head(hs))
            target_qs = (self.target_q1_head(hs), self.target_q2_head(hs))
        else:
            qs = self.q1_head(hs)
            target_qs = self.target_q1_head(hs)

        return QVOutput(self.gpt.lm_head(hs), qs, target_qs, self.v_head(hs), out.past_key_values)

    def sync_target_q(self, alpha):
        for target_param, copy_param in zip(self.target_q1_head.parameters(), self.q1_head.parameters()):
            target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

        if self.two_qs:
            for target_param, copy_param in zip(self.target_q2_head.parameters(), self.q2_head.parameters()):
                target_param.data.copy_((alpha * copy_param.data) + (1.0 - alpha) * target_param.data)

    @th.inference_mode()
    def sample(self, query, beta=1, max_length=32, temperature=1, top_k=20, logit_mask=None, logs=True, eos_token_id=50256):
        input = query.clone()
        past_key_values = None
        tensors = defaultdict(list)

        finished = th.zeros(input.shape[0], 1, dtype=th.long, device=query.device)

        for _ in range(max_length-1):
            out = self.forward(input_ids=input, past_key_values=past_key_values)

            if self.two_qs:
                qs = th.minimum(out.target_qs[0][:, -1], out.target_qs[1][:, -1])
            else:
                qs = out.target_qs[:, -1]

            logits = out.logits[:, -1]

            if logit_mask is not None:
                logits[th.where(logit_mask[input[:, -1]])] = -np.inf

            vs = out.vs[:, -1]

            adv = qs - vs
            pi = F.log_softmax(logits, -1)
            modpi = topk_mask(pi + beta * adv, top_k)
            ps = F.softmax(modpi / temperature, -1)

            tokens = th.multinomial(ps, 1)
            tokens = (1 - finished) * tokens + finished * eos_token_id

            query = th.hstack((query, tokens))

            input = tokens
            past_key_values = out.past_key_values
            finished = (tokens == eos_token_id).long()

            if logs:
                tensors['qs'].append(qs)
                tensors['vs'].append(vs)
                tensors['adv'].append(adv)

        stats = {}
        for name, xs in tensors.items():
            xs = th.vstack(xs)
            stats.update({
                f'{name}-min': xs.min(),
                f'{name}-max': xs.max(),
                f'{name}-std': xs.std(),
                f'{name}-avg': xs.mean(),
            })

        return query, stats

    @property
    def dummy_inputs(self):
        return {'input_ids': th.ones(1, 1, device=self.gpt.device, dtype=th.long)}

    @property
    def device(self):
        return self.gpt.device
