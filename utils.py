import os
import numpy as np
import torch as th
from tqdm import tqdm
import math
from contextlib import contextmanager
import torch.nn.functional as F
from time import time
import deepspeed

try:
    __IPYTHON__
    run_from_ipython = True
except NameError:
    run_from_ipython = False

def randexclude(rng: np.random.RandomState, n: int, exclude: int) -> int:
    while True:
        x = rng.randint(n)
        if x != exclude:
            return x

def tohuman(n: int) -> str:
    if n > 1e9:
        return f'{n / 1e9:.1f}B'
    elif n > 1e6:
        return f'{n / 1e6:.1f}M'
    elif n > 1e3:
        return f'{n / 1e3:.1f}K'
    return str(n)

def logvars(name, logs, xs):
    xs = th.vstack(xs)
    logs.update({ f'{name}-mean': xs.mean(),
                  f'{name}-std': xs.std(),
                  f'{name}-min': xs.min(),
                  f'{name}-max': xs.max() })

def batch_map(fn, xs, bsize: int, desc=None):
    out = []
    for ind in tqdm(range(math.ceil(len(xs) / bsize)), desc=desc, disable=not desc):
        batch = xs[ind*bsize:min(len(xs), (ind+1)*bsize)]
        out.extend(fn(batch))

    return out

def load_tensors(name, texts, reward_model, tokenizer, max_length=64, use_cache=True):
    cache_path = f'cache/{name}_{max_length=}_tokenizer={tokenizer.name_or_path.split("/")[-1]}.pt'
    if use_cache and os.path.exists(cache_path):
        tensors = th.load(cache_path)
    else:
        tensors = tokenizer(
            [tokenizer.bos_token + x for x in texts],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        trimmed_texts = tokenizer.batch_decode(tensors['input_ids'], skip_special_tokens=True)
        rewards = th.as_tensor(reward_model(trimmed_texts))
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-30)
        rewards = rewards.view(-1, 1).repeat(1, tensors['input_ids'].shape[1])
        rewards[tensors['attention_mask'].eq(0)] = 0

        tensors['rewards'] = rewards
        tensors['attention_mask'] = F.pad(tensors['attention_mask'], (0, 1), value=0)
        tensors['input_ids'] = F.pad(tensors['input_ids'], (0, 1), value=tokenizer.eos_token_id)

        if not os.path.exists(os.path.dirname(cache_path)):
            os.mkdir(os.path.dirname(cache_path))

        th.save(tensors, cache_path)

    print(f"{tohuman(np.prod(tensors['input_ids'].shape))} tokens")
    return tensors


def isdelim(c: str):
    return c == '?' or c == '!' or c == '.' or c == ';'

def pprint(s):
    trig = False
    si = 0
    l = len(s)-1

    for i in range(len(s)):
        if i == l:
            print(s[si:].strip())

        elif trig or isdelim(s[i]):
            trig = True

            if s[i].isspace():
                print(s[si:i+1].strip())
                si = i + 1
                trig = False

@contextmanager
def timeit(desc='something important'):
    print(f'{desc}...')
    stime = time()
    try:
        yield None
    finally:
        print(f'done with {desc.lower()} in {time() - stime:.1f}s')

def check_weights(param):
    if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') == '3':
        with deepspeed.zero.GatheredParameters(param[0].weight, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                return param[0].weight.sum()
    else:
        return param[0].weight.sum()
