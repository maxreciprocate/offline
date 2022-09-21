import os
import numpy as np
import torch as th
from tqdm import tqdm
import math
from contextlib import contextmanager
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
