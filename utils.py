import numpy as np
import torch as th

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

def logvars(name, logs, xs):
    xs = th.vstack(xs)
    logs.update({ f'{name}-mean': xs.mean(),
                  f'{name}-std': xs.std(),
                  f'{name}-min': xs.min(),
                  f'{name}-max': xs.max() })

def topk_mask(xs, k):
    mintop = th.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)

def sizesplit(size: int, xs):
    for ind in range(len(xs) // size + 1):
        yield xs[ind*size:min(len(xs), (ind+1)*size)]

def flatten(xs):
    return sum(xs, [])

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
