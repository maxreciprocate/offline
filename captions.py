import os
import numpy as np
import torch as th
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import wandb
import sqlite3

from utils import batch_map, tohuman

def load_tensors(cache_path, tokenizer, use_cache=True):
    cache_path = f'{cache_path}_tokenizer={tokenizer.name_or_path}.pt'

    if use_cache and os.path.exists(cache_path):
        out = th.load(cache_path)
    else:
        conn = sqlite3.connect('sac_public_2022_06_29.sqlite')
        c = conn.cursor()
        c.execute("SELECT prompt, rating FROM ratings "
                  "JOIN images ON images.id=ratings.iid "
                  "JOIN generations ON images.gid=generations.id "
                  "WHERE rating IS NOT NULL;")

        prompts, ratings = tuple(map(list, zip(*c.fetchall())))
        out = tokenizer(prompts, padding=True, return_tensors='pt')

        # append eos
        input_ids = F.pad(out['input_ids'], (0, 1), value=tokenizer.eos_token_id)
        attention_mask = F.pad(out['attention_mask'], (0, 1), value=0)

        # figure out sentences' ending indices
        diff_padding = th.zeros(input_ids.shape[0], 1, dtype=th.long)
        endings = input_ids.eq(tokenizer.eos_token_id).diff(prepend=diff_padding, dim=-1).nonzero(as_tuple=True)

        rewards = th.zeros(input_ids.shape)
        rewards[endings] = tensor(ratings, dtype=th.float32)

        # prepend bos
        input_ids = F.pad(input_ids, (1, 0), value=tokenizer.eos_token_id)
        attention_mask = F.pad(attention_mask, (1, 0), value=1)

        out = {'input_ids': input_ids, 'attention_mask': attention_mask, 'rewards': rewards}

        if not os.path.exists(os.path.dirname(cache_path)):
            os.mkdir(os.path.dirname(cache_path))

        th.save(out, cache_path)

    print(f"Total {tohuman(np.prod(out['input_ids'].shape))} tokens")
    return out

class AestheticCaptions(TensorDataset):
    def __init__(self, tokenizer, max_length=77, n_samples=32, batch_size=1, use_cache=True):
        self.max_length = max_length
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        tensors = load_tensors('cache/aesthetic-captions', self.tokenizer, use_cache=use_cache)
        super().__init__(tensors['input_ids'], tensors['attention_mask'], tensors['rewards'])

    def eval(self, logs, model, betas=[1]):
        query = tensor([self.tokenizer.eos_token_id] * self.n_samples, device=model.device).view(self.n_samples, 1)

        for beta in betas:
            responses = batch_map(
                lambda batch: model.sample(query, beta=beta, max_length=self.max_length,
                                           eos_token_id=self.tokenizer.eos_token_id)[0],
                query, bsize=self.batch_size, desc='Generating')

            responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            print(f'\n{beta=}\n' + '\n'.join([f'{text}' for text in responses[:16]]))

            logs.update({f'responses/{beta}': wandb.Table(columns=['response'], data=[[r] for r in responses[:32]])})

        return np.inf, {}
