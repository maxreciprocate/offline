import os
import numpy as np
import torch as th
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from functools import partial, reduce

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import wandb
from tqdm import tqdm
import math

from utils import batch_map, tohuman

def load_tensors(name, get_text, get_reward, tokenizer, max_length=64, use_cache=True):
    cache_path = f'cache/{name}_{max_length=}_tokenizer={tokenizer.name_or_path.split("/")[-1]}.pt'
    if use_cache and os.path.exists(cache_path):
        tensors = th.load(cache_path)
    else:
        tensors = tokenizer(
            get_text(),
            max_length=max_length,
            padding=True,
            truncation='longest_first',
            return_overflowing_tokens=True,
            return_tensors='pt'
        )

        reward = get_reward(tensors['input_ids'])
        reward = (reward - reward.mean()) / (reward.std() + 1e-30)
        rewards = reward.view(-1, 1).repeat(1, max_length)

        rewards[tensors['attention_mask'].eq(tokenizer.pad_token_id)] = 0
        tensors['rewards'] = rewards[:, :-1].contiguous()

        if not os.path.exists(os.path.dirname(cache_path)):
            os.mkdir(os.path.dirname(cache_path))

        th.save(tensors, cache_path)

    print(f"{tohuman(np.prod(tensors['input_ids'].shape))} tokens")
    return tensors

def get_reward(sentiment_pipe, tokenizer, input_ids):
    sentiments = batch_map(
        lambda batch: sentiment_pipe(tokenizer.batch_decode(batch)),
        input_ids, bsize=1024, desc='Sentiments'
    )

    return tensor([-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in sentiments])

def get_text(tokenizer):
    ds = load_dataset('imdb', split='train+test')

    def wrap_text(x):
        x['text'] = tokenizer.bos_token + x['text'] + tokenizer.eos_token
        return x

    return ds.map(wrap_text)['text']

class Sentiments:
    def __init__(self, tokenizer: AutoTokenizer, max_length=50, n_samples=256, needs_pipe=False):
        self.max_length = max_length
        self.tokenizer = tokenizer

        if needs_pipe:
            self.sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=th.device(0))
        else:
            self.sentiment_pipe = None

        tensors = load_tensors(
            'sentiments',
            get_text=partial(get_text, self.tokenizer),
            get_reward=partial(get_reward, self.sentiment_pipe, self.tokenizer),
            tokenizer=self.tokenizer,
            max_length=max_length,
            use_cache=True
        )

        query = tensor([self.tokenizer.eos_token_id] * n_samples).view(n_samples, 1)
        self.logit_mask = None

        self.dataset = TensorDataset(tensors['input_ids'], tensors['attention_mask'], tensors['rewards'])
        self.eval_dataset = TensorDataset(query)

    def eval(self, samples, beta):
        reviews = self.tokenizer.batch_decode(samples, skip_special_tokens=True)

        rewards = [1-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in self.sentiment_pipe(reviews)]
        reward = np.mean(rewards)

        rows = list(zip(reviews, rewards))
        print(f'\n{beta=} {reward=:.2f}\n' + '\n'.join([f'[{sent:.2f}] {text}' for text, sent in rows[:8]]))

        stats = { f'reward/{beta}': reward,
                  f'responses/{beta}': wandb.Table(columns=['response', 'sentiment'], rows=rows[:32]) }

        return reward, stats
