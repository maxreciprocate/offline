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

def load_tensors(cache_path, tokenizer, sentiment_pipe, max_length=128, use_cache=True):
    if use_cache and os.path.exists(cache_path):
        cache = th.load(cache_path)
    else:
        ds = load_dataset('imdb', split='train+test')

        tokens = []
        for text in tqdm(ds['text']):
            ids = tokenizer(text)['input_ids']
            ids.insert(0, tokenizer.eos_token_id)
            ids.append(tokenizer.eos_token_id)

            for i in range(0, max(1, len(ids)-max_length+1), max_length):
                sids = ids[i:i + max_length]
                tokens.append(sids + [tokenizer.eos_token_id] * (max_length - len(ids)))

        tokens = tensor(tokens)
        attention_masks = tokens.ne(tokenizer.eos_token_id).int()
        attention_masks[:, 0] = 1

        sentiments = batch_map(lambda batch: sentiment_pipe(tokenizer.batch_decode(batch)), tokens, bsize=1024)
        sentiments = tensor([-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in sentiments])
        rewards = sentiments.view(-1, 1).repeat(1, max_length-1)

        cache = {'tokens': tokens, 'attention_masks': attention_masks, 'rewards': rewards}

        if not os.path.exists(os.path.dirname(cache_path)):
            os.mkdir(os.path.dirname(cache_path))

        th.save(cache, cache_path)

    print(f"{tohuman(np.prod(cache['tokens'].shape))} tokens")
    return cache

class Sentiments(TensorDataset):
    def __init__(self, tokenizer: AutoTokenizer, max_length=64, n_samples=32):
        self.max_length = max_length
        self.n_samples = n_samples
        self.tokenizer = tokenizer

        device = 0 if th.cuda.is_available() else -1
        self.sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=device)

        cache_path = f'cache/imdb-sentiments_{max_length=}_tokenizer={tokenizer.name_or_path}.pt'
        tensors = load_tensors(cache_path, self.tokenizer, self.sentiment_pipe, max_length=max_length)

        super().__init__(tensors['tokens'], tensors['attention_masks'], tensors['rewards'])

    def eval(self, logs, model, betas=[1]):
        query = tensor([self.tokenizer.eos_token_id] * self.n_samples, device=model.device).view(self.n_samples, 1)

        for beta in betas:
            responses, stats = model.sample(query, beta=beta, max_length=self.max_length)
            reviews = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            rewards = [1-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in self.sentiment_pipe(reviews)]
            reward = np.mean(rewards)

            rows = list(zip(reviews, rewards))
            print(f'\n{beta=} {reward=:.2f}\n' + '\n'.join([f'[{sent:.2f}] {text}' for text, sent in rows[:8]]))

            logs[f'reward/{beta}'] = reward
            logs.update({f'responses/{beta}': wandb.Table(columns=['response', 'sentiment'], rows=rows[:32])})

        stats = {'reward': f'{reward:.2f}'}
        return reward, stats
