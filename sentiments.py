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

from utils import batch_map, tohuman, load_tensors

def get_reward(sentiment_pipe, texts):
    sentiments = batch_map(lambda batch: sentiment_pipe(batch), texts, bsize=1024, desc='Sentiments')
    return tensor([-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in sentiments])

class Sentiments:
    def __init__(self, tokenizer: AutoTokenizer, max_length=50, n_samples=64, needs_reward_model=False):
        self.max_length = max_length
        self.tokenizer = tokenizer

        if needs_reward_model:
            self.sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=th.device(0))
        else:
            self.sentiment_pipe = None

        texts = load_dataset('imdb', split='train+test')['text']
        tensors = load_tensors(
            'sentiments',
            texts=texts,
            reward_model=partial(get_reward, self.sentiment_pipe),
            tokenizer=self.tokenizer,
            max_length=max_length,
            use_cache=True
        )

        query = tensor([self.tokenizer.bos_token_id] * n_samples).view(n_samples, 1)
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

if __name__ == '__main__':
    import sys
    from rich import print
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    tokenizer.pad_token = tokenizer.eos_token
    ds = Sentiments(tokenizer, needs_reward_model=True).dataset
    print(f'{next(iter(ds))=}')
