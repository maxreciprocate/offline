import os
import numpy as np
import torch as th
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial, reduce
from transformers import AutoTokenizer
from datasets import load_dataset
from util.carp_util import load_carp, scorer
import wandb

tokenizer = AutoTokenizer.from_pretrained('gpt2')

carp = load_carp(
    model_type='coop',
    config_path='ControlledCarp/magiCARP/configs/coop/alignment_coop.yml',
    ckpt_path='New_Alignment_CoOp_Carp_L/'
).to('cuda')
carp.eval()

def clean_text(text):
    return '. '.join(map(
        lambda x: x.strip(),
        text.replace(' . ', '. ').replace(' , ', ', ').replace(" '", "'").replace(" n't", "n't").split('. ')
    ))

def sizesplit(size: int, xs):
    for ind in range(len(xs) // size + int((len(xs) % size) > 0)):
        yield xs[ind*size:min(len(xs), (ind+1)*size)]

def topk_mask(xs, k):
    mintop = th.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)

def tokenize(max_length, diff_reward, offset_reward, review, sample):
    text = clean_text(sample['text'])
    tokens = tokenizer.encode(text, return_tensors='pt')[:, :max_length-1]
    tokens = F.pad(tokens, (0, max_length-tokens.shape[1]-1), value=tokenizer.eos_token_id)

    if diff_reward:
        substrings = []
        newtext = ""
        for token in tokens[0]:
            newtext += tokenizer.decode(token)
            substrings.append(newtext)

        rewards = carp_score(substrings, review).cpu()
        rewards = th.hstack((tensor([offset_reward]), rewards)).diff()
        rewards = th.where(tokens[0] == tokenizer.eos_token_id, 0, rewards)

    else:
        r = carp_score(text, review).item()
        rewards = th.empty(max_length-1)
        rewards.fill_(r)
        rewards[tokens[0] == tokenizer.eos_token_id] = 0

    attn = [1] * max_length
    attn[-1] = 0
    sample['text'] = text
    sample['tokens'] = th.hstack((tensor([[tokenizer.eos_token_id]]), tokens))
    sample['attention'] = attn
    sample['rewards'] = rewards
    return sample

@th.inference_mode()
def carp_score(texts, review):
    return scorer(texts, [review], carp, mode='coop').view(-1)

@th.inference_mode()
def sample(model, query=None, n_samples=128, beta=1, max_length=32, temperature=0.8, top_k=20):
    if query is None:
        query = tensor([tokenizer.bos_token_id] * n_samples, device=model.device).view(n_samples, 1)

    for _ in range(max_length):
        logits, qs, _, vs = model(input_ids=query)
        logits = logits[:, -1, :]
        qs = qs[:, -1, :]
        vs = vs[:, -1, :]

        adv = qs - vs
        pi = F.log_softmax(logits, -1)
        modpi = topk_mask(pi + beta * adv, top_k)
        ps = F.softmax(modpi / temperature, -1)

        tokens = th.multinomial(ps, 1)
        query = th.hstack((query, tokens))

    return query

class Carps(Dataset):
    def __init__(self, review='good', max_length=48, diff_reward=True, n_samples=64):
        self.review = review
        self.max_length = max_length
        self.n_samples = n_samples

        cache_path = f'stash/carps-{max_length}l-{diff_reward}d.pt'

        if os.path.exists(cache_path):
            cache = th.load(cache_path)
            self.tokens = cache['tokens']
            self.rewards = cache['rewards']
            self.attention_masks = cache['attention_masks']
            self.validation_queries = cache['validation_queries']
        else:
            ds, valid = load_dataset(
                'text',
                data_files={'train': 'roc_train_all.txt', 'valid': 'roc_valid.txt'},
                split=['train', f'valid[:{n_samples}]'])

            if diff_reward:
                vocab = list(tokenizer.get_vocab().keys())
                offset = th.hstack([carp_score(words, review) for words in sizesplit(32, vocab)]).mean()
            else:
                offset = 0

            ds = ds.map(partial(tokenize, max_length, diff_reward, offset, review))
            valid = valid.map(partial(tokenize, max_length, diff_reward, offset, review))

            self.tokens = th.tensor(ds['tokens']).squeeze(1)
            self.rewards = tensor(ds['rewards'])
            self.attention_masks = tensor(ds['attention'])
            self.validation_queries = tensor(valid['tokens']).squeeze(1)[:n_samples, :6]

            th.save({ 'tokens': self.tokens,
                      'rewards': self.rewards,
                      'attention_masks': self.attention_masks,
                      'validation_queries': self.validation_queries }, cache_path)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, ind):
        return self.tokens[ind], self.attention_masks[ind], self.rewards[ind]

    def eval(self, logs, model, betas=[1]):
        model.eval()
        queries = self.validation_queries.to(model.device)

        for beta in betas:
            responses = sample(model, query=queries, beta=beta, max_length=self.max_length, n_samples=self.n_samples)
            texts = [tokenizer.decode(response[1:]) for response in responses]

            rewards = th.hstack([carp_score(ts, self.review) for ts in sizesplit(8, texts)])
            reward = rewards.mean().item()
            rows = list(zip(texts, rewards.tolist()))

            print(f'\n{beta=} {reward=:.2f}\n' + '\n'.join([f'[{r:.2f}] {text}' for text, r in rows[:8]]))

            logs[f'reward/beta{beta}'] = reward
            logs.update({f'responses/beta{beta}': wandb.Table(columns=['response', 'reward'], rows=rows[:32])})

        stats = {'reward': f'{reward:.2f}'}
        model.train()
        return reward, stats
