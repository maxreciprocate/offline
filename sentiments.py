import numpy as np
import torch as th
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial, reduce

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import wandb
from utils import topk_mask, sizesplit, flatten

tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize(max_length, sample):
    input = tokenizer.encode(tokenizer.bos_token + sample['review'])[:max_length]
    sample['input'] = F.pad(tensor(input), (0, max_length-len(input)), value=tokenizer.eos_token_id)
    sample['text'] = tokenizer.decode(sample['input'], skip_special_tokens=True)
    attn = th.ones(len(input)-1)
    sample['attention_mask'] = F.pad(attn, (0, max_length-len(attn)))
    return sample

@th.inference_mode()
def sample(model, query=None, n_samples=128, beta=1, max_length=32, temperature=1, top_k=20):
    if query is None:
        query = tensor([tokenizer.bos_token_id] * n_samples, device=model.device).view(n_samples, 1)

    for _ in range(max_length):
        logits, _, target_qs, vs = model(input_ids=query)
        if model.two_qs:
            qs = th.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
        else:
            qs = target_qs[:, -1, :]

        logits = logits[:, -1, :]
        vs = vs[:, -1, :]

        adv = qs - vs
        pi = F.log_softmax(logits, -1)
        modpi = topk_mask(pi + beta * adv, top_k)
        ps = F.softmax(modpi / temperature, -1)

        tokens = th.multinomial(ps, 1)
        query = th.hstack((query, tokens))

    return query

class Sentiments(Dataset):
    def __init__(self, max_length=16, n_samples=4):
        pipe_device = 0 if th.cuda.is_available() else -1
        self.sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=pipe_device)
        self.max_length = max_length

        ds = load_dataset('imdb', split='train+test').shuffle(seed=1000)
        ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
        ds = ds.filter(lambda x: 200 < len(x['review']) < tokenizer.max_len_single_sentence, batched=False)
        ds = ds.map(partial(tokenize, max_length))
        ds_train, ds_valid = ds[:-n_samples], ds[-n_samples:]

        self.tokens = tensor(ds_train['input'])
        self.attention_masks = tensor(ds_train['attention_mask'])

        sentiments = flatten([self.sentiment_pipe(batch) for batch in sizesplit(1024, ds_train['text'])])
        sentiments = tensor([-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in sentiments])
        self.rewards = sentiments.view(-1, 1).repeat(1, max_length-1)

        queries = []
        min_query_len = 4
        max_query_len = max(min_query_len+1, max_length // 2)
        cutoffs = np.random.RandomState(1000).randint(min_query_len, max_query_len, size=n_samples)
        for i, cutoff in zip(range(n_samples), cutoffs):
            query = ds_valid['input'][i][:cutoff]
            query = F.pad(tensor(query), (max_query_len-cutoff, 0), value=tokenizer.eos_token_id)
            queries.append(query)

        self.queries = th.vstack(queries)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, ind):
        return self.tokens[ind], self.attention_masks[ind], self.rewards[ind]

    def eval(self, logs, model, betas=[1]):
        model.eval()

        for beta in betas:
            responses = sample(model, self.queries.to(model.device), beta=beta, max_length=self.max_length)
            reviews = [tokenizer.decode(response, skip_special_tokens=True) for response in responses]

            rewards = [1-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in self.sentiment_pipe(reviews)]
            reward = np.mean(rewards)

            rows = list(zip(reviews, rewards))
            print(f'\n{beta=} {reward=:.2f}\n' + '\n'.join([f'[{sent:.2f}] {text}' for text, sent in rows[:8]]))

            logs[f'reward/{beta}'] = reward
            logs.update({f'responses/{beta}': wandb.Table(columns=['response', 'sentiment'], rows=rows[:32])})

        stats = {'reward': f'{reward:.2f}'}
        model.train()
        return reward, stats
