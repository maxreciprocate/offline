import numpy as np
import torch as th
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial, reduce

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import wandb

tokenizer = AutoTokenizer.from_pretrained('gpt2')

def tokenize(maxlen, sample):
    input = tokenizer.encode(tokenizer.bos_token + sample['review'])[:maxlen]
    sample['input'] = input
    sample['query'] = tokenizer.decode(input)
    sample['attention'] = [1] * maxlen
    sample['reward']= [0] * (maxlen - 1)
    sample['reward'][-1] = sample['sentiment'] * 2 - 1
    return sample

def filter_outliers(x):
    return len(x['review']) > 200

def process_data(dataset, maxlen):
    dataset = dataset.rename_columns({'text': 'review', 'label': 'sentiment'})
    dataset = dataset.filter(filter_outliers, batched=False)
    dataset = dataset.map(partial(tokenize, maxlen))
    return dataset

def topk_mask(xs, k):
    mintop = th.topk(xs, k)[0][:, -1].unsqueeze(-1)
    return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)

def sizesplit(size: int, xs):
    for ind in range(len(xs) // size + 1):
        yield xs[ind*size:min(len(xs), (ind+1)*size)]

def flatten(xs):
    return list(reduce(lambda acc, x: acc + x, xs, []))

@th.inference_mode()
def sample(model, query=None, nsamples=128, beta=1, maxlen=32):
    if query is None:
        query = tensor([tokenizer.bos_token_id] * nsamples, device=model.device).view(nsamples, 1)

    for _ in range(maxlen):
        logits, qs, _, vs = model(input_ids=query)
        qs = qs[:, -1, :]
        vs = vs[:, -1, :]
        logits = logits[:, -1, :]

        adv = qs - vs
        pi = F.log_softmax(logits, -1)
        modpi = topk_mask(pi + beta * adv, 10)
        ps = F.softmax(modpi, -1)

        tokens = th.multinomial(ps, 1)
        query = th.hstack((query, tokens))

    return query

class Sentiments(Dataset):
    def __init__(self, maxlen=16):
        pipe_device = 0 if th.cuda.is_available() else -1
        self.sentiment_pipe = pipeline('sentiment-analysis', 'lvwerra/distilbert-imdb', device=pipe_device)
        self.maxlen = maxlen

        ds_train = load_dataset('imdb', split='train+test')
        ds_train = process_data(ds_train, maxlen=maxlen)

        self.texts = tensor(ds_train['input'])

        sentiments = flatten([self.sentiment_pipe(batch) for batch in sizesplit(1024, ds_train['query'])])
        self.sentiments = tensor([-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in sentiments])
        self.rewards = th.zeros(len(ds_train), maxlen-1)
        self.rewards[:, -1] = self.sentiments

        self.attention_masks = tensor(ds_train['attention'])

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, ind):
        return self.texts[ind], self.attention_masks[ind], self.rewards[ind]

    def eval(self, logs, tbar, model, beta=1, nsamples=128):
        responses = sample(model, beta=beta, maxlen=self.maxlen, nsamples=nsamples)
        sentences = [tokenizer.decode(response[1:]) for response in responses]

        sentiments = [1-s['score'] if s['label'] == 'NEGATIVE' else s['score'] for s in self.sentiment_pipe(sentences)]
        sentiment = np.mean(sentiments)

        rows = list(zip(sentences, sentiments))

        print(f'\n{beta=} {sentiment=:.2f}\n' + '\n'.join([f'[{sent:.2f}] {text}' for text, sent in rows[:8]]))

        logs[f'sentiment/{beta}'] = sentiment
        logs.update({f'responses/{beta}': wandb.Table(columns=['response', 'sentiment'], rows=rows[:32])})
        tbar.set_postfix({'sentiment': f'{sentiment:.2f}'})
