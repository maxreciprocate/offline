import numpy as np
import torch as th
from torch import tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import wandb

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 10,
}
gpt2_tokenizer = AutoTokenizer.from_pretrained('lvwerra/gpt2-imdb')

def tokenize(sample, maxlen=16):
    input = gpt2_tokenizer.encode(gpt2_tokenizer.bos_token + sample['review'])[:maxlen]
    sample['input'] = input
    sample['query'] = gpt2_tokenizer.decode(input)
    sample['attention'] = [1] * maxlen
    sample['reward']= [0] * (maxlen - 1)
    sample['reward'][-1] = sample['sentiment'] * 2 - 1
    return sample

def process_data(dataset):
    dataset = dataset.rename_columns({'text': 'review', 'label': 'sentiment'})
    dataset = dataset.filter(lambda x: len(x["review"]) > 200, batched=False)
    dataset = dataset.map(tokenize, batched=False)
    return dataset

def topk_mask(xs, k):
    mintop = th.topk(xs, k)[0][-1]
    return th.where(xs < mintop, -np.inf * th.ones_like(xs, dtype=xs.dtype), xs)

@th.no_grad()
def sample(pi_beta, qv_model, beta, maxlen):
    input = tensor([gpt2_tokenizer.bos_token_id], device=pi_beta.device)

    for _ in range(maxlen):
        logits = pi_beta(input_ids=input)[-1]

        if qv_model:
            qs, vs = qv_model(input_ids=input)
            mod = beta * (qs[-1] - vs[-1])
            pi = F.log_softmax(logits, dim=-1)
            modpi = pi + mod
            modpi = topk_mask(modpi, 10)
            ps = F.softmax(modpi, -1)
        else:
            logits = topk_mask(logits, 10)
            ps = F.softmax(logits, -1)

        token = th.multinomial(ps, 1)
        input = th.hstack((input, token))

    return gpt2_tokenizer.decode(input[1:])

class Sentiments(Dataset):
    def __init__(self):
        pipe_device = 0 if th.cuda.is_available() else -1
        self.sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=pipe_device)

        ds_train = load_dataset('imdb')['train']
        ds_train = process_data(ds_train)
        self.texts = tensor(ds_train['input'])
        self.rewards = tensor(ds_train['reward'], dtype=th.float32)
        self.attention_masks = tensor(ds_train['attention'])

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, ind):
        return self.texts[ind], self.attention_masks[ind], self.rewards[ind]

    def eval(self, logs, pi_beta, qv_model=None, beta=1, nsamples=256):
        reward = 0

        gentexts = []
        for _ in range(nsamples):
            text = sample(pi_beta, qv_model, beta=beta, maxlen=16)
            s = self.sentiment_pipe(text)[0]
            r = -s['score'] if s['label'] == 'NEGATIVE' else s['score']

            gentexts.append([text, r])
            reward += r / nsamples

        logs['mean_reward'] = reward
        logs.update({'texts': wandb.Table(columns=['gentext', 'reward'], rows=gentexts)})
        return reward
