import numpy as np
import torch as th
from torch import tensor
import networkx as nx
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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

# Toy dataset from Decision Transformer (Chen et. al 2021)
class RandomWalks(Dataset):
    def __init__(self, nnodes=20, walksize=10, nwalks=1000, pedge=0.1, seed=1002):
        self.nnodes = nnodes
        self.nwalks = nwalks
        self.walksize = walksize
        rng = np.random.RandomState(seed)

        walks, rewards = [], []
        while True:
            self.adj = rng.rand(nnodes, nnodes) > (1 - pedge)
            np.fill_diagonal(self.adj, 0)
            if np.all(self.adj.sum(1)): break

        # terminal state
        self.adj[0, :] = 0
        self.adj[0, 0] = 1

        self.goal = 0
        for _ in range(nwalks):
            node = randexclude(rng, nnodes, self.goal)
            walk = [node]

            for istep in range(walksize-1):
                node = rng.choice(np.nonzero(self.adj[node])[0])
                walk.append(node)
                if node == self.goal:
                    break

            r = th.zeros(walksize-1)
            r[:len(walk)-1] = -1 if walk[-1] == self.goal else -100

            rewards.append(r)
            walks.append(walk)

        states = []
        attention_masks = []

        for r, walk in zip(rewards, map(th.tensor, walks)):
            attention_mask = th.zeros(walksize, dtype=int)
            attention_mask[:len(walk)] = 1

            attention_masks.append(attention_mask)
            states.append(F.pad(walk, (0, walksize-len(walk))))

        self.rewards = th.stack(rewards)
        self.attention_masks = th.stack(attention_masks)
        self.states = th.stack(states)

        self.worstlen = self.walksize
        self.avglen = sum(map(len, walks)) / self.nwalks
        self.bestlen = 0
        g = nx.from_numpy_array(self.adj)
        for start in set(range(self.nnodes)) - {self.goal}:
            shortest_path = nx.shortest_path(g, start, self.goal)[:self.walksize]
            self.bestlen += len(shortest_path)
        self.bestlen /= self.nnodes - 1

        print(f'{self.nwalks} walks of which {(np.array([r[0] for r in self.rewards])==-1).mean()*100:.0f}% arrived at destination')

    def __len__(self):
        return self.nwalks

    def __getitem__(self, ind):
        return self.states[ind], self.attention_masks[ind], self.rewards[ind]

    def render(self):
        from matplotlib import pyplot

        g = nx.from_numpy_array(self.adj)
        pos = nx.spring_layout(g, seed=7357)

        pyplot.figure(figsize=(10, 8))
        nx.draw_networkx_edges(g, pos=pos, alpha=0.5, width=1, edge_color='#d3d3d3')
        nx.draw_networkx_nodes(g, nodelist=set(range(len(self.adj))) - {self.goal}, pos=pos, node_size=300, node_color='orange')
        nx.draw_networkx_nodes(g, nodelist=[self.goal], pos=pos, node_size=300, node_color='darkblue')
        pyplot.show()

    @th.inference_mode()
    def eval(self, logs, tbar, model, beta=1):
        paths = th.arange(1, self.nnodes).view(self.nnodes - 1, -1).to(model.device)

        store_qs = []
        store_vs = []
        store_adv = []

        for _ in range(self.walksize):
            logits, qs, _, vs = model(input_ids=paths)

            qs = qs[:, -1, :]
            vs = vs[:, -1, :]

            unreachable = np.where(~self.adj[paths[:, -1].cpu().numpy()])

            logits = logits[:, -1, :]
            logits[unreachable] = -np.inf
            pi = F.log_softmax(logits, -1)

            advantage = qs - vs
            pi = F.log_softmax(pi + beta * advantage, -1)

            pi = th.exp(pi)
            pi /= pi.sum()

            steps = th.multinomial(pi, 1)
            # steps = th.argmax(pi, -1, keepdim=True)
            paths = th.hstack((paths, steps))

            store_qs.append(qs)
            store_vs.append(vs)
            store_adv.append(advantage)

        logvars('qs', logs, store_qs)
        logvars('vs', logs, store_vs)
        logvars('adv', logs, store_adv)

        nodes = th.where(paths == 0)[0]
        narrived = len(set(nodes.tolist()))

        actlen = 0
        for node in range(self.nnodes-1):
            for istep in range(self.walksize):
                if paths[node, istep] == self.goal:
                    break

            actlen += (istep + 1) / (self.nnodes - 1)

        current = (self.worstlen - actlen)/(self.worstlen - self.bestlen)
        average = (self.worstlen - self.avglen)/(self.worstlen - self.bestlen)

        logs.update({ 'actlen': actlen,
                      'avglen': self.avglen,
                      'bestlen': self.bestlen,
                      'worstlen': self.worstlen })

        tbar.set_postfix({ 'arrived': f'{narrived / (self.nnodes-1) * 100:.0f}%',
                           'optimal': f'{current*100:.0f}% > {average*100:.0f}%' })

        return actlen
