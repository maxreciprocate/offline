import numpy as np
import torch as th
from torch import tensor
import networkx as nx
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import logvars, randexclude, flatten
import wandb

# Toy dataset from Decision Transformer (Chen et. al 2021)
class RandomWalks(Dataset):
    def __init__(self, n_nodes=20, walk_size=10, n_walks=1000, p_edge=0.1, seed=1002):
        self.n_nodes = n_nodes
        self.n_walks = n_walks
        self.walk_size = walk_size
        rng = np.random.RandomState(seed)

        walks, rewards = [], []
        while True:
            self.adj = rng.rand(n_nodes, n_nodes) > (1 - p_edge)
            np.fill_diagonal(self.adj, 0)
            if np.all(self.adj.sum(1)): break

        # terminal state
        self.adj[0, :] = 0
        self.adj[0, 0] = 1

        self.goal = 0
        for _ in range(n_walks):
            node = randexclude(rng, n_nodes, self.goal)
            walk = [node]

            for istep in range(walk_size-1):
                node = rng.choice(np.nonzero(self.adj[node])[0])
                walk.append(node)
                if node == self.goal:
                    break

            r = th.zeros(walk_size-1)
            r[:len(walk)-1] = -1 if walk[-1] == self.goal else -100

            rewards.append(r)
            walks.append(walk)

        states = []
        attention_masks = []

        for r, walk in zip(rewards, map(th.tensor, walks)):
            attention_mask = th.zeros(walk_size, dtype=int)
            attention_mask[:len(walk)] = 1

            attention_masks.append(attention_mask)
            states.append(F.pad(walk, (0, walk_size-len(walk))))

        self.rewards = th.stack(rewards)
        self.attention_masks = th.stack(attention_masks)
        self.states = th.stack(states)

        self.worstlen = self.walk_size
        self.avglen = sum(map(len, walks)) / self.n_walks
        self.bestlen = 0
        g = nx.from_numpy_array(self.adj)
        for start in set(range(self.n_nodes)) - {self.goal}:
            shortest_path = nx.shortest_path(g, start, self.goal)[:self.walk_size]
            self.bestlen += len(shortest_path)
        self.bestlen /= self.n_nodes - 1

        print(f'{self.n_walks} walks of which {(np.array([r[0] for r in self.rewards])==-1).mean()*100:.0f}% arrived at destination')

    def __len__(self):
        return self.n_walks

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
    def eval(self, logs, model, two_qs=True, betas=[1]):
        model.eval()
        paths = th.arange(1, self.n_nodes).view(self.n_nodes - 1, -1).to(model.device)
        beta = betas[-1]

        store_qs = []
        store_vs = []
        store_adv = []
        store_done = [th.ones_like(paths)]

        for _ in range(self.walk_size):
            logits, _, target_qs, vs = model(input_ids=paths)
            if two_qs:
                qs = th.minimum(target_qs[0][:, -1, :], target_qs[1][:, -1, :])
            else:
                qs = target_qs[:, -1, :]

            vs = vs[:, -1, :]

            unreachable = np.where(~self.adj[paths[:, -1].cpu().numpy()])

            logits = logits[:, -1, :]
            logits[unreachable] = -np.inf
            pi = F.log_softmax(logits, -1)

            advs = qs - vs
            pi = F.softmax(pi + beta * advs, -1)

            steps = th.argmax(pi, 1, keepdims=True)
            paths = th.hstack((paths, steps))

            store_done.append((steps != self.goal).int())
            store_qs.append(qs)
            store_vs.append(vs)
            store_adv.append(advs)

        dones = th.hstack(store_done)
        qs = th.stack(store_qs, dim=1)
        vs = th.stack(store_vs, dim=1)

        logvars('qs', logs, store_qs)
        logvars('vs', logs, store_vs)
        logvars('adv', logs, store_adv)

        narrived = 0
        actlen = 0
        for node in range(self.n_nodes-1):
            for istep in range(self.walk_size):
                if paths[node, istep] == self.goal:
                    narrived += 1
                    break

            actlen += (istep + 1) / (self.n_nodes - 1)

        current = (self.worstlen - actlen)/(self.worstlen - self.bestlen)
        average = (self.worstlen - self.avglen)/(self.worstlen - self.bestlen)

        logs.update({ 'actlen': actlen,
                      'avglen': self.avglen,
                      'bestlen': self.bestlen,
                      'worstlen': self.worstlen })

        stats = { 'arrived': f'{narrived / (self.n_nodes-1) * 100:.0f}%',
                  'optimal': f'{current*100:.0f}% > {average*100:.0f}%' }

        model.train()
        return -actlen, stats
