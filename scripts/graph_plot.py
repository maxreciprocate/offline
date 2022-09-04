import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ilql import main
from torch import tensor
import networkx as nx
import numpy as np

optimal_lengths = []
sampled_lengths = []
iql_lengths = []

for seed in range(10):
    model, data = main(seed=seed)
    model.eval()

    g = nx.from_numpy_array(data.adj)

    # optimal
    for start in set(range(data.n_nodes)) - {data.goal}:
        shortest_path = nx.shortest_path(g, start, data.goal)[:data.walk_size]
        if shortest_path[-1] != data.goal:
            optimal_lengths.append(data.walk_size)
        else:
            optimal_lengths.append(len(shortest_path)-1)

    # ilql
    starts = th.arange(1, data.n_nodes).unsqueeze(1).to(model.device)
    paths, _ = model.sample(starts, max_length=data.walk_size, logit_mask=tensor(~data.adj), beta=10) # argmax
    for path in paths:
        length = data.walk_size
        for ind, node in enumerate(path):
            if node == data.goal:
                length = ind
                break

        iql_lengths.append(length)

    # all samples
    for path in data.tensors[0]:
        length = data.walk_size
        for ind, node in enumerate(path):
            if node == data.goal:
                length = ind
                break

        sampled_lengths.append(length)

# ■ ~
from matplotlib import pyplot
import matplotlib

fontcolor = '#444'
matplotlib.rcParams['text.color'] = fontcolor
matplotlib.rcParams['axes.labelcolor'] = fontcolor
matplotlib.rcParams['xtick.color'] = fontcolor
matplotlib.rcParams['ytick.color'] = fontcolor

matplotlib.rcParams["font.family"] = "Futura"
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20
matplotlib.rcParams["figure.titlesize"] = 12
matplotlib.rcParams["figure.figsize"] = 15, 8

matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.dpi'] = 70

ax = pyplot.gca()
ax.set_facecolor('#fff')
ax.grid(color='lightgray', alpha=0.4, axis='y')
ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)

optimal_hist = np.histogram(optimal_lengths, bins=np.arange(1, data.walk_size+2), density=True)[0]
sampled_hist = np.histogram(sampled_lengths, bins=np.arange(1, data.walk_size+2), density=True)[0]
iql_hist = np.histogram(iql_lengths, bins=np.arange(1, data.walk_size+2), density=True)[0]

barsize = 0.36
iql_color = '#99a3fd'
opt_color = '#f2ad48'
random_color='lightgray'

pyplot.bar(np.arange(1, data.walk_size+1)-barsize/1.5, optimal_hist, width=barsize, label='shortest path', color=opt_color, zorder=2)
pyplot.bar(np.arange(1, data.walk_size+1), iql_hist, width=barsize, label='ILQL', color=iql_color, zorder=3)
pyplot.bar(np.arange(1, data.walk_size+1)+barsize/1.5, sampled_hist, width=barsize, label='random walk', color=random_color, zorder=1)

pyplot.legend(fontsize=16)
pyplot.xticks(np.arange(1, data.walk_size+1), list(np.arange(1, data.walk_size)) + ['∞'])

pyplot.xlabel('# of steps to goal', fontsize=22)
pyplot.ylabel('proportion of paths', fontsize=22)

pyplot.savefig('graph_plot.svg')
