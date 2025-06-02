import os
import json
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchvision import utils, transforms


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons

seed = 3                            # pick any integer you like
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

## parameters to change
grid_size = 100
grid_range = [-2., 3.]
# grid_range = [-20., 21.]
contour_cmap = 'RdBu_r'
inclass_color = 'green'
outclass_color = 'orange'
cmap_level = 11

## sample data part
data_all, label_all = make_moons(n_samples=500, noise=0.1)
df = pd.DataFrame(dict(x=data_all[:,0], y=data_all[:,1], label=label_all))

## train network
dim_data = 2
dim_net = 240
network = nn.Sequential(*[nn.Linear(dim_data, dim_net),
                         nn.ReLU(),
                         nn.Linear(dim_net, dim_net),
                         nn.ReLU(),
                         nn.Linear(dim_net, 2)])
                        #  nn.Softmax(dim=1)]

optim = optim.SGD(network.parameters(), lr=0.1)
crit = nn.CrossEntropyLoss()
data_all = torch.tensor(data_all).type(torch.float32)
label_all = torch.tensor(label_all).type(torch.LongTensor)

total_iters = 5000
crit.cuda()
network.cuda()
network.train()
for i in range(total_iters):
    out = network(data_all.cuda())
    loss = crit(out, label_all.cuda())

    optim.zero_grad()
    loss.backward()
    optim.step()

    _, _ = torch.max(out, dim=1)



## plot contour using grids
x_list = np.linspace(grid_range[0], grid_range[1], grid_size)
y_list = np.linspace(grid_range[0], grid_range[1], grid_size)
grid_x, grid_y = np.meshgrid(x_list, y_list)
grid_inputs = np.concatenate((grid_x.reshape(-1,1), grid_y.reshape(-1,1)), axis=1)
grid_inputs = torch.tensor(grid_inputs).type(torch.float32)

network.cpu()
network.eval()


feature_extractor = nn.Sequential(*list(network.children())[:-1])

# Evaluation with KNN using cosine distance
network.eval()
with torch.no_grad():
    # Extract features from training data and grid points
    train_features = feature_extractor(data_all)
    grid_features = feature_extractor(grid_inputs)
    
    # Normalize features for cosine similarity calculation
    train_features_norm = F.normalize(train_features, p=2, dim=1)
    grid_features_norm = F.normalize(grid_features, p=2, dim=1)
    
    # Compute cosine similarity matrix
    # (Cosine similarity = dot product of normalized vectors)
    similarity = torch.mm(grid_features_norm, train_features_norm.t())

    # For each grid point, find the top-k similarity scores and indices
    k = 5
    sims, indices = torch.topk(similarity, k=k, dim=1)

    # Get the k-th similarity score for each grid point
    sims = sims[:, :k-1].mean(dim=1)  # k-1 because of 
    # sims = sims[:, k-1]
    logits = network(grid_inputs)
    probs = F.softmax(logits)
    msps, _ = torch.max(probs, dim=1)

    scores = msps * sims


# scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)
scores = torch.clamp((scores - torch.quantile(scores, 0.05)) /
                     (torch.quantile(scores, 0.95) - torch.quantile(scores, 0.05) + 1e-5),
                     0.0, 1.0)

scores = torch.clip(scores, 0, 1)
scores = scores.reshape(grid_size, grid_size).numpy()

fig, ax = plt.subplots()
levels = np.linspace(scores.min(), 1.0, cmap_level)
contour = ax.contourf(grid_x, grid_y, scores, levels=levels, cmap="RdBu_r", vmax=1, vmin=0.0)
fig.colorbar(contour, ax=[ax])

for data, label in zip(data_all, label_all):
    if label == 0:
        ax.scatter(data[0], data[1], c=inclass_color)
    elif label == 1:
        ax.scatter(data[0], data[1], c=outclass_color)

ax.set_title('Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig('./nnguide.jpg')
plt.close()

