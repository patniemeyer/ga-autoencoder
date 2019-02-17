import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from itertools import chain
from ga_lib import *
from time import time
import numpy as np
from model import *

def millis(): return time() * 1000

def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# e.g. [128,784]->[128,28,28], [-1,1] to [0,1]
def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # normalize(mean, std): input[channel] = (input[channel] - mean[channel]) / std[channel]
    # [0-1] -> [-1,1]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# todo: reshape the data into regular image batches
def get_data(limit=None):
    batch_size = 128
    traindataset = MNIST('./data', train=True, transform=img_transform, download=True)
    if limit:
        traindataset = list(traindataset)[:limit]  # limit images

    # Subtract mean of training data images
    data = torch.stack([d[0] for d in traindataset])  # one big batch [128, 1, 28, 28])
    mean_img = torch.mean(data, 0, keepdim=True)  # batch average image
    traindataset = [((batch - mean_img), clas) for batch, clas in traindataset]
    return DataLoader(traindataset, batch_size=batch_size, shuffle=False)
    # return torch.stack([(batch - mean_img) for batch, clas in traindataset])







