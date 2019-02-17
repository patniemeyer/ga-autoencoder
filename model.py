import os
import torch
from torch import nn
from itertools import chain
import numpy as np

class autoencoder(nn.Module):
    def __init__(self, filename='./ae.pth'):
        super(autoencoder, self).__init__()
        self.filename = filename
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 12), nn.ReLU(True),
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), nn.ReLU(True),
            nn.Linear(12, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # Return the weights and biases
    def getWeights(self) -> [torch.Tensor]:
        for param in self.parameters():
            yield param.data

    def getWeightsFlat(self) -> torch.Tensor:
        return torch.cat([w.flatten() for w in self.getWeights()])

    # Set weights and biases from the list
    def setWeights(self, weights: [torch.Tensor]):
        weight_data = iter(weights)
        for param in self.parameters():
            param.data = next(weight_data)

    def setWeightsFlat(self, weights_flat: torch.Tensor):
        weights = []
        offset = 0
        for param in self.getWeights():
            param_size = np.prod(param.shape)
            data = weights_flat[offset: offset + param_size]
            if len(param.shape) > 1:
                data = data.reshape(param.shape)
            weights.append(torch.Tensor(data))
            offset += param_size
        self.setWeights(weights)

    def save(self):
        if self.filename:
            torch.save(self.state_dict(), self.filename)
            print("model saved: ", self.filename)

    def load(self):
        if self.filename and os.path.isfile(self.filename):
            self.load_state_dict(torch.load(self.filename))
            print("loaded model: ", self.filename)
            return True
        else:
            return False


