import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from itertools import chain
import numpy as np

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not os.path.exists('./out/'): os.mkdir('./out/')
f = open('loss.txt', 'w'); f.truncate(); f.close()

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

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
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
        # if epoch % 100 == 0: print(epoch, x.numpy())
        x = self.decoder(x)
        return x

    # Return the weights and biases
    def getWeights(self) -> [torch.Tensor]:
        for module in chain(self.encoder.modules(), self.decoder.modules()):
            if hasattr(module, 'weight'):
                yield module.weight.data
                yield module.bias.data

    # Set weights and biases from the list
    def setWeights(self, weights: [torch.Tensor]):
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))

        weight_bias_pairs = chunker(weights, 2)
        for module in chain(self.encoder.modules(), self.decoder.modules()):
            if hasattr(module, 'weight'):
                module.weight.data, module.bias.data = next(weight_bias_pairs)


def fitness(weights, save=False):
    model.setWeights(weights)
    epochloss = 0
    for batch in traindataloader:
        img, _ = batch  # [128, 1, 28, 28]
        img = img.view(img.size(0), -1)  # flat [128, 784]
        if torch.cuda.is_available(): img = img.cuda()
        output = model(img)  # [128, 784]

        batch_loss = criterion(output, img).item()  # scalar
        epochloss += batch_loss

    if save:
        images = [
            img[0].view(1, 28, 28), to_img(output[0:1]).squeeze().unsqueeze(0),
            img[1].view(1, 28, 28), to_img(output[1:2]).squeeze().unsqueeze(0)
        ]
        save_image(images, './out/image_{}.png'.format(epoch))

    return epochloss


batch_size = 128
traindataset = MNIST('./data', train=True, transform=img_transform, download=True)
#traindataset = list(traindataset)[:128]  # limit images
traindataset = list(traindataset)[:2]  # limit images

# Subtract mean of training data images
data = torch.stack([d[0] for d in traindataset])  # one big batch [128, 1, 28, 28])
mean_img = torch.mean(data, 0, keepdim=True)  # batch average image
traindataset = [(batch - mean_img, clas) for batch, clas in traindataset]

traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)
model = autoencoder()
if torch.cuda.is_available(): model.cuda()
criterion = nn.MSELoss(reduction='sum')
weight_start = list(model.getWeights())

npop = 25  # population size [25]
sigma = 1e-3  # noise standard deviation [1e-3]
alpha = 0.01 # learning rate [0.01]

epoch=0
def train(num_epochs = 10000, target_fitness=1.0):
    global epoch
    # initial weight guess
    # weight_current = [torch.randn_like(weight) * sigma for weight in weight_start]
    weight_current = [torch.zeros_like(weight) for weight in weight_start]

    for epoch in range(num_epochs):
        current_fitness = fitness(weight_current)
        if current_fitness <= target_fitness: break
        f=open('loss.txt', 'a'); print(current_fitness, file=f); f.close()
        if epoch % 1 == 0:
            print('epoch %d. fitness: %f' % (epoch, current_fitness))
        if epoch % 100 == 0:
            fitness(weight_current, save=True)

        R = np.zeros(npop)  # population fitness
        N = [[]] * npop # population noise
        for j in range(npop):
            N[j] = [torch.randn_like(weight) for weight in weight_current]
            wtry = [a + b * sigma for a, b in zip(weight_current, N[j])]
            R[j] = fitness(wtry)  # evaluate jittered version

        # standardize the rewards to have a gaussian distribution
        F = (R - np.mean(R)) / np.std(R)

        # w = w + alpha / (npop * sigma) * np.dot(N.T, A)
        for i in range(npop):
            for j in range(len(weight_current)):
                weight_current[j] = weight_current[j] - alpha / npop * F[i] * N[i][j]


train()

