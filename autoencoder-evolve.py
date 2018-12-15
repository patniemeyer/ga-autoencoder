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

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not os.path.exists('./out/'): os.mkdir('./out/')

# e.g. [128,784]->[128,28,28], [-1,1] to [0,1]
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # normalize(mean, std): input[channel] = (input[channel] - mean[channel]) / std[channel]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # Return the weights and biases
    def getWeights(self)->[torch.Tensor]:
        for module in chain(self.encoder.modules(), self.decoder.modules()):
            if hasattr(module, 'weight'):
                yield module.weight.data
                yield module.bias.data

    # Set weights and biases from the list
    def setWeights(self, weights: [torch.Tensor]):
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        weight_bias_pairs=chunker(weights,2)
        for module in chain(self.encoder.modules(), self.decoder.modules()):
            if hasattr(module, 'weight'):
                module.weight.data, module.bias.data = next(weight_bias_pairs)

def fitness(weights, save=False):
    model.setWeights(weights)
    epochloss=0
    for data in traindataloader:
        img, _ = data  # [128, 1, 28, 28]
        img = img.view(img.size(0), -1)  # flat [128, 784]
        if torch.cuda.is_available(): img = img.cuda()
        output = model(img)  # [128, 784]
        batch_loss = criterion(output, img).item() # scalar
        #print(batch_loss, file=open('loss.txt', 'a'))
        epochloss+=batch_loss

    if save:
        global save_count
        if not 'save_count' in globals(): save_count=0
        showBatch=False
        if showBatch:
            pic = to_img(output.cpu().data)
            save_image(pic, './out/image_{}.png'.format(save_count))
        else:
            pairs=[
                img[0].view(1,28,28), to_img(output[0:1]).squeeze().unsqueeze(0),
                img[1].view(1,28,28), to_img(output[1:2]).squeeze().unsqueeze(0)
            ]
            save_image(pairs, './out/image_{}.png'.format(save_count))
        save_count += 1
        #print("output diff = ", nn.MSELoss()(output[0], output[1]).item())

    return epochloss

def create_individual()->[torch.Tensor]:
    return [(torch.rand_like(weight)-0.5)*0.01 for weight in weight_start]

def mutate_per_weight(individual: [torch.Tensor]):
    for weight in individual:
        mutate_weight(weight)

def mutate_weight(weight: torch.Tensor):
    length = len(weight.view(-1))
    target_gene_mutation_count = random.choice(
        [x for x in [1, 1, 1, 1, 10, 10, 100, 1000, 10000, 100000, 1000000] if x < length])
    per_gene_mutation_rate = target_gene_mutation_count / length

    type = random.choice(['weighted_offset', 'offset'])
    if type == 'weighted_offset':
        if target_gene_mutation_count <= 10:  # a bit faster for small mutations
            for _ in range(target_gene_mutation_count):
                i=random.randint(0,length-1)
                weight.view(-1)[i] += (random.random()-0.5) * weight.view(-1)[i]
        else:
            maskrand = torch.rand_like(weight)
            mask = maskrand.lt(per_gene_mutation_rate).float()
            rand = torch.rand_like(weight)-0.5  # [-0.5,0.5]
            weight += rand * mask * weight

    elif type == 'offset':
        if target_gene_mutation_count <= 10:
            for _ in range(target_gene_mutation_count):
                i=random.randint(0,length-1)
                weight.view(-1)[i] += random.random()-0.5
        else:
            maskrand = torch.rand_like(weight)
            mask = maskrand.lt(per_gene_mutation_rate).float()
            rand = torch.rand_like(weight)-0.5  # [-0.5,0.5]
            weight += rand * mask

def two_point_crossover_per_weight(ind1: [torch.Tensor], ind2: [torch.Tensor]):
    for i in range(min(len(ind1), len(ind2))):
        w1, w2 = ind1[i], ind2[i]
        c1, c2 = ga.two_point_crossover(w1.view(-1), w2.view(-1))
        ind1[i], ind2[i] = c1.view_as(w1), c2.view_as(w2)
    return ind1, ind2

start_time = time()
def report(ga: GeneticAlgorithm, i: int):
    global start_time
    end_time = time()
    print(i, ga.best_individual().fitness, ga.worst_individual().fitness, end_time - start_time);
    start_time = time()
    if i % 10 == 0:
        fitness(ga.best_individual().genes, save=True)

batch_size = 128
traindataset = MNIST('./data', train=True, transform=img_transform, download=True)
traindataset = list(traindataset)[:2]  # limit two two images
traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)
model = autoencoder()
if torch.cuda.is_available(): model.cuda()
criterion = nn.MSELoss()
weight_start = list(model.getWeights())

ga = GeneticAlgorithm(
    population_size=100,
    crossover_probability=0.7,
    mutation_probability=0.3,
    elitism=True,
    maximise_fitness=False)

ga.create_individual = create_individual
ga.selection_function = ga.tournament_selection
ga.crossover_function = two_point_crossover_per_weight
ga.mutate_function = mutate_per_weight
ga.fitness_function = fitness
ga.iteration_function = report

ga.run(10000)

"""
# alternate objectives
traindataset = MNIST('./data', train=True, transform=img_transform, download=True)
traindataset1 = list(traindataset)[0:1]
traindataloader1 = DataLoader(traindataset1, batch_size=batch_size, shuffle=False)
traindataset = MNIST('./data', train=True, transform=img_transform, download=True)
traindataset2 = list(traindataset)[1:2]
traindataloader2 = DataLoader(traindataset2, batch_size=batch_size, shuffle=False)
for i in range(10000):
    traindataloader = traindataloader1
    ga.run(100)
    traindataloader = traindataloader2
    ga.run(100)
"""

