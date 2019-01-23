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

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')
if not os.path.exists('./out/'): os.mkdir('./out/')
f=open('loss.txt', 'w'); f.truncate(); f.close()

# e.g. [128,784]->[128,28,28], [-1,1] to [0,1]
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # [0-1] -> [-1,1]
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
    def getWeights(self) -> [torch.Tensor]:
        for module in chain(self.encoder.modules(), self.decoder.modules()):
            for param in module.parameters():
                yield param.data

    # Set weights and biases from the list
    def setWeights(self, weights: [torch.Tensor]):
        weight_data = iter(weights)
        for module in chain(self.encoder.modules(), self.decoder.modules()):
            for param in module.parameters():
                param.data = next(weight_data)


def fitness(weights, save=False):
    model.setWeights(weights)
    epochloss=0
    for batch in traindataloader:
        img, _ = batch  # [128, 1, 28, 28]
        img = img.view(img.size(0), -1)  # flat [128, 784]
        if torch.cuda.is_available(): img = img.cuda()
        output = model(img)  # [128, 784]

        # image loss
        batch_loss = criterion(output, img).item() # scalar
        epochloss += batch_loss

    if save:
        # output the first two image/output pairs
        images=[
            img[0].view(1,28,28), to_img(output[0:1]).squeeze().unsqueeze(0),
            img[1].view(1,28,28), to_img(output[1:2]).squeeze().unsqueeze(0),
        ]
        save_image(images, './out/image_{}.png'.format(ga.iteration_num))

    return epochloss

def create_individual()->[torch.Tensor]:
    return [(torch.rand_like(weight)-0.5)/25 for weight in weight_start]

def mutate_all_weights(individual: [torch.Tensor]):
    for weight in individual:
        mutate_weight(weight)

def mutate_one_weight(individual: [torch.Tensor]):
    mutate_weight(random.choice(individual))

def mutate_weight(weight: torch.Tensor):
    # Ensure that we have a variety of mutation sizes.
    length = len(weight.view(-1))
    target_gene_mutation_count = int(max(1, random.choice([1, 1, 1, length/100, length/10, length])))
    per_gene_mutation_rate = target_gene_mutation_count / length

    # Probably only need one (offset) mutation type here but trying everything
    type = random.choice(['scale', 'offset', 'replace'])
    if type == 'scale':
        if target_gene_mutation_count <= 10:  # a bit faster for small mutations
            for _ in range(target_gene_mutation_count):
                i=random.randint(0,length-1)
                weight.view(-1)[i] *= 1.0 + np.random.randn() / 5.0
        else:
            maskrand = torch.rand_like(weight)
            mask = maskrand.lt(per_gene_mutation_rate).float()
            rand = 1.0 + torch.randn_like(weight) / 5.0
            weight *= rand * mask

    elif type == 'offset':
        if target_gene_mutation_count <= 10:
            for _ in range(target_gene_mutation_count):
                i=random.randint(0,length-1)
                weight.view(-1)[i] += np.random.randn() / 50.0
        else:
            maskrand = torch.rand_like(weight)
            mask = maskrand.lt(per_gene_mutation_rate).float()
            rand = torch.randn_like(weight) / 50.0
            weight += rand * mask

    elif type == 'replace':
        if target_gene_mutation_count <= 10:
            for _ in range(target_gene_mutation_count):
                i=random.randint(0,length-1)
                weight.view(-1)[i] = np.random.randn() / 5.0
        else:
            maskrand = torch.rand_like(weight)
            mask = maskrand.lt(per_gene_mutation_rate).float()
            rand = torch.randn_like(weight) / 5.0
            weight -= weight*mask
            weight += rand*mask

def two_point_crossover_per_weight(ind1: [torch.Tensor], ind2: [torch.Tensor]):
    for i in range(min(len(ind1), len(ind2))):
        w1, w2 = ind1[i], ind2[i]
        c1, c2 = ga.two_point_crossover(w1.view(-1), w2.view(-1))
        ind1[i], ind2[i] = c1.view_as(w1), c2.view_as(w2)
    return ind1, ind2

def two_point_crossover_one_weight(ind1: [torch.Tensor], ind2: [torch.Tensor]):
    i=random.randint(0, min(len(ind1), len(ind2))-1)
    w1, w2 = ind1[i], ind2[i]
    c1, c2 = ga.two_point_crossover(w1.view(-1), w2.view(-1))
    ind1[i], ind2[i] = c1.view_as(w1), c2.view_as(w2)
    return ind1, ind2

start_time = time()
def report(ga: GeneticAlgorithm, i: int):
    global start_time
    end_time = time()
    print(i, ga.best_individual().fitness, end_time - start_time);
    f=open('loss.txt', 'a'); print(ga.best_individual().fitness, file=f); f.close()
    start_time = time()
    if i % 25 == 0:
        fitness(ga.best_individual().genes, save=True)

batch_size = 128
traindataset = MNIST('./data', train=False, transform=img_transform, download=True)
traindataset = list(traindataset)[:128]  # limit images
# traindataset = list(traindataset)[:2]  # limit images

# Subtract mean of training data images
data=torch.stack([d[0] for d in traindataset]) # one big batch [128, 1, 28, 28])
mean_img=torch.mean(data,0,keepdim=True)  # batch average image
traindataset = [(batch-mean_img, clas) for batch, clas in traindataset]

traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)
model = autoencoder()
if torch.cuda.is_available(): model.cuda()
criterion = nn.MSELoss(reduction='sum')
# criterion = nn.MSELoss()
weight_start = list(model.getWeights())

ga = GeneticAlgorithm(
    population_size=200,
    crossover_probability=0.0, # crossover doesn't seem to help
    mutation_probability=1.0,
    elitism=True,
    maximise_fitness=False)

ga.create_individual = create_individual
ga.selection_function = ga.tournament_selection
# ga.selection_function = ga.elite_selection
ga.crossover_function = two_point_crossover_one_weight
ga.mutate_function = mutate_one_weight
ga.fitness_function = fitness
ga.iteration_function = report

ga.run(50000)

