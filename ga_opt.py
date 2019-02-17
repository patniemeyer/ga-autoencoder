from util import *
from copy import deepcopy
from scipy import stats
from ga_lib import *
from collections import deque

class GAOptimizer(object):

    def __init__(self, ga, model, loaded_model, weight_start, criterion, batches, fitness, sigma):
        super().__init__()
        self.ga = ga
        self.model = model
        self.loaded_model = loaded_model
        self.weight_start = weight_start
        self.criterion = criterion
        self.batches = batches
        self.sigma = sigma
        self.fitness = fitness

        self.start_time = millis()
        self.past_fitnesses = deque(maxlen=10) #; past_fitnesses.append(initial_fitness)
        self.initial_fitness = None

    def create_individual(self)->Individual:
        # If we have loaded a model we need to start with its weights.
        # todo: how do we restore diversity? Save it? Randomize?
        if self.loaded_model:
            weights = deepcopy(self.weight_start)
        else:
            weights = [(torch.rand_like(weight)-0.5) * 0.2 for weight in self.weight_start]

        individual = Individual(weights)
        individual.last_mutation_count = 0
        return individual

    def mutate_all_weights(self, individual: Individual):
        for i in range(len(individual.genes)):
            self.mutate_weight(individual, weight_index=i)

    def mutate_one_weight(self, individual: Individual):
        self.mutate_weight(individual, weight_index=random.choice(range(len(individual.genes))))

    def mutate_weight(self, individual: Individual, weight_index: int):
        weight=individual.genes[weight_index]
        length = len(weight.view(-1))
        # target_gene_mutation_count = length
        # target_gene_mutation_count = max(1, int(length * 0.1))
        # target_gene_mutation_count = 1
        target_gene_mutation_count = max(1,random.choice([1, 1, 1, 2, 3, 4, 5, int(length*0.1), int(length*0.5)]))
        individual.last_mutation_count = target_gene_mutation_count

        if target_gene_mutation_count <= 10:
            for _ in range(target_gene_mutation_count):
                i=random.randint(0,length-1)
                weight.view(-1)[i] += np.random.randn() * self.sigma
        else:
            per_gene_mutation_rate = target_gene_mutation_count / length
            maskrand = torch.rand_like(weight)
            mask = maskrand.lt(per_gene_mutation_rate).float()
            rand = torch.randn_like(weight) * self.sigma
            weight += rand * mask

    def two_point_crossover_per_weight(self, ind1: [torch.Tensor], ind2: [torch.Tensor]):
        for i in range(min(len(ind1), len(ind2))):
            w1, w2 = ind1[i], ind2[i]
            c1, c2 = ga.two_point_crossover(w1.view(-1), w2.view(-1))
            ind1[i], ind2[i] = c1.view_as(w1), c2.view_as(w2)
        return ind1, ind2

    def two_point_crossover_one_weight(self, ind1: [torch.Tensor], ind2: [torch.Tensor]):
        i=random.randint(0, min(len(ind1), len(ind2))-1)
        w1, w2 = ind1[i], ind2[i]
        c1, c2 = self.ga.two_point_crossover(w1.view(-1), w2.view(-1))
        ind1[i], ind2[i] = c1.view_as(w1), c2.view_as(w2)
        return ind1, ind2

    def swap_two_layers(self, ind1: [torch.Tensor], ind2: [torch.Tensor]):
        i=random.randint(0, min(len(ind1), len(ind2))-1)
        w1, w2 = ind1[i], ind2[i]
        ind1[i]=w2; ind2[i]=w1
        return ind1, ind2

    def differential_crossover(self, ind1: [torch.Tensor], ind2: [torch.Tensor]):
        for w1, w2 in zip(ind1, ind2):
            diff = w2-w1
            w1 += diff * random.random()
            w2 -= diff * random.random()
        return ind1, ind2

    def report(self, ga: GeneticAlgorithm, epoch: int):
        end_time = millis()
        best_fitness = ga.best_individual().fitness
        # mean_fitness = np.median([ind.fitness for ind in ga.current_generation])
        if not self.initial_fitness: self.initial_fitness = best_fitness
        self.past_fitnesses.append(best_fitness)

        f = open('loss.txt', 'a'); print(best_fitness, file=f); f.close()

        # Save images
        if epoch % 100 == 0:
            self.fitness(ga.best_individual(), save=True)
            if best_fitness < self.initial_fitness:
                self.model.save(); print("saved fitness: ", best_fitness)

        # Show stats
        if epoch > 0 and epoch % 1 == 0:
            slope, _, _, _, _ = stats.linregress(x=np.arange(len(self.past_fitnesses)), y=self.past_fitnesses)
            rate=1000/(end_time-self.start_time)
            eta=-best_fitness/slope/rate
            # print('e: %d, bf: %.10f, mf: %.10f, slope: %.10f, fps: %.3f, eta: %.1fs'
            #       % (epoch, best_fitness, mean_fitness, slope, rate, eta))
            print('e: %d, bf: %.10f, slope: %.10f, fps: %.3f, eta: %.1fs'
                  % (epoch, best_fitness, slope, rate, eta))

        # Save sample images for entire population
        """
        if epoch % 10 == 0:
            images=[]
            show_count=ga.population_size
            for i in range(show_count):
                images.append(ga.current_generation[i].sample_img)
            save_image(images, './out/population_{}.png'.format(epoch), nrow=10)
        """

        self.start_time = millis() # for next round




