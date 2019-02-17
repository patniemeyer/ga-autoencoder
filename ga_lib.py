#
# This is from pyeasyga
#   https://github.com/remiomosowon/pyeasyga/blob/develop/pyeasyga/pyeasyga.py
# with some modifications by Pat Niemeyer (pat@pat.net)
#

import random
import copy
from operator import attrgetter
from typing import Callable
import numpy as np

class Individual(object):
    """ class that encapsulates an individual's fitness and solution representation.
    """
    def __init__(self, genes):
        self.genes: [] = genes
        self.fitness: float = 0

    def __repr__(self):
        """Return initialised representation in human readable form.
        """
        return repr((self.fitness, self.genes))

class GeneticAlgorithm(object):

    def __init__(self,
                 population_size=50,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 elitism=True,
                 maximise_fitness=True):
        """Instantiate the Genetic Algorithm.
        :param int population_size: size of population
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation

        """

        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness

        self.current_generation: [Individual] = []  # ranked individuals

        def create_individual()->Individual:
            """Create a candidate solution representation.
            :returns: candidate solution representation as a list
            """
            return Individual([])

        def crossover(parent_1: [], parent_2: [])->([], []):
            """Crossover (mate) two parents to produce two children.

            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children

            """
            index = random.randrange(1, len(parent_1))
            child_1 = parent_1[:index] + parent_2[index:]
            child_2 = parent_2[:index] + parent_1[index:]
            return child_1, child_2

        def two_point_crossover(ind1, ind2):
            size = min(len(ind1), len(ind2))
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1 # Ensure separated by at least one
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1 # Swap the two cx points

            ind1_chunk = copy.deepcopy(ind1[cxpoint1:cxpoint2]) # works with tensors
            ind1[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2]
            ind2[cxpoint1:cxpoint2] = ind1_chunk
            return ind1, ind2

        def binary_mutate(individual: Individual):
            """Reverse the bit of a random index in an individual."""
            genes = individual.genes
            mutate_index = random.randrange(len(genes))
            genes[mutate_index] = (0, 1)[genes[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0: self.tournament_size = 2
            members = random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        def elite_selection(population):
            elite_frac=0.1
            return random.choice(population[0:int(len(population)*elite_frac)])

        # todo: I'm sure this can be faster
        def roulette_selection(population):
            if self.maximise_fitness:
                max = sum([i.fitness for i in population])
                probs = [c.fitness / max for c in population]
            else:
                max = sum([1.0 / i.fitness for i in population])
                probs = [(1.0 / i.fitness) / max for i in population]
            return population[np.random.choice(len(population), p=probs)]

        self.tournament_selection = tournament_selection
        self.elite_selection = elite_selection
        self.roulette_selection = roulette_selection
        self.two_point_crossover = two_point_crossover
        # self.tournament_size = self.population_size / 10
        self.tournament_size = 2
        self.random_selection = random_selection
        self.create_individual = create_individual

        # Supply this or the simple fitness function.
        # If set, called with an individual and the full population to return the fitness
        self.population_fitness_function: Callable[[Individual, [Individual]], float] = None

        # Supply this or the full population fitness function.
        self.fitness_function: Callable[[Individual], float] = None

        self.mutate_function: Callable[[Individual], None] = binary_mutate

        self.crossover_function = crossover
        self.selection_function = self.tournament_selection

        self.termination_function: Callable[[GeneticAlgorithm], bool] = None
        self.iteration_function: Callable[[GeneticAlgorithm, int], None] = None
        self.iteration_num: int = None  # the current iteration number

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            individual = self.create_individual()
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            if self.population_fitness_function is None:
                individual.fitness = self.fitness_function(individual)
            else:
                individual.fitness = self.population_fitness_function(individual, self.current_generation)

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def evolve_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            # choose pairs of parents using selection function twice
            # TODO: Copy here is expensive.
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            # children are initially clones of parents
            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            if random.random() < self.crossover_probability:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if random.random() < self.mutation_probability:
                self.mutate_function(child_1)
                self.mutate_function(child_2)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.evolve_population()
        self.calculate_population_fitness()
        self.rank_population()

    def run(self, generations):
        """Run (solve) the Genetic Algorithm."""
        if len(self.current_generation) == 0:
            self.create_first_generation()

        for i in range(generations):
            self.iteration_num = i
            self.create_next_generation()
            if self.iteration_function:
                self.iteration_function(self, i)
            if self.termination_function and self.termination_function(self):
                break

    def best_individual(self) -> Individual:
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return best

    def worst_individual(self) -> Individual:
        """Return the individual with the best fitness in the current
        generation.
        """
        worst = self.current_generation[len(self.current_generation)-1]
        return worst




