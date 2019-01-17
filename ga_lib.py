#
# This is from pyeasyga
#   https://github.com/remiomosowon/pyeasyga/blob/develop/pyeasyga/pyeasyga.py
# with some modifications by Pat Niemeyer (pat@pat.net)
#

import random
import copy
from operator import attrgetter
from typing import Callable

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

        def create_individual()->[]:
            """Create a candidate solution representation.
            :returns: candidate solution representation as a list
            """
            return []

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

        def binary_mutate(individual):
            """Reverse the bit of a random index in an individual."""
            mutate_index = random.randrange(len(individual))
            individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.tournament_selection = tournament_selection
        self.two_point_crossover = two_point_crossover
        self.tournament_size = self.population_size // 10
        self.random_selection = random_selection
        self.create_individual = create_individual

        # Supply this or the simple fitness function.
        # If set, called with an individual and the full population to return the fitness
        self.population_fitness_function: Callable[[Individual, [Individual]], float] = None

        # Supply this or the full signature population fitness function.
        # If set, called with an individual's genes list to return the fitness
        self.fitness_function: Callable[[[]], float] = None

        self.crossover_function = crossover
        self.mutate_function = binary_mutate
        self.selection_function = self.tournament_selection

        self.termination_function: Callable[[GeneticAlgorithm], bool] = None
        self.iteration_function: Callable[[GeneticAlgorithm, int], None] = None
        self.iteration_num: int = None  # the current iteration number

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual()
            individual = Individual(genes)
            initial_population.append(individual)
        self.current_generation = initial_population

    # TODO: delegate and allow us to parallelize
    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            if self.population_fitness_function is None:
                individual.fitness = self.fitness_function(individual.genes)
            else:
                individual.fitness = self.population_fitness_function(individual, self.current_generation)

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)

    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            # choose pairs of parents using selection function twice
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            # children are initially clones of parents
            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

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
        self.create_new_population()
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




