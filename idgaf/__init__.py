import random

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty


class State(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def fitness(self):
        """Fitness of state"""
        raise NotImplementedError

    @abstractmethod
    def crossover(self, other):
        """Returns crossover between this state and another

        :type other: State
        :rtype: State
        """
        raise NotImplementedError

    @abstractmethod
    def mutate(self):
        """Mutates this state in-place"""
        raise NotImplementedError

    @abstractmethod
    def new(self, state=None):
        """Create a new (chaotic) state based upon the provided state

        If the ``state`` given is None, the current state is to be used.

        :type state: State|None
        """
        raise NotImplementedError


class Population(object):
    """Population

    :param State initial_state: The initial state to generate population from
    :param int population_size: Desired size of the population
    :param float elitism_pct: Percentage of population to carried over
        unchanged to the next generation when evolving. 1.0% elitism by
        default. Set this to 0 to disable.
    """

    __metaclass__ = ABCMeta

    TOURNAMENT_SIZE = 5

    def __init__(self, initial_state, population_size,
                 elitism_pct=1.0):
        assert 0.0 <= elitism_pct <= 100.0
        self.elitism_pct = elitism_pct
        assert population_size > 0
        self.population_size = population_size
        assert isinstance(initial_state, State)
        generation = [initial_state.new()
                      for i in range(population_size)]
        self.generation = generation

    ###################
    # Utility Methods #
    ###################

    @staticmethod
    def sort_by_fittest(states):
        """Returns sorted states iterable"""
        return sorted(states, key=lambda s: s.fitness, reverse=True)

    @classmethod
    def tournament(cls, states):
        """Returns the fittest of all given states"""
        # fittest = states[0], states[0].fitness
        # for state in states[1:]:
        #     if state.fitness > fittest[1]:
        #         fittest = state, state.fitness
        # return state
        return cls.sort_by_fittest(states)[0]

    ##################
    # Public Methods #
    ##################

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, gen):
        self._generation = self.sort_by_fittest(gen)

    def evolve(self):
        next_generation = []

        # Add elites to next generation
        elite_size = int(self.elitism_pct * len(self.generation) / 100.0)
        elite = self.generation[:elite_size]
        next_generation.extend(elite)

        # For the non-elites, evolve
        for state in self.generation[elite_size:]:
            other = self.tournament(random.sample(
                self.generation,
                self.TOURNAMENT_SIZE
            ))
            child = state.crossover(other)  # mate
            child.mutate()  # mutate
            next_generation.append(child)

        self.generation = next_generation

    @property
    def fittest(self):
        return self.generation[0]


class GeneticAlgorithm(object):
    """GeneticAlgorithm

    :type population: Population
    """
    def __init__(self, population):
        self.population = population

    @property
    def fittest(self):
        """Returns the fittest state in the population

        :rtype: State
        """
        return self.population.fittest

    def run(self, generations=100, yield_every=None):
        """Run the algorithm for the given number of generations

        :param int generations: Number of generations to run for
        :param int|None yield_every: If a non-zero int is given, this method
            will yield the fittest individual of the population every
            ``yield_every`` generations. If this is set to None, this
            method will not yield
        """
        yield_counter = 0
        for i in range(generations):
            self.population.evolve()
            if yield_every:
                yield_counter += 1
                if yield_counter == yield_every:
                    yield_counter = 0
                    yield i, self.fittest
