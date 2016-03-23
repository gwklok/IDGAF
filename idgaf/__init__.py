import time
import random
from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from itertools import chain


class State(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def fitness(self):
        """Fitness of state"""
        raise NotImplementedError

    @abstractmethod
    def crossover(self, other):
        """Performs in-place crossover between this state and other

        :type other: State
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

    :param float elitism_pct: Percentage of population to carried over
        unchanged to the next generation when evolving. 1.0% elitism by
        default. Set this to 0 to disable.
    :param int tournament_size: Size of tournament pool of individuals
        that compete to crossover
    """
    def __init__(self, elitism_pct=1.0, tournament_size=5):
        assert tournament_size >= 2
        self._tournament_size = tournament_size
        assert 0.0 <= elitism_pct <= 100.0
        self._elitism_pct = elitism_pct
        self._generation = None

    def init_from_state(self, initial_state, population_size):
        """Initialize population from initial_state

        :param State initial_state: The initial state to generate population
            from
        :param int population_size: Desired size of the population
        """
        assert population_size > 0
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
        assert len(self._generation) >= self._tournament_size, \
            "Population size is smaller than tournament size!"

    def evolve(self):
        next_generation = []

        # Add elites to next generation
        elite_size = int(self._elitism_pct * len(self.generation) / 100.0)
        elite = self.generation[:elite_size]
        next_generation.extend(elite)

        # For the non-elites, evolve
        for state in self.generation[elite_size:]:
            other = self.tournament(random.sample(
                self.generation,
                self._tournament_size
            ))
            child = state
            child.crossover(other)  # mate
            child.mutate()  # mutate
            next_generation.append(child)

        self.generation = next_generation

    @property
    def fittest(self):
        return self.generation[0]

    def combine(self, *others):
        """Combines generation with other(s) generation and returns new
        generation

        :param list others: List of Population objects
        :return: new generation
        :rtype: list
        """
        num_populations = len(others)+1
        full_len = len(self.generation)
        part_len = full_len/num_populations
        new = self.generation[:part_len] + \
              [state for other in others for
               state in other.generation[:part_len]]
        print("Expected size {}; actual size {}".format(full_len, len(new)))
        len_diff = full_len - len(new)
        if len_diff:
            new.extend(self.generation[:len_diff])
        assert len(new) == full_len
        return new

    #########
    # Stubs #
    #########

    def serialize(self):
        """Stub method to be used as an efficient serializer
        for an entire population

        The returned value from this method should be able to used
        by load to create a new Population instance

        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def load(cls, s):
        raise NotImplementedError


class GeneticAlgorithm(object):
    """GeneticAlgorithm

    :type population: Population
    """
    def __init__(self, population):
        if population.generation is None:
            raise ValueError("Population is not initalized!")
        self.population = population

    @property
    def fittest(self):
        """Returns the fittest state in the population

        :rtype: State
        """
        return self.population.fittest

    def run(self, generations=100, yield_every=None):
        """Run the algorithm for the given number of ``generations``

        :param int generations: Number of generations to run for
        :param int|None yield_every: If a non-zero int is given, this method
            will yield the (generation, fittest individual) of
            the population  every ``yield_every`` generations. If this is
            set to None, this method will not yield.
        """
        assert generations >= 1
        assert isinstance(yield_every, (int, type(None)))
        yield_counter = 0
        for i in range(generations):
            self.population.evolve()
            if yield_every:
                yield_counter += 1
                if yield_counter == yield_every:
                    yield_counter = 0
                    yield i, self.fittest

    def autorun(self, minutes, yield_every=None):
        """Automatically run GA for the given ``minutes``, approximately

        :param int|None yield_every: If this is set, this method will
            yield every ``yield_every`` seconds. (Seconds, not minutes).
        """
        generations = self.evolutions_in(minutes=minutes)
        seconds_per_evolution = (minutes*60.0)/generations
        if yield_every:
            yield_every = int(yield_every / seconds_per_evolution)
        for i, fittest in self.run(generations=generations,
                                   yield_every=yield_every):
            yield i, fittest

    def evolutions_in(self, minutes, test_generations=5):
        """Determines the number of generations possible in
        given time ``minutes``

        :type minutes: float
        :param int test_generations: Number of generations used for
            testing
        """
        seconds = minutes * 60.0
        start = time.time()
        for i, _ in self.run(generations=test_generations):
            pass
        time_per_gen = (time.time() - start)/test_generations
        generations = int(seconds/time_per_gen)
        return generations
