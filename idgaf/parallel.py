from multiprocessing import cpu_count, Pool
from collections import namedtuple
from importlib import import_module

import time

from idgaf import Population, GeneticAlgorithm
from idgaf.util import chunks


PopulationClassPath = namedtuple('PopulationClassPath', ['module', 'cls'])

def runner((id, pcp, population_s, generations)):
    """GA runner

    :type pcp: PopulationClassPath
    :param str population_s: serialized Population
    :type generations: int
    :return: serialized evolved population
    :rtype: str
    """
    pcls_module = import_module(pcp.module)
    pcls = getattr(pcls_module, pcp.cls)
    population = pcls.load(population_s)
    ga = GeneticAlgorithm(population)
    for i, fittest in ga.run(generations=generations, yield_every=2):
        print("{}: Generation {} - Fitness {}".format(id, i, fittest.fitness))
    return ga.population.serialize()


class ParallelGAManager(object):
    """ParallelGAManager

    :param list populations: List of serialized populations. Each population
        will be run in a separate process with recombinations occurring
        to make use of results as specified in the run or autorun parameters
    :type pcp: PopulationClassPath
    :type pc: Population.__class__
    """
    def __init__(self, pcp, pc):
        self.pcp = pcp
        self.pc = pc
        self._populationss = None

    def init_populations_from_state(
        self, initial_state, population_size, num_populations=None, *args,
        **kwargs
    ):
        # assert num_populations % 4 == 0, "num_populations must be a " \
        #                                  "multiple of 4"
        self._init_from_state_args = (initial_state, population_size)
        self._pop_init_args = args
        self._pop_init_kwargs = kwargs

        if num_populations is None:
            num_populations = cpu_count()
        populations = [
            self.pc(*args, **kwargs) for i in range(num_populations)
        ]
        for p in populations:
            p.init_from_state(*self._init_from_state_args)
        self.populations = populations

    @property
    def populations(self):
        return [self.pc.load(s) for s in self._populationss]

    @populations.setter
    def populations(self, ps):
        """setter for populations

        :param list ps: list of Population
        """
        self._populationss = [p.serialize() for p in ps]

    def run(self, metagenerations, generations):
        if self._populationss is None:
            raise ValueError("ParallelGAManager is not initalized!"
                             " Please call init_populations_from_state"
                             " to initialize.")
        start = time.time()
        process_pool = Pool()
        for mi in range(metagenerations):
            print("Starting metageneration {}".format(mi))
            if not self._populationss:
                break
            self._populationss = process_pool.map(
                runner,
                [
                    (i, self.pcp, pop, generations) for i, pop in
                    enumerate(self._populationss)
                ]
            )
            populations = self.populations
            print("Metageneration {} complete:\n{}".format(
                mi,
                ["Population {} - Fitness {}".format(i, pop.fittest.fitness)
                 for i, pop in enumerate(populations)]
            ))

            print("Performing recombination of populations...")
            self.populations = self.recombination(populations)
            # self.populations = self.evolve(populations)
            print("Current time: {:.2f}".format(time.time() - start))

        return max(p.fittest.fitness for p in self.populations)

    def recombination(self, populations):
        """Recombine ``populations`` into a list of new populations
        to return

        :type populations: list
        :rtype: list
        """
        populations = populations[:]
        new_populations = []
        r = len(populations) % 3
        for i in range(r):
            new_populations.append(populations.pop())

        for a, b, c in chunks(populations, 3):
            a_generation = a.combine(b)
            b_generation = b.combine(c)
            c_generation = c.combine(a)
            a.generation = a_generation
            b.generation = b_generation
            c.generation = c_generation
            new_populations.extend([a,b,c])
        return new_populations

    # def evolve(self, populations):
    #     len_populations = len(populations)
    #     mega_population = sorted(
    #         [i for pop in populations for i in pop.generation],
    #         key=lambda i: i.fitness, reverse=True
    #     )
    #     mega_population = mega_population[:int(len(mega_population)*0.8)]
    #     generations = [[] for i in range(len_populations)]
    #     for individuals in chunks(mega_population, len_populations):
    #         for individual, generation in zip(individuals,
    #                                           generations):
    #             generation.append(individual)
    #     for population, generation in zip(populations, generations):
    #         population.generation = generation
    #         print("New generation size: {}".format(len(generation)))
    #     return populations
    #
    # def combine_top(self, populations):
    #     len_populations = len(populations)
    #     best_population =
