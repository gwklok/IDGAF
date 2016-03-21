from multiprocessing import Process, cpu_count, Pool
from collections import namedtuple
from importlib import import_module

from idgaf import Population, GeneticAlgorithm


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
    for i, fittest in ga.run(generations=generations, yield_every=10):
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
        if num_populations is None:
            num_populations = cpu_count()
        populations = [
            self.pc(*args, **kwargs) for i in range(num_populations)
        ]
        for p in populations:
            p.init_from_state(initial_state, population_size)
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
        process_pool = Pool()
        for mi in range(metagenerations):
            self._populationss = process_pool.map(
                runner,
                [
                    (i, self.pcp, pop, generations) for i, pop in
                    enumerate(self._populationss)
                ]
            )
            populations = self.populations
            print(populations)
            print("Metageneration {} complete:\n{}".format(
                mi,
                ["Population {} - Fitness {}".format(i, pop.fittest.fitness)
                 for i, pop in enumerate(populations)]
            ))
        return max(p.fittest.fitness for p in self.populations)
