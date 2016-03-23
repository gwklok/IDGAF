import random
import time
from multiprocessing import cpu_count, Pool
from collections import namedtuple
from importlib import import_module

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
        assert num_populations % 4 == 0, "num_populations must be a " \
                                         "multiple of 4"
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

    def run(self, target_fitness=None, metagenerations=None, generations=100):
        if self._populationss is None:
            raise ValueError("ParallelGAManager is not initalized!"
                             " Please call init_populations_from_state"
                             " to initialize.")
        start = time.time()
        process_pool = Pool()

        if target_fitness and not metagenerations:
            import itertools
            mgs = itertools.count()
            print("Running until target fitness of {} reached..."
                  .format(target_fitness))
        elif target_fitness and metagenerations:
            print("Running until {} metagenerations complete OR "
                  "until target fitness of {} reached..."
                  .format(metagenerations, target_fitness))
            mgs = range(metagenerations)
        elif metagenerations and not target_fitness:
            print("Running until {} metagenerations complete..."
                  .format(metagenerations))
            mgs = range(metagenerations)
        else:
            raise ValueError

        for mi in mgs:
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
            print("Current time: {:.2f}".format(time.time() - start))
            best_fitness = max(p.fittest.fitness for p in populations)
            print("Best current fitness: {}".format(best_fitness))
            if target_fitness:
                if best_fitness >= target_fitness:
                    print("We beat target fitness of {}!"
                          .format(target_fitness))
                break

        return max(p.fittest.fitness for p in self.populations)

    def recombination(self, populations):
        populations = populations[:]
        random.shuffle(populations)

        new_pops = []
        for pops in chunks(populations, 4):
            if len(pops) != 4:
                raise ValueError
            pop1, pop2, pop3, pop4 = pops

            new_gen = pop1.combine(pop4)
            pop1.generation = new_gen
            new_gen = pop1.combine(pop2)
            pop1.generation = new_gen
            new_gen = pop1.combine(pop3)
            pop1.generation = new_gen
            new_pops.append(pop1)

            # TODO need original not recombinated gens here
            new_gen = pop2.combine(pop4)
            pop2.generation = new_gen
            new_gen = pop2.combine(pop3)
            pop2.generation = new_gen
            new_gen = pop2.combine(pop1)
            pop2.generation = new_gen
            new_pops.append(pop2)

            # TODO change this combination of ALL best not just
            #  best of these 4
            gen_len = len(pop3.generation)/len(pops)
            gens = [p.generation[:gen_len] for p in pops]
            new_gen = []
            for gen in gens:
                new_gen.extend(gen)
            len_diff = self._init_from_state_args[1] - len(new_gen)
            if len_diff:
                new_gen.extend(new_gen[:len_diff])
            assert len(new_gen) == self._init_from_state_args[1]
            pop3.generation = new_gen
            new_pops.append(pop3)

            pop4.generation = pop4.combine(pop3)
            new_pops.append(pop4)
        return new_pops
