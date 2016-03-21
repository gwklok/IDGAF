import time
import math
import random
from copy import copy

from idgaf import State, GeneticAlgorithm, Population


def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    # print("a: {} b: {}".format(a,b))
    try:
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    except:
        raise
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


def get_distance_matrix(cities):
    # create a distance matrix
    distance_matrix = {}
    for ka, va in cities.items():
        distance_matrix[ka] = {}
        for kb, vb in cities.items():
            if kb == ka:
                distance_matrix[ka][kb] = 0.0
            else:
                distance_matrix[ka][kb] = distance(va, vb)
    return distance_matrix


class TSPState(State):
    """TSPState

    :param list route: List of cities that form the route
    :param dict cities: Dict of cities with their coordinates
    :param distance_matrix: Pre-computed distance_matrix for cities
    """
    def __init__(self, route, cities, distance_matrix):
        self.route = route
        self.cities = cities
        self.distance_matrix = distance_matrix

    @property
    def fitness(self):
        state = self.route
        f = 0
        for i in range(len(state)):
            f += self.distance_matrix[state[i-1]][state[i]]
        return -f

    def crossover(self, other):
        route_len = len(self.route)
        startpos, endpos = sorted(random.randint(0, route_len) for _ in
                                  range(2))
        child = [None]*route_len
        child[startpos:endpos] = other.route[startpos:endpos]
        child_ptr = endpos if startpos == 0 else 0
        for node in self.route:
            if node not in child:
                child[child_ptr] = node
                child_ptr += 1
                if child_ptr == route_len:
                    break
                if child[child_ptr] is not None:
                    child_ptr = endpos
        # Just in case the algorithm above is broken
        assert len(set(child)) == route_len
        return TSPState(child, self.cities, self.distance_matrix)

    def mutate(self):
        state = self.route
        a = random.randint(0, len(state) - 1)
        b = random.randint(0, len(state) - 1)
        state[a], state[b] = state[b], state[a]

    def new(self, state=None):
        if state is None:
            state = self.route
        state = copy(state)
        random.shuffle(state)
        return TSPState(
            route=state,
            cities=self.cities,
            distance_matrix=self.distance_matrix
        )


def tsp_example():
    from .cities import cities_120 as cities

    distance_matrix = get_distance_matrix(cities)
    initial_state = TSPState(cities.keys(), cities, distance_matrix)
    population = Population(initial_state, population_size=1000,
                            elitism_pct=1.0)
    ga = GeneticAlgorithm(population)

    print("Initial fitness: {}".format(initial_state.fitness))
    start_time = time.time()
    for i, fittest in ga.run(generations=10000, yield_every=10):
        print("Fitness {} at generation {}; runtime: {:.2f}s".format(
            fittest.fitness,
            i,
            time.time() - start_time
        ))
    print("Best fitness: {}".format(ga.fittest.fitness))
