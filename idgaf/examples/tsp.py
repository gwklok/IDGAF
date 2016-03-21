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

    def fitness(self):
        state = self.route
        f = 0
        for i in range(len(state)):
            f += self.distance_matrix[state[i-1]][state[i]]
        return f

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
                if child[child_ptr] is not None:
                    child_ptr = endpos
        # Just in case the algorithm above is broken
        assert len(set(child)) == route_len
        self.route = child

    def mutation(self):
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
    # latitude and longitude for the twenty largest U.S. cities
    cities = {
        'New York City': (40.72, 74.00),
        'Los Angeles': (34.05, 118.25),
        'Chicago': (41.88, 87.63),
        'Houston': (29.77, 95.38),
        'Phoenix': (33.45, 112.07),
        'Philadelphia': (39.95, 75.17),
        'San Antonio': (29.53, 98.47),
        'Dallas': (32.78, 96.80),
        'San Diego': (32.78, 117.15),
        'San Jose': (37.30, 121.87),
        'Detroit': (42.33, 83.05),
        'San Francisco': (37.78, 122.42),
        'Jacksonville': (30.32, 81.70),
        'Indianapolis': (39.78, 86.15),
        'Austin': (30.27, 97.77),
        'Columbus': (39.98, 82.98),
        'Fort Worth': (32.75, 97.33),
        'Charlotte': (35.23, 80.85),
        'Memphis': (35.12, 89.97),
        'Baltimore': (39.28, 76.62)
    }
    distance_matrix = get_distance_matrix(cities)
    initial_state = TSPState(cities.keys(), cities, distance_matrix)
    population = Population(initial_state, population_size=100,
                            elitism_pct=1.0)
    ga = GeneticAlgorithm(population)
    for i, fittest in ga.run(generations=100, yield_every=10):
        print("Fitness {} at generation {}".format(
            fittest.fitness,
            i
        ))
