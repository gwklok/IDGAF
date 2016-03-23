import time
import math
import random
import json
from copy import copy

from idgaf import State, GeneticAlgorithm, Population
from idgaf.parallel import ParallelGAManager, PopulationClassPath, runner


def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
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
        # self.start_node = start_node
        # assert self.route[0] == self.start_node
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
        #return TSPState(child, self.cities, self.distance_matrix)
        self.route = child
        return self

    def crossover2(self, other):
        d = {}
        my_route = self.route[:]
        my_start = self.route[0]
        other_index = other.route.index(my_start)
        other_route = other.route[other_index:]+other.route[:other_index]

        for i, node in enumerate(my_route[:-1]):
            d[node] = {"self": my_route[i+1]}
        d[my_route[-1]] = {"self": my_start}
        for i, node in enumerate(other_route[:-1]):
            try:
                d[node]["other"] = other_route[i+1]
            except KeyError:
                d[node] = {"other": other_route[i+1]}
        d[other_route[-1]]["other"] = my_start

        # child_connections = {}
        # ssroute = set(my_route)
        # for i, node in enumerate(sorted(other_route[:-1])):
        #     ccv = set(child_connections.values())
        #     candidates = d[node].values()
        #     random.shuffle(candidates)
        #     for c in candidates:
        #         if c not in ccv:
        #             if node == c or c == my_start or child_connections.get(
        #                     node) == c or (c == other_route[-1] and (i!=len(
        #                 my_route)-2)) or child_connections.get(c) in \
        #                     child_connections:
        #                 continue
        #             child_connections[node] = c
        #             print(1)
        #             break
        #     else:
        #         child_connections[node] = random.choice(list(
        #             ssroute - ccv - {my_start}))
        #         print(2)
        #     print(child_connections)
        # assert len(child_connections) == len(my_route)-1

        # child = [my_start]
        # node = my_start
        # while node:
        #     node = child_connections.get(node)
        #     child.append(node)
        # assert child[-1] is None
        # child = child[:-1]
        # assert len(child) == len(my_route)
        # assert len(set(child)) == len(child)
        # # return TSPState(child, self.cities, self.distance_matrix)
        # self.route = child
        # return self

        # TODO this might work well but might have a bug; double-check later
        new_route = [my_start]
        #current = my_start
        destinations = my_route[1:]
        while destinations:
            current = new_route[-1]
            candidates = d[current].values()
            random.shuffle(candidates)
            for c in candidates:
                if c not in destinations:
                    continue
            else:
                c = random.choice(destinations)
            new_route.append(c)
            destinations.remove(c)
        assert len(new_route) == len(my_route)
        assert len(set(new_route)) == len(new_route)
        #self.route = new_route
        #return self
        return TSPState(new_route, self.cities, self.distance_matrix)


    def crossover3(self, other):
        """

        :type other: TSPState
        :param other:
        :return:
        """
        route_len = len(self.route)
        cities_to_replace = set(random.sample(
            self.route,
            random.randint(int(round(route_len/4.0)), int(round(
                route_len*3/4.0))-1)
        ))
        other_route = [node for node in other.route
                       if node in cities_to_replace]
        # child = self.route[:]
        # for i, node in enumerate(child[:]):
        #     if not other_route:
        #         break
        #     if node == other_route[0]:
        #         child[i] = other_route.pop(0)

        # _my_route = self.route[:]
        # _other_route = other.route[:]
        child = []
        nodes = self.route[:]
        while nodes:
            node = nodes[0]
            if node in cities_to_replace:
                other_node = other_route.pop(0)
                child.append(other_node)
                nodes.remove(other_node)
            else:
                child.append(nodes.pop(0))

        #return TSPState(child, self.cities, self.distance_matrix)
        self.route = child
        return self

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


class TSPPopulation(Population):

    def serialize(self):
        sample_state=self.generation[0]
        d = dict(
            routes=[state.route for state in self.generation],
            cities=sample_state.cities,
            distance_matrix=sample_state.distance_matrix,
            elitism_pct=self._elitism_pct,
            tournament_size=self._tournament_size
        )
        return json.dumps(d)

    @classmethod
    def load(cls, s):
        d = json.loads(s)
        kwargs = {k: v for k, v in d.items() if k in ("elitism_pct",
                                                      "tournament_size")}
        p = cls(**kwargs)
        cities = d['cities']
        distance_matrix = d['distance_matrix']
        generation = [TSPState(route, cities, distance_matrix) for route in
                      d['routes']]
        p.generation = generation
        return p


def tsp_auto_example(cities=None, minutes=1, yield_every=10):
    if cities is None:
        from .cities import cities_120 as cities

    distance_matrix = get_distance_matrix(cities)
    initial_state = TSPState(cities.keys(), cities, distance_matrix)
    population = Population(elitism_pct=1.0)
    population.init_from_state(initial_state, population_size=500)
    ga = GeneticAlgorithm(population)

    print("Initial fitness: {}".format(initial_state.fitness))
    generations = ga.evolutions_in(minutes=minutes)
    print("GA will run for {} generations".format(generations))
    start_time = time.time()
    for i, fittest in ga.autorun(
        minutes=minutes,
        yield_every=yield_every
    ):
        print("Fitness {} at generation {}; runtime: {:.2f}s".format(
            fittest.fitness,
            i,
            time.time() - start_time
        ))
    print("Best fitness: {}".format(ga.fittest.fitness))
    return ga.fittest.fitness


def tsp_parallel_test():
    from .cities import cities_120 as cities
    distance_matrix = get_distance_matrix(cities)
    initial_state = TSPState(cities.keys(), cities, distance_matrix)

    pcp = PopulationClassPath("idgaf.examples.tsp", "TSPPopulation")
    pc = TSPPopulation
    pgam = ParallelGAManager(pcp, pc)
    pgam.init_populations_from_state(initial_state=initial_state,
                                     population_size=500, num_populations=8)
    start = time.time()
    fitness = pgam.run(30, 10)
    print("Best fitness: {}; took time {:.2f}s to find".format(
        fitness,
        time.time() - start
    ))
