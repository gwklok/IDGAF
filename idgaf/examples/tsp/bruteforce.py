import itertools

from . import TSPState, get_distance_matrix


def bruteforce_tsp(cities, announce_every=None):
    t = TSPState(cities.keys(), cities, get_distance_matrix(cities))

    fittest = [t, t.fitness]
    announce_count = 0
    for p in itertools.permutations(t.cities.keys()):
        announce_count += 1
        t.route = p
        if t.fitness > fittest[1]:
            fittest[1] = t.fitness
        if announce_count:
            if announce_count == announce_every:
                print(fittest[1])
                announce_count = 0
    print("Best fitness: {}".format(fittest[1]))
    return fittest[1]
