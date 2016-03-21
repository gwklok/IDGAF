import unittest
import json

import pytest

from idgaf import State
from idgaf.examples.tsp.cities import cities_7
from idgaf.examples.tsp.bruteforce import bruteforce_tsp
from idgaf.examples.tsp import tsp_auto_example, TSPPopulation, \
    get_distance_matrix, TSPState


class TestTSPExample(unittest.TestCase):
    def test_correctness(self):
        ga_fittest = tsp_auto_example(cities_7, minutes=0.02)
        bf_fittest = bruteforce_tsp(cities_7, announce_every=50)
        assert ga_fittest == bf_fittest

    def test_seralize_and_load(self):
        p1 = TSPPopulation()
        distance_matrix = get_distance_matrix(cities_7)
        p1.init_from_state(
            TSPState(cities_7.keys(), cities_7, distance_matrix),
            population_size=5
        )
        s = p1.serialize()
        p2 = TSPPopulation.load(s)
        assert p1._elitism_pct == p2._elitism_pct
        assert p1._tournament_size == p2._tournament_size
        assert tuple(s.route for s in p1.generation) == \
            tuple(s.route for s in p2.generation)
        assert json.dumps(p1.generation[0].cities) == \
               json.dumps(p2.generation[0].cities)
        assert json.dumps(p1.generation[0].distance_matrix) == \
               json.dumps(p2.generation[0].distance_matrix)


def test_state_abc():
    with pytest.raises(TypeError):
        class IncompleteState(State):
            pass
        b = IncompleteState()
