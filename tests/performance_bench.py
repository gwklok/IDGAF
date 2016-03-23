from idgaf.examples.tsp import tsp_auto_example, tsp_parallel_test
from idgaf.examples.tsp.cities import cities_7, cities_120



#tsp_auto_example(minutes=3, cities=cities_120)
tsp_parallel_test(cities=cities_120, metagenerations=None,
                  target_fitness=-300000)
