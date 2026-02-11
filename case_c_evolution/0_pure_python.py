import random
import math
import time

DIMENSIONS = 20
POPULATION_SIZE = 10_000
SELECT_TOP_K = 50
ELITE = 10
GENERATIONS = 100
MIN_VAL = -5.12
MAX_VAL = 5.12


def rastrigin(vector):
    """Rastrigin function
    https://en.wikipedia.org/wiki/Rastrigin_function"""
    A = 10
    n = len(vector)
    s = 0
    for x in vector:
        s += x**2 - A * math.cos(2 * math.pi * x)
    return A * n + s


class Individual:
    def __init__(self, vector=None):
        if vector is None:
            self.vector = [random.uniform(MIN_VAL, MAX_VAL) for _ in range(DIMENSIONS)]
        else:
            self.vector = vector
        self.fitness = rastrigin(self.vector)


def crossover(p1, p2):
    """One-point crossover"""
    point = random.randint(1, DIMENSIONS - 1)
    child_vec = p1.vector[:point] + p2.vector[point:]
    return Individual(child_vec)


def mutate(ind, mutation_rate=0.01, sigma=0.1):
    """Gaussian mutation"""
    new_vec = list(ind.vector)
    for i in range(DIMENSIONS):
        if random.random() < mutation_rate:
            new_vec[i] += random.gauss(0, sigma)
            new_vec[i] = max(MIN_VAL, min(MAX_VAL, new_vec[i]))
    return Individual(new_vec)


def run_evolution():
    start_time = time.time()
    population = [Individual() for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        population.sort(key=lambda x: x.fitness)

        elites = population[:ELITE]
        next_gen = elites

        while len(next_gen) < POPULATION_SIZE:
            p1 = random.choice(population[:SELECT_TOP_K])
            p2 = random.choice(population[:SELECT_TOP_K])
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen

        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {population[0].fitness:.4f}")

    end_time = time.time()
    print(f"Pure Python Execution Time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    run_evolution()
