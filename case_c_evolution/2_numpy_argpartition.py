import numpy as np
import time

DIMENSIONS = 20
POPULATION_SIZE = 10_000
SELECT_TOP_K = 50
ELITE = 10
GENERATIONS = 100
MIN_VAL = -5.12
MAX_VAL = 5.12


def rastrigin(X):
    A = 10
    n = X.shape[1]
    s = np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    return A * n + s


def crossover(parents1, parents2):
    """One-point crossover (vectorized)"""
    cut_points = np.random.randint(1, DIMENSIONS, (parents1.shape[0], 1))
    mask = np.arange(DIMENSIONS) < cut_points
    return np.where(mask, parents1, parents2)


def mutate(X, mutation_rate=0.01, sigma=0.1):
    """Gaussian mutation"""
    mask = np.random.random(X.shape) < mutation_rate
    changes = np.random.normal(0, sigma, X.shape)
    X[mask] += changes[mask]
    return np.clip(X, MIN_VAL, MAX_VAL)


def run_evolution():
    start_time = time.time()
    population = np.random.uniform(MIN_VAL, MAX_VAL, (POPULATION_SIZE, DIMENSIONS))

    for gen in range(GENERATIONS):
        fitness = rastrigin(population)

        idx = np.argpartition(fitness, SELECT_TOP_K)
        population = population[idx]
        fitness = fitness[idx]

        # Sort only top K to get elites and best fitness correctly
        top_k_idx = np.argsort(fitness[:SELECT_TOP_K])
        population[:SELECT_TOP_K] = population[top_k_idx]
        fitness[:SELECT_TOP_K] = fitness[top_k_idx]

        elites = population[:ELITE]

        parents_idx = np.random.randint(0, SELECT_TOP_K, (POPULATION_SIZE - ELITE, 2))
        parents1 = population[parents_idx[:, 0]]
        parents2 = population[parents_idx[:, 1]]

        children = crossover(parents1, parents2)
        children = mutate(children)

        population = np.vstack([elites, children])

        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {fitness[0]:.4f}")

    end_time = time.time()
    print(f"NumPy Argpartition Execution Time: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    run_evolution()
