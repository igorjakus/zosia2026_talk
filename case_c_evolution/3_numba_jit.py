import numpy as np
import time
from numba import njit, prange

# --- Konfiguracja ---
DIMENSIONS = 20
POPULATION_SIZE = 10_000
GENERATIONS = 50
MIN_VAL = -5.12
MAX_VAL = 5.12

@njit(parallel=True)
def rastrigin_numba(X):
    pop_size, dims = X.shape
    fitness = np.empty(pop_size, dtype=np.float64)
    A = 10.0
    for i in prange(pop_size):
        s = A * dims
        for j in range(dims):
            val = X[i, j]
            s += val**2 - A * np.cos(2 * np.pi * val)
        fitness[i] = s
    return fitness

@njit(parallel=True)
def mutate_numba(X, mutation_rate=0.01, sigma=0.1):
    pop_size, dims = X.shape
    for i in prange(pop_size):
        for j in range(dims):
            if np.random.random() < mutation_rate:
                X[i, j] += np.random.normal(0, sigma)
                X[i, j] = max(MIN_VAL, min(MAX_VAL, X[i, j]))

def run_evolution():
    # Warmup
    dummy = np.random.uniform(MIN_VAL, MAX_VAL, (10, DIMENSIONS))
    rastrigin_numba(dummy)
    mutate_numba(dummy)
    
    start_time = time.time()
    population = np.random.uniform(MIN_VAL, MAX_VAL, (POPULATION_SIZE, DIMENSIONS))
    
    for gen in range(GENERATIONS):
        fitness = rastrigin_numba(population)
        
        # Numba + argpartition O(N)
        K = POPULATION_SIZE // 2
        idx = np.argpartition(fitness, (10, K))
        population = population[idx]
        fitness = fitness[idx]
        
        elites = population[:10]
        
        parents_idx = np.random.randint(0, K, (POPULATION_SIZE - 10, 2))
        parents1 = population[parents_idx[:, 0]]
        parents2 = population[parents_idx[:, 1]]
        
        cross_mask = np.random.random((POPULATION_SIZE - 10, DIMENSIONS)) < 0.5
        children = np.where(cross_mask, parents1, parents2)
        mutate_numba(children)
        
        population = np.vstack([elites, children])
        
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {np.min(fitness[:10]):.4f}")

    end_time = time.time()
    print(f"Numba JIT Execution Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    run_evolution()
