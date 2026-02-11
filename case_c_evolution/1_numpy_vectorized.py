import numpy as np
import time

# --- Konfiguracja ---
DIMENSIONS = 20
POPULATION_SIZE = 10_000
GENERATIONS = 50
MIN_VAL = -5.12
MAX_VAL = 5.12

def rastrigin_vec(X):
    A = 10
    n = X.shape[1]
    s = np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    return A * n + s

def mutate(X, mutation_rate=0.01, sigma=0.1):
    mask = np.random.random(X.shape) < mutation_rate
    changes = np.random.normal(0, sigma, X.shape)
    X[mask] += changes[mask]
    return np.clip(X, MIN_VAL, MAX_VAL)

def run_evolution():
    start_time = time.time()
    population = np.random.uniform(MIN_VAL, MAX_VAL, (POPULATION_SIZE, DIMENSIONS))
    
    for gen in range(GENERATIONS):
        fitness = rastrigin_vec(population)
        
        # Wektoryzacja: argsort O(N log N)
        idx = np.argsort(fitness)
        population = population[idx]
        fitness = fitness[idx]
        
        elites = population[:10]
        
        K = POPULATION_SIZE // 2
        parents_idx = np.random.randint(0, K, (POPULATION_SIZE - 10, 2))
        parents1 = population[parents_idx[:, 0]]
        parents2 = population[parents_idx[:, 1]]
        
        cross_mask = np.random.random((POPULATION_SIZE - 10, DIMENSIONS)) < 0.5
        children = np.where(cross_mask, parents1, parents2)
        children = mutate(children)
        
        population = np.vstack([elites, children])
        
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {fitness[0]:.4f}")

    end_time = time.time()
    print(f"NumPy Vectorized Execution Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    run_evolution()
