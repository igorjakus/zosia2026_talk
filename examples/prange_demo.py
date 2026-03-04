import numpy as np
import time
from numba import njit, prange

@njit
def compute_distance(x, y):
    s = 0.0
    for j in range(x.shape[0]):
        s += (x[j] - y[j]) ** 2
    return s ** 0.5

@njit
def distances_serial(X, Y):
    n = X.shape[0]
    result = np.empty(n)
    for i in range(n):
        result[i] = compute_distance(X[i], Y[i])
    return result

@njit(parallel=True)
def distances_parallel(X, Y):
    n = X.shape[0]
    result = np.empty(n)
    for i in prange(n):
        result[i] = compute_distance(X[i], Y[i])
    return result

def main():
    print("Inicjalizacja danych...")
    N = 10000
    D = 1000
    X = np.random.rand(N, D)
    Y = np.random.rand(N, D)
    
    print("Trwa rozgrzewka kompilatora JIT...")
    distances_serial(X[:10], Y[:10])
    distances_parallel(X[:10], Y[:10])
    
    print("\n[1] Uruchamianie wersji sekwencyjnej (@njit + range)...")
    start = time.time()
    res_serial = distances_serial(X, Y)
    time_serial = time.time() - start
    print(f"    Czas wykonania: {time_serial:.4f} s")
    
    print("\n[2] Uruchamianie wersji zrównoleglonej (@njit(parallel=True) + prange)...")
    start = time.time()
    res_parallel = distances_parallel(X, Y)
    time_parallel = time.time() - start
    print(f"    Czas wykonania: {time_parallel:.4f} s")
    
    if time_parallel > 0:
        speedup = time_serial / time_parallel
        print(f"\n🚀 Przyspieszenie: {speedup:.1f}x")
    
    assert np.allclose(res_serial, res_parallel), "Błąd: Wyniki nie są identyczne!"

if __name__ == "__main__":
    main()
