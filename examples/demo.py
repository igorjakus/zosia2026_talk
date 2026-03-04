"""
Demo: Numba @njit vs @njit(parallel=True) z prange.
Oblicza odległości euklidesowe między parami wektorów.
Pokazuje: Python → Numba (1 rdzeń) → Numba (parallel) → NumPy
"""
import numpy as np
import time
from numba import njit, prange


def euclidean_python(X, Y):
    """Czysta pętla Pythona."""
    n = X.shape[0]
    result = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(X.shape[1]):
            s += (X[i, j] - Y[i, j]) ** 2
        result[i] = s ** 0.5
    return result


@njit
def euclidean_njit(X, Y):
    """Numba — 1 rdzeń."""
    n = X.shape[0]
    result = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(X.shape[1]):
            s += (X[i, j] - Y[i, j]) ** 2
        result[i] = s ** 0.5
    return result


@njit(parallel=True)
def euclidean_parallel(X, Y):
    """Numba — wiele rdzeni (prange)."""
    n = X.shape[0]
    result = np.empty(n)
    for i in prange(n):  # <-- jedyna zmiana!
        s = 0.0
        for j in range(X.shape[1]):
            s += (X[i, j] - Y[i, j]) ** 2
        result[i] = s ** 0.5
    return result


def euclidean_numpy(X, Y):
    """NumPy — wektoryzacja."""
    return np.sqrt(np.sum((X - Y) ** 2, axis=1))


@njit(parallel=True)
def euclidean_njit_numpy(X, Y):
    """Numba + NumPy — auto-fuzja operacji NumPy!"""
    return np.sqrt(np.sum((X - Y) ** 2, axis=1))


if __name__ == "__main__":
    N, D = 500_000, 128
    X = np.random.randn(N, D)
    Y = np.random.randn(N, D)

    # Warmup
    print("Kompilacja Numba (warmup)...")
    euclidean_njit(X[:10], Y[:10])
    euclidean_parallel(X[:10], Y[:10])
    euclidean_njit_numpy(X[:10], Y[:10])

    # --- Pure Python ---
    n_py = 5_000
    print(f"\n1. Pure Python (N={n_py})...")
    t0 = time.perf_counter()
    euclidean_python(X[:n_py], Y[:n_py])
    t_py = time.perf_counter() - t0
    t_py_scaled = t_py / n_py * N
    print(f"   Czas: {t_py:.3f}s  (ekstrapolacja na N={N}: ~{t_py_scaled:.1f}s)")

    # --- Numba single-threaded ---
    print(f"\n2. Numba @njit (N={N})...")
    t0 = time.perf_counter()
    euclidean_njit(X, Y)
    t_nb = time.perf_counter() - t0
    print(f"   Czas: {t_nb:.3f}s")

    # --- Numba parallel ---
    print(f"\n3. Numba @njit + prange (N={N})...")
    t0 = time.perf_counter()
    euclidean_parallel(X, Y)
    t_par = time.perf_counter() - t0
    print(f"   Czas: {t_par:.3f}s")

    # --- NumPy ---
    print(f"\n4. NumPy (N={N})...")
    t0 = time.perf_counter()
    euclidean_numpy(X, Y)
    t_np = time.perf_counter() - t0
    print(f"   Czas: {t_np:.3f}s")

    # --- Numba + NumPy style ---
    print(f"\n5. Numba @njit + NumPy syntax (N={N})...")
    t0 = time.perf_counter()
    euclidean_njit_numpy(X, Y)
    t_nbnp = time.perf_counter() - t0
    print(f"   Czas: {t_nbnp:.3f}s")

    # Podsumowanie
    print(f"\n{'='*50}")
    print(f"Python (ekstr.):    {t_py_scaled:.1f}s  (baseline)")
    print(f"Numba 1-rdzeń:      {t_nb:.3f}s  ({t_py_scaled/t_nb:.0f}× szybciej)")
    print(f"Numba parallel:     {t_par:.3f}s  ({t_py_scaled/t_par:.0f}× szybciej)")
    print(f"NumPy:              {t_np:.3f}s  ({t_py_scaled/t_np:.0f}× szybciej)")
    print(f"Numba + NumPy:      {t_nbnp:.3f}s  ({t_py_scaled/t_nbnp:.0f}× szybciej)")
