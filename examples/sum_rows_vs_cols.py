import torch
import timeit


def benchmark_torch(rows=1000, cols=1000, runs=10, device="cpu"):
    device = torch.device(device)
    tensor = torch.randn(rows, cols, device=device)
    print(f"Benchmarking sum over array of shape: {rows}x{cols}")
    print(f"Device: {device}")

    t_rows = timeit.timeit(lambda: torch.sum(tensor, dim=1), number=runs)
    print(f"Sum Rows: {t_rows / runs:.6f} seconds per run")

    t_cols = timeit.timeit(lambda: torch.sum(tensor, dim=0), number=runs)
    print(f"Sum Columns: {t_cols / runs:.6f} seconds per run")

    print(f"Rows faster by: {t_cols / t_rows:.2f}x\n")


if __name__ == "__main__":
    runs = 100
    configs = [(50_000, 50_000), (10_000_000, 100), (100, 10_000_000)]
    for rows, cols in configs:
        benchmark_torch(rows, cols, runs=runs, device="cuda")

    configs


"""Results on RTX 5070ti:
Benchmarking sum over array of shape: 50000x50000
Device: cuda
Sum Rows: 0.000246 seconds per run
Sum Columns: 0.016512 seconds per run
Rows faster by: 67.00x

Benchmarking sum over array of shape: 10000000x100
Device: cuda
Sum Rows: 0.001533 seconds per run
Sum Columns: 0.002999 seconds per run
Rows faster by: 1.96x

Benchmarking sum over array of shape: 100x10000000
Device: cuda
Sum Rows: 0.003040 seconds per run
Sum Columns: 0.004951 seconds per run
Rows faster by: 1.63x
"""
