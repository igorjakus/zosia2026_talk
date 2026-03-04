import torch
import timeit


@torch.compile
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def gelu_slow(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


x = torch.randn(1_000_000)
# Warmup
gelu(x)
gelu_slow(x)

runs = 10
t_gelu = timeit.timeit(lambda: gelu(x), number=runs)
t_gelu_slow = timeit.timeit(lambda: gelu_slow(x), number=runs)
print(f"GELU (compiled): {t_gelu / runs:.6f} seconds per run")
print(f"GELU (slow): {t_gelu_slow / runs:.6f} seconds per run")
print(f"Compiled GELU faster by: {t_gelu_slow / t_gelu:.2f}x\n")
