# Zosia 2026 Talk: High-Performance Python

## Quickstart

Install uv from https://github.com/astral-sh/uv

```bash
# 1. Install dependencies
uv sync

# 2. Run the main demo from the talk
uv run examples/demo.py

# 3. Run benchmarks
uv run run_all.py
```

## Structure
- `examples/pytorch_optimization/`: PyTorch training optimizations and JIT (`torch.compile`)
- `examples/`: Additional demos used during the presentation
