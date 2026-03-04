import subprocess
import re
import json
import os

# Konfiguracja skryptów do uruchomienia
BENCHMARKS = [
    # PyTorch
    {"case": "PyTorch", "step": "0_baseline", "path": "examples/pytorch_optimization/0_baseline.py"},
    {"case": "PyTorch", "step": "1_fast_loader", "path": "examples/pytorch_optimization/1_fast_loader.py"},
    {"case": "PyTorch", "step": "2_inference", "path": "examples/pytorch_optimization/2_inference_mode.py"},
    {"case": "PyTorch", "step": "3_tensor_cores", "path": "examples/pytorch_optimization/3_tensor_cores.py"},
    {"case": "PyTorch", "step": "4_compile", "path": "examples/pytorch_optimization/4_compile.py"},
    {"case": "PyTorch", "step": "5_amp", "path": "examples/pytorch_optimization/5_amp.py"},
    {"case": "PyTorch", "step": "6_vram", "path": "examples/pytorch_optimization/6_vram_caching.py"},
    {"case": "PyTorch", "step": "7_benchmark", "path": "examples/pytorch_optimization/7_cudnn_benchmark.py"},
    {"case": "PyTorch", "step": "8_fused", "path": "examples/pytorch_optimization/8_fused_optimizer.py"},
]

RESULTS_FILE = "results.jsonl"

def run_benchmark(config):
    print(f">>> Running {config['path']}...")
    try:
        # Uruchamiamy skrypt i przechwytujemy wyjście
        process = subprocess.Popen(
            ["python", config['path']], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        last_time = None
        for line in process.stdout:
            print(line, end="") # Wyświetlamy na żywo
            
            # Szukamy wzorca czasu (np. "Total Execution Time: 12.34s")
            match = re.search(r"(?:Total|Execution)\s+Time.*:\s+([\d.]+)", line)
            if match:
                last_time = float(match.group(1))
        
        process.wait()
        
        if last_time is not None:
            return {
                "case": config['case'],
                "step": config['step'],
                "time": last_time
            }
    except Exception as e:
        print(f"Error running {config['path']}: {e}")
    return None

def main():
    # Uruchomienie benchmarków
    results = []
    for bench in BENCHMARKS:
        res = run_benchmark(bench)
        if res:
            results.append(res)
            # Zapisujemy na bieżąco
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(res) + "\n")

    print(f"\nBenchmarks finished. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
