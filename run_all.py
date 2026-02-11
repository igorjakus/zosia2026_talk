import subprocess
import re
import json
import os

# Konfiguracja skryptów do uruchomienia
BENCHMARKS = [
    # Case A
    {"case": "A", "step": "0_baseline", "path": "case_a_pytorch/0_baseline.py"},
    {"case": "A", "step": "1_fast_loader", "path": "case_a_pytorch/1_fast_loader.py"},
    {"case": "A", "step": "2_inference", "path": "case_a_pytorch/2_inference_mode.py"},
    {"case": "A", "step": "3_tensor_cores", "path": "case_a_pytorch/3_tensor_cores.py"},
    {"case": "A", "step": "4_compile", "path": "case_a_pytorch/4_compile.py"},
    {"case": "A", "step": "5_amp", "path": "case_a_pytorch/5_amp.py"},
    {"case": "A", "step": "6_vram", "path": "case_a_pytorch/6_vram_caching.py"},
    {"case": "A", "step": "7_benchmark", "path": "case_a_pytorch/7_cudnn_benchmark.py"},
    {"case": "A", "step": "8_fused", "path": "case_a_pytorch/8_fused_optimizer.py"},
    
    # Case B
    {"case": "B", "step": "0_pandas", "path": "case_b_data_processing/0_pandas.py"},
    {"case": "B", "step": "1_polars_eager", "path": "case_b_data_processing/1_polars_eager.py"},
    {"case": "B", "step": "2_polars_lazy", "path": "case_b_data_processing/2_polars_lazy.py"},
    
    # Case C
    {"case": "C", "step": "0_pure_python", "path": "case_c_evolution/0_pure_python.py"},
    {"case": "C", "step": "1_numpy", "path": "case_c_evolution/1_numpy_vectorized.py"},
    {"case": "C", "step": "2_argpartition", "path": "case_c_evolution/2_numpy_argpartition.py"},
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
    # 1. Przygotowanie danych dla Case B
    if not os.path.exists("data/yellow_tripdata_2023-01.csv"):
        print("Data missing for Case B. Downloading...")
        subprocess.run(["python", "case_b_data_processing/download_data.py"])

    # 2. Uruchomienie benchmarków
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
