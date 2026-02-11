import json
import matplotlib.pyplot as plt
import pandas as pd
import os

RESULTS_FILE = "results.jsonl"

def load_results():
    data = []
    if not os.path.exists(RESULTS_FILE):
        print(f"No results found in {RESULTS_FILE}")
        return None
        
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def main():
    df = load_results()
    if df is None: return

    # Tworzymy figurę z 3 podwykresami (3 wiersze, 1 kolumna)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    cases = [
        ("A", "Case A: PyTorch Optimization", axes[0], False),
        ("B", "Case B: Pandas vs Polars", axes[1], False),
        ("C", "Case C: Evolutionary Algorithms", axes[2], True) # Log scale dla Case C
    ]

    for case_id, title, ax, use_log in cases:
        case_df = df[df['case'] == case_id]
        if case_df.empty:
            ax.text(0.5, 0.5, f"No data for Case {case_id}", ha='center')
            continue

        bars = ax.bar(case_df['step'], case_df['time'], color='skyblue')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel("Time (s)")
        if use_log:
            ax.set_yscale('log')
            ax.set_ylabel("Time (s) - Log Scale")
        
        # Etykiety nad słupkami
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("performance_summary.png", dpi=150)
    print("Saved summary plot: performance_summary.png")

if __name__ == "__main__":
    main()
