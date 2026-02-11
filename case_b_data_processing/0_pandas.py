import pandas as pd
import time
import os

CSV_FILE = "data/yellow_tripdata_2023-01.csv"

# --- Pandas Benchmark ---
print("--- Pandas (Eager) ---")
start_time = time.time()

# 1. Wczytanie CSV
print(f"Reading {CSV_FILE}...")
try:
    # Pandas domyślnie parsuje wszystko jako object/int/float. Daty trzeba wymusić albo sparsować osobno.
    # Używamy low_memory=False, aby uniknąć ostrzeżeń o typach w dużych plikach.
    df = pd.read_csv(CSV_FILE, low_memory=False)
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found. Run download_data.py first.")
    exit(1)

read_time = time.time()
print(f"Read Time: {read_time - start_time:.2f}s")

# 2. Przetwarzanie
# Konwersja kolumny z datą (bardzo kosztowna w Pandas)
print("Parsing dates...")
df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

# Wyciągnięcie godziny
print("Extracting hour...")
df['hour'] = df['pickup_datetime'].dt.hour

# Grupowanie i agregacja
print("Grouping & Aggregating...")
result = df.groupby('hour')['tip_amount'].mean()

# Sortowanie
result = result.sort_index()

processing_time = time.time()
print(f"Processing Time: {processing_time - read_time:.2f}s")
print(f"Total Time: {processing_time - start_time:.2f}s")

print("\nResult (head):")
print(result.head())
