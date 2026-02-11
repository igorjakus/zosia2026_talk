import polars as pl
import time
import os

CSV_FILE = "data/yellow_tripdata_2023-01.csv"

# --- Polars Eager Benchmark ---
print("--- Polars (Eager) ---")
start_time = time.time()

# 1. Wczytanie CSV
print(f"Reading {CSV_FILE}...")
try:
    # Polars automatycznie próbuje wykryć daty, ale dla uczciwości zrobimy to explicite lub pozwolimy mu działać
    # try_parse_dates=True w read_csv jest bardzo szybkie
    df = pl.read_csv(CSV_FILE, try_parse_dates=True)
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found. Run download_data.py first.")
    exit(1)

read_time = time.time()
print(f"Read Time: {read_time - start_time:.2f}s")

# 2. Przetwarzanie
# Jeśli data nie została wykryta (zależy od formatu w CSV), musimy parsować.
# Sprawdźmy typ. Jeśli String, parsujemy.
if df["tpep_pickup_datetime"].dtype == pl.String:
    print("Parsing dates (str -> datetime)...")
    # strptime jest bardzo szybkie w Polars
    df = df.with_columns(
        pl.col("tpep_pickup_datetime").str.strptime(pl.Datetime)
    )

print("Grouping & Aggregating...")
result = (
    df
    .with_columns(pl.col("tpep_pickup_datetime").dt.hour().alias("hour"))
    .group_by("hour")
    .agg(pl.col("tip_amount").mean())
    .sort("hour")
)

processing_time = time.time()
print(f"Processing Time: {processing_time - read_time:.2f}s")
print(f"Total Time: {processing_time - start_time:.2f}s")

print("\nResult (head):")
print(result.head())
