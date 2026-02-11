import polars as pl
import time

CSV_FILE = "data/yellow_tripdata_2023-01.csv"

# --- Polars Lazy Benchmark ---
print("--- Polars (Lazy) ---")
start_time = time.time()

# 1. Definicja Planu (LazyFrame)
print(f"Scanning {CSV_FILE} (Lazy)...")
try:
    q = pl.scan_csv(CSV_FILE, try_parse_dates=True)
except FileNotFoundError:
    print(f"Error: {CSV_FILE} not found. Run download_data.py first.")
    exit(1)

# Budowanie zapytania
# Polars zauważy, że używamy tylko 'tpep_pickup_datetime' i 'tip_amount',
# więc nie wczyta reszty kolumn (Projection Pushdown).
q = (
    q.with_columns(pl.col("tpep_pickup_datetime").dt.hour().alias("hour"))
    .group_by("hour")
    .agg(pl.col("tip_amount").mean())
    .sort("hour")
)

# Explain (opcjonalnie, żeby pokazać plan)
# print(q.explain())

# 2. Wykonanie (collect)
print("Collecting result...")
result = q.collect()

end_time = time.time()
print(f"Total Time (Scan + Process): {end_time - start_time:.2f}s")

print("\nResult (head):")
print(result.head())
