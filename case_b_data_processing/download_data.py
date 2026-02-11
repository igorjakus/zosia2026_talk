import os
import requests
import pandas as pd
import sys

# URL do danych (NYC Yellow Taxi - Styczeń 2023)
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
PARQUET_FILE = "data/yellow_tripdata_2023-01.parquet"
CSV_FILE = "data/yellow_tripdata_2023-01.csv"


def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)


def convert_parquet_to_csv(parquet_path, csv_path):
    if os.path.exists(csv_path):
        print(f"{csv_path} already exists. Skipping conversion.")
        return

    print(f"Converting {parquet_path} to {csv_path} (for Pandas CSV benchmark)...")
    try:
        # Używamy pandas do konwersji, bo i tak będzie potrzebny
        df = pd.read_parquet(parquet_path)
        # Zapisujemy jako CSV bez indeksu
        df.to_csv(csv_path, index=False)
        print(f"Conversion complete. CSV size: {os.path.getsize(csv_path) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error converting to CSV: {e}")
        # Nie przerywamy, bo może użytkownik chce testować na Parquet


if __name__ == "__main__":
    download_file(DATA_URL, PARQUET_FILE)
    convert_parquet_to_csv(PARQUET_FILE, CSV_FILE)
    print("\nData preparation ready.")
