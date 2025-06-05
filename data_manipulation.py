import pandas as pd
import os
import time
from glob import glob

def load_latest_csv_and_display():
    # Pattern for your CSV files
    path_pattern = '/Users/kushagrasachdeva/Desktop/Kusha/nifty100_symbols_CSV_Download_20250604_145848.csv'

    # Find all matching files
    csv_files = glob(path_pattern)
    if not csv_files:
        print("No CSV files found matching the pattern.")
        return

    # Get the latest file by modification time
    latest_file = max(csv_files, key=os.path.getmtime)

    try:
        df = pd.read_csv(latest_file)
        print(f"\nSuccessfully loaded: {latest_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData Info:")
        print(df.info())
        print("\nDataFrame is ready for data manipulation!")
    except Exception as e:
        print(f"Error reading {latest_file}: {e}")

if __name__ == "__main__":
    print("Starting CSV loader. Updates every 10 minutes.")
    while True:
        load_latest_csv_and_display()
        print("\nWaiting 10 minutes for next update...\n")
        time.sleep(10 * 60)  # Sleep for 10 minutes