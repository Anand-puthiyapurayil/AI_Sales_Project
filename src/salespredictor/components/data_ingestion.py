# src/salespredictor/components/data_ingestion.py

import pandas as pd
from pathlib import Path

def load_data(file_path: str = "research/Data/amazon.csv"):
    """
    Loads the raw data from a specified CSV file.

    Args:
    - file_path (str): Path to the CSV file containing raw data.

    Returns:
    - df (pd.DataFrame): Loaded data as a DataFrame.
    """
    # Build the absolute path to the data file from the project root
    project_root = Path(__file__).resolve().parents[3]  # Adjust based on structure
    full_path = project_root / file_path

    # Now, use the full_path to load the CSV
    df = pd.read_csv(full_path)
    return df

# Usage example
if __name__ == "__main__":
    data = load_data()
    print("Sample data:", data.head())
