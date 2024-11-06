# src/salespredictor/components/data_preprocessing.py

import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from src.salespredictor.components.data_ingestion import load_data  # Import load_data from data_ingestion

def clean_and_preprocess_data(df: pd.DataFrame, save_dir: str = "processed_data"):
    """
    Clean and preprocess the dataset, saving the processed train and inference data to CSV files.
    
    Args:
    - df (pd.DataFrame): The raw input data.
    - save_dir (str): Directory to save the processed files.
    
    Returns:
    - train_data (pd.DataFrame): Data prepared for training.
    - inference_data (pd.DataFrame): Data prepared for inference.
    """
    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Clean 'actual_price' and 'discounted_price' columns
    df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
    df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)

    # Handle missing values in 'rating' and 'rating_count'
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    rating_imputer = SimpleImputer(strategy='mean')
    df['rating'] = rating_imputer.fit_transform(df[['rating']])

    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['rating_count'] = rating_imputer.fit_transform(df[['rating_count']])

    # Encode 'category' column
    label_encoder = LabelEncoder()
    df['category'] = label_encoder.fit_transform(df['category'])

    # Define the features you want to use for training
    features = ['actual_price', 'rating', 'rating_count', 'category']

    # Prepare train and inference data
    train_data = df[features + ['discounted_price']]  # Include target column for training
    inference_data = df[features]  # Only features for inference

    # Save processed data to CSV files
    train_data.to_csv(Path(save_dir) / "train_data.csv", index=False)
    inference_data.to_csv(Path(save_dir) / "inference_data.csv", index=False)

    return train_data, inference_data

# Usage example
if __name__ == "__main__":
    df = load_data()  # Load raw data
    train_data, inference_data = clean_and_preprocess_data(df)  # Preprocess data
    print("Train Data:", train_data.head())
    print("Inference Data:", inference_data.head())
