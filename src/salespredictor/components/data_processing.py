import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving and loading the encoder
from src.salespredictor.components.data_ingestion import load_data  # Import load_data from data_ingestion

def clean_and_preprocess_data(df: pd.DataFrame, save_dir: str = "processed_data", model_dir: str = "models", is_training=True):
    """
    Clean and preprocess the dataset, saving the processed train and inference data to CSV files.
    
    Args:
    - df (pd.DataFrame): The raw input data.
    - save_dir (str): Directory to save the processed files.
    - model_dir (str): Directory to save/load the encoder.
    - is_training (bool): Whether to include the target column (for training data).
    
    Returns:
    - train_data (pd.DataFrame): Data prepared for training (if is_training is True).
    - inference_data (pd.DataFrame): Data prepared for inference (if is_training is False).
    """
    # Ensure save and model directories exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Clean 'actual_price' and 'discounted_price' columns
    df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
    if 'discounted_price' in df.columns:
        df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)

    # Handle missing values in 'rating' and 'rating_count'
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    rating_imputer = SimpleImputer(strategy='mean')
    df['rating'] = rating_imputer.fit_transform(df[['rating']])

    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['rating_count'] = rating_imputer.fit_transform(df[['rating_count']])

    # Encode 'category' column
    encoder_path = Path(model_dir) / 'label_encoder.joblib'
    if is_training:
        # Fit and save the encoder for training data
        label_encoder = LabelEncoder()
        df['category'] = label_encoder.fit_transform(df['category'])
        joblib.dump(label_encoder, encoder_path)
        print("Label encoder fitted and saved for training.")
    else:
        # Load the encoder for inference data
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
            print("Label encoder loaded for inference.")

            # Transform categories, handling unknowns by setting them to a default value
            known_categories = set(label_encoder.classes_)
            df['category'] = df['category'].apply(lambda x: x if x in known_categories else 'unknown')
            label_encoder.classes_ = list(known_categories | {'unknown'})
            df['category'] = label_encoder.transform(df['category'])
        else:
            raise FileNotFoundError(f"Encoder not found at {encoder_path}. Please run in training mode first.")

    # Define features for both training and inference
    features = ['actual_price', 'rating', 'rating_count', 'category']

    # Processed data for training or inference
    if is_training:
        train_data = df[features + ['discounted_price']]
        train_data.to_csv(Path(save_dir) / "train_data.csv", index=False)
        print(f"Training data saved to {Path(save_dir) / 'train_data.csv'}")
        return train_data
    else:
        inference_data = df[features]
        inference_data.to_csv(Path(save_dir) / "inference_data.csv", index=False)
        print(f"Inference data saved to {Path(save_dir) / 'inference_data.csv'}")
        return inference_data

# Usage example
if __name__ == "__main__":
    df = load_data()  # Load raw data
    train_data = clean_and_preprocess_data(df, is_training=True)  # Preprocess for training
    inference_data = clean_and_preprocess_data(df, is_training=False)  # Preprocess for inference
    print("Train Data:", train_data.head())
    print("Inference Data:", inference_data.head())
