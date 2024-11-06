# src/salespredictor/components/model_inference.py

import pandas as pd
import joblib
from pathlib import Path


def load_model(model_path: str = "models/best_model.joblib"):
    """
    Load the trained model from a file.
    
    Args:
    - model_path (str): Path to the saved model file.
    
    Returns:
    - model: The loaded model.
    """
    model = joblib.load(model_path)
    return model

def load_inference_data(file_path: str = "processed_data/inference_data.csv"):
    """
    Load the preprocessed inference data from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file containing inference data.
    
    Returns:
    - inference_data (pd.DataFrame): Data prepared for making predictions.
    """
    inference_data = pd.read_csv(file_path)
    return inference_data

def make_predictions(model, inference_data: pd.DataFrame):
    """
    Make predictions using the loaded model on the inference data.
    
    Args:
    - model: The trained model.
    - inference_data (pd.DataFrame): The data for making predictions.
    
    Returns:
    - predictions (pd.Series): The predicted values.
    """
    predictions = model.predict(inference_data)
    return predictions

def save_predictions(predictions, output_path: str = "processed_data/predictions.csv"):
    """
    Save the predictions to a CSV file.
    
    Args:
    - predictions (pd.Series): The predicted values.
    - output_path (str): Path to save the predictions CSV file.
    """
    # Ensure the save directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions, columns=["predicted_discounted_price"])
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def main():
    # Load the model
    model = load_model()

    # Load inference data
    inference_data = load_inference_data()

    # Make predictions
    predictions = make_predictions(model, inference_data)

    # Save predictions
    save_predictions(predictions)

    # Print sample predictions
    print("Sample Predictions:")
    print(predictions[:5])

if __name__ == "__main__":
    main()
