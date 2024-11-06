# src/salespredictor/components/model_training.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_train_data(file_path: str = "processed_data/train_data.csv"):
    """
    Load the preprocessed training data from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file containing training data.
    
    Returns:
    - X (pd.DataFrame): Features for training.
    - y (pd.Series): Target variable.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=['discounted_price'])
    y = df['discounted_price']
    return X, y

def train_models(X_train, y_train):
    """
    Train multiple regression models and store them in a dictionary.
    
    Args:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target variable.
    
    Returns:
    - trained_models (dict): Dictionary of trained models.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor()
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate models on the test set and return their performance metrics.
    
    Args:
    - trained_models (dict): Dictionary of trained models.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target variable.
    
    Returns:
    - model_performance (dict): Performance metrics for each model.
    """
    model_performance = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_performance[name] = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
    
    return model_performance

def save_best_model(trained_models, model_performance, save_path: str = "models/best_model.joblib"):
    """
    Save the best model based on R2 score using joblib.
    
    Args:
    - trained_models (dict): Dictionary of trained models.
    - model_performance (dict): Performance metrics for each model.
    - save_path (str): Path to save the best model.
    """
    # Identify the best model based on R2 score
    best_model_name = max(model_performance, key=lambda x: model_performance[x]['R2'])
    best_model = trained_models[best_model_name]
    
    # Ensure save directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the best model
    joblib.dump(best_model, save_path)
    print(f"Best model ({best_model_name}) saved to {save_path}.")

def main():
    # Load training data
    X, y = load_train_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate models
    model_performance = evaluate_models(trained_models, X_test, y_test)

    # Display model performance
    print("Model Performance on Test Data:")
    for model, metrics in model_performance.items():
        print(f"{model}: MAE={metrics['MAE']}, MSE={metrics['MSE']}, R2={metrics['R2']}")
    
    # Save the best model
    save_best_model(trained_models, model_performance)

if __name__ == "__main__":
    main()
