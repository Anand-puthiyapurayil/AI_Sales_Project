from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.salespredictor.components.data_processing import clean_and_preprocess_data
from src.salespredictor.components.model_training import load_train_data, train_models, evaluate_models, save_best_model

app = FastAPI()

# Define model and data paths
MODEL_PATH = "models/best_model.joblib"
DATA_DIR = "data"
PROCESSED_DIR = "processed_data"

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load the trained model once when the app starts
def load_model(model_path: str = MODEL_PATH):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model at startup
model = load_model()

class PredictionInput(BaseModel):
    actual_price: float
    rating: float
    rating_count: float
    category: int  # Assuming 'category' is encoded as an integer

@app.get("/health")
async def health_check():
    """
    Health check endpoint to ensure the model and API are functioning.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return {"status": "API is running and model is loaded"}

@app.post("/ingest-data")
async def ingest_data(file: UploadFile = File(...), is_training: bool = True):
    """
    Endpoint to ingest data, save it, and preprocess it.
    """
    try:
        file_location = os.path.join(DATA_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        df = pd.read_csv(file_location)
        processed_data = clean_and_preprocess_data(df, save_dir=PROCESSED_DIR, is_training=is_training)
        output_file = "train_data.csv" if is_training else "inference_data.csv"
        processed_file_path = os.path.join(PROCESSED_DIR, output_file)
        
        processed_data.to_csv(processed_file_path, index=False)
        
        return {
            "message": "Data ingested and preprocessed successfully",
            "file_location": file_location,
            "processed_file_location": processed_file_path,
            "data_preview": processed_data.head().to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data ingestion and preprocessing failed: {str(e)}")

@app.post("/train-model")
async def train_model_endpoint():
    """
    Endpoint to train the model on preprocessed data and save it.
    """
    try:
        X, y = load_train_data(file_path=f"{PROCESSED_DIR}/train_data.csv")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        trained_models = train_models(X_train, y_train)
        model_performance = evaluate_models(trained_models, X_test, y_test)
        save_best_model(trained_models, model_performance, save_path=MODEL_PATH)

        global model
        model = load_model()

        return {"message": "Model trained, evaluated, and saved successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    API endpoint to make predictions based on input data.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please check the model path or load the model.")

    try:
        data = pd.DataFrame([input_data.dict()])
        inference_data = clean_and_preprocess_data(data, is_training=False)

        if inference_data.empty or inference_data.isna().any().any():
            raise ValueError("Preprocessing returned empty or invalid data for inference.")

        prediction = model.predict(inference_data)
        return {"predictions": prediction.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/evaluate-model")
async def evaluate_model():
    """
    Endpoint to evaluate the model and return performance metrics and actual vs. predicted values.
    """
    try:
        X, y = load_train_data(file_path=f"{PROCESSED_DIR}/train_data.csv")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        response = {
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "predictions": y_pred.tolist(),
            "actuals": y_test.tolist()
        }
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")
