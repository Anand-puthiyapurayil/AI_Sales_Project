import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the FastAPI server URL
API_URL = "http://127.0.0.1:8000"

# Set up the main title and sidebar for navigation
st.title("ML Model Pipeline Interface")
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose a service", ["Ingest Data", "Train Model", "Make Predictions", "Evaluate Model", "Check API Health"])

# Ingest Data Section
if options == "Ingest Data":
    st.header("Ingest and Preprocess Data")
    
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        is_training = st.radio("Is this data for training?", ("Yes", "No")) == "Yes"
        
        # Display preview of uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())
        
        # Send file to FastAPI endpoint for ingestion and preprocessing
        if st.button("Ingest and Preprocess Data"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/ingest-data", files=files, params={"is_training": is_training})
            
            # Display response message
            if response.status_code == 200:
                st.success("Data ingested and preprocessed successfully.")
                st.json(response.json())
            else:
                st.error("Data ingestion failed.")
                st.json(response.json())

# Train Model Section
elif options == "Train Model":
    st.header("Train Model")
    
    # Trigger model training on FastAPI server
    if st.button("Train the Model"):
        response = requests.post(f"{API_URL}/train-model")
        
        # Display training response
        if response.status_code == 200:
            st.success("Model trained, evaluated, and saved successfully.")
            st.json(response.json())
        else:
            st.error("Model training failed.")
            st.json(response.json())

# Make Predictions Section
elif options == "Make Predictions":
    st.header("Make Predictions")
    
    # Input fields for model features
    actual_price = st.number_input("Actual Price", min_value=0.0)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    rating_count = st.number_input("Rating Count", min_value=0)
    category = st.number_input("Category (as an integer)", min_value=0)
    
    # Prepare input data for prediction
    input_data = {
        "actual_price": actual_price,
        "rating": rating,
        "rating_count": rating_count,
        "category": category
    }
    
    # Send data to FastAPI for prediction
    if st.button("Predict Discounted Price"):
        response = requests.post(f"{API_URL}/predict", json=input_data)
        
        # Display prediction result
        if response.status_code == 200:
            prediction = response.json()["predictions"][0]
            st.success(f"Predicted Discounted Price: {prediction}")
        else:
            st.error("Prediction failed.")
            st.json(response.json())

# Evaluate Model Section
elif options == "Evaluate Model":
    st.header("Evaluate Model")
    
    # Trigger model evaluation on FastAPI server
    if st.button("Evaluate the Model"):
        response = requests.post(f"{API_URL}/evaluate-model")
        
        # Display evaluation metrics and plot
        if response.status_code == 200:
            metrics = response.json()
            st.write("Model Performance Metrics:")
            st.write(f"MAE: {metrics['MAE']}")
            st.write(f"MSE: {metrics['MSE']}")
            st.write(f"R2 Score: {metrics['R2']}")
            
            # Plot actual vs. predicted values
            actuals = metrics["actuals"]
            predictions = metrics["predictions"]
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=actuals, y=predictions, alpha=0.6)
            plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')  # Line for perfect predictions
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Actual vs. Predicted Values")
            st.pyplot(plt)

        else:
            st.error("Model evaluation failed.")
            st.json(response.json())

# Check API Health Section
elif options == "Check API Health":
    st.header("Check API Health")
    
    # Send request to health check endpoint
    if st.button("Check API Health Status"):
        response = requests.get(f"{API_URL}/health")
        
        # Display health check status
        if response.status_code == 200:
            st.success("API is running and model is loaded.")
            st.json(response.json())
        else:
            st.error("API health check failed.")
            st.json(response.json())
