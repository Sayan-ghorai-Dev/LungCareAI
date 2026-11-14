import joblib
import numpy as np

def predict_lung_risk(input_data):
    # Load the trained model and scaler
    model = joblib.load("models/lightgbm.pkl")  # or dynamically selected model
    scaler = joblib.load("models/scaler.pkl")
    
    # Map categorical string inputs to numerical values matching your training encoding
    # (Verify in src/preprocessing.py whether Yes/No was encoded as 1/0 or 2/1)
    mapping = {
        "Yes": 1, "No": 0,
        "YES": 1, "NO": 0,
        "Male": 1, "Female": 0,
        "M": 1, "F": 0
    }
    
    processed_input = [mapping.get(val, val) for val in input_data]
    
    # Convert all values to a 2D float NumPy array to prevent ufunc 'isnan' errors
    numeric_input = np.array([processed_input], dtype=float)
    
    scaled_input = scaler.transform(numeric_input)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]
    
    return prediction[0], probability