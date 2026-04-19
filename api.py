"""
Simple REST API for EV range prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="EV Range Predictor API", version="1.0.0")

# Load model
try:
    pipeline = joblib.load('models/xgboost.pkl')
    print("✅ Model loaded successfully")
except:
    print("⚠️ Model not found. Run training first.")
    pipeline = None

# Request model
class EVRequest(BaseModel):
    soc: float
    soh: float = 95.0
    battery_temperature: float = 22.0
    battery_capacity: float = 75.0
    vehicle_efficiency: float = 6.2
    ambient_temperature: float = 18.0
    traffic_conditions: str = "light"
    road_elevation: float = 0.0
    speed: float = 60.0
    hvac_usage: str = "off"
    payload: float = 150.0
    driving_style: str = "moderate"

# Response model
class EVResponse(BaseModel):
    predicted_range: float
    status: str

@app.get("/")
def root():
    return {"message": "EV Range Predictor API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=EVResponse)
def predict(ev: EVRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert to DataFrame
    data = pd.DataFrame([ev.dict()])
    
    # Feature engineering
    data['temp_delta'] = data['battery_temperature'] - data['ambient_temperature']
    data['speed_squared'] = data['speed'] ** 2
    data['efficiency_score'] = data['vehicle_efficiency'] * data['soh'] / 100
    
    # Select features
    features = ['soc', 'soh', 'battery_temperature', 'battery_capacity',
               'vehicle_efficiency', 'ambient_temperature', 'road_elevation',
               'speed', 'payload', 'temp_delta', 'speed_squared', 'efficiency_score',
               'traffic_conditions', 'hvac_usage', 'driving_style']
    
    X = data[features]
    
    # Predict
    prediction = pipeline.predict(X)[0]
    
    return EVResponse(predicted_range=prediction, status="success")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)