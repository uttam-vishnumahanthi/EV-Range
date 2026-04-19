"""
Prediction utilities for EV range
"""
import joblib
import pandas as pd
import numpy as np

class EVRangePredictor:
    def __init__(self, model_path='models/xgboost.pkl'):
        """Load trained model"""
        self.pipeline = joblib.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    
    def predict(self, ev_data):
        """
        Predict range for a single EV
        
        Args:
            ev_data: dict with keys:
                - soc, soh, battery_temperature, battery_capacity, vehicle_efficiency
                - ambient_temperature, traffic_conditions, road_elevation
                - speed, hvac_usage, payload, driving_style
        
        Returns:
            predicted_range (float)
        """
        # Convert to DataFrame
        df = pd.DataFrame([ev_data])
        
        # Feature engineering (same as training)
        df['temp_delta'] = df['battery_temperature'] - df['ambient_temperature']
        df['speed_squared'] = df['speed'] ** 2
        df['efficiency_score'] = df['vehicle_efficiency'] * df['soh'] / 100
        
        # Select features in correct order
        features = ['soc', 'soh', 'battery_temperature', 'battery_capacity',
                   'vehicle_efficiency', 'ambient_temperature', 'road_elevation',
                   'speed', 'payload', 'temp_delta', 'speed_squared', 'efficiency_score',
                   'traffic_conditions', 'hvac_usage', 'driving_style']
        
        X = df[features]
        
        # Predict
        prediction = self.pipeline.predict(X)[0]
        
        return prediction
    
    def predict_batch(self, ev_data_list):
        """Predict range for multiple EVs"""
        predictions = []
        for ev_data in ev_data_list:
            pred = self.predict(ev_data)
            predictions.append(pred)
        return predictions

# Example usage
if __name__ == "__main__":
    # Test prediction
    predictor = EVRangePredictor('models/xgboost.pkl')
    
    sample_ev = {
        'soc': 75,
        'soh': 95,
        'battery_temperature': 22,
        'battery_capacity': 78,
        'vehicle_efficiency': 6.2,
        'ambient_temperature': 18,
        'traffic_conditions': 'light',
        'road_elevation': 10,
        'speed': 65,
        'hvac_usage': 'low',
        'payload': 180,
        'driving_style': 'moderate'
    }
    
    predicted_range = predictor.predict(sample_ev)
    print(f"\n📊 Sample Prediction:")
    print(f"   SoC: {sample_ev['soc']}%")
    print(f"   Speed: {sample_ev['speed']} km/h")
    print(f"   Driving Style: {sample_ev['driving_style']}")
    print(f"   Predicted Range: {predicted_range:.1f} km")