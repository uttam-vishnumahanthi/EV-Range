"""
Generate synthetic EV dataset for range prediction
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_ev_dataset(n_samples=10000, random_seed=42):
    """Generate realistic EV driving data"""
    np.random.seed(random_seed)
    
    print(f"Generating {n_samples} EV samples...")
    
    # Internal factors
    soc = np.random.uniform(10, 100, n_samples)
    soh = np.random.uniform(70, 100, n_samples)
    battery_temp = np.random.normal(25, 10, n_samples)
    battery_capacity = np.random.normal(75, 5, n_samples)
    vehicle_efficiency = np.random.normal(6, 0.5, n_samples)
    
    # Environmental factors
    ambient_temp = np.random.normal(20, 15, n_samples)
    traffic = np.random.choice(['light', 'moderate', 'heavy'], n_samples, p=[0.4, 0.4, 0.2])
    elevation = np.random.normal(0, 50, n_samples)
    
    # Usage factors
    speed = np.random.normal(60, 20, n_samples)
    hvac = np.random.choice(['off', 'low', 'medium', 'high'], n_samples)
    payload = np.random.normal(150, 50, n_samples)
    driving_style = np.random.choice(['eco', 'moderate', 'aggressive'], n_samples, p=[0.3, 0.5, 0.2])
    
    # Calculate range
    base_range = (soc/100) * (soh/100) * battery_capacity * vehicle_efficiency
    
    # Adjustments
    temp_penalty = np.where(np.abs(ambient_temp - 25) > 15, 0.8, 1.0)
    traffic_penalty = {'light': 1.0, 'moderate': 0.85, 'heavy': 0.7}
    traffic_factor = np.array([traffic_penalty[t] for t in traffic])
    
    elevation_penalty = 1 / (1 + np.abs(elevation) / 200)
    speed_efficiency = 1 - np.abs(speed - 50) / 150
    speed_efficiency = np.clip(speed_efficiency, 0.6, 1.0)
    
    hvac_penalty = {'off': 1.0, 'low': 0.95, 'medium': 0.88, 'high': 0.8}
    hvac_factor = np.array([hvac_penalty[h] for h in hvac])
    
    payload_penalty = 1 - (payload - 100) / 1000
    payload_penalty = np.clip(payload_penalty, 0.7, 1.0)
    
    style_factor = {'eco': 1.1, 'moderate': 1.0, 'aggressive': 0.85}
    driving_factor = np.array([style_factor[s] for s in driving_style])
    
    # Final range
    remaining_range = (base_range * temp_penalty * traffic_factor * elevation_penalty * 
                      speed_efficiency * hvac_factor * payload_penalty * driving_factor)
    remaining_range *= np.random.normal(1, 0.05, n_samples)
    remaining_range = np.maximum(remaining_range, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'soc': soc,
        'soh': soh,
        'battery_temperature': battery_temp,
        'battery_capacity': battery_capacity,
        'vehicle_efficiency': vehicle_efficiency,
        'ambient_temperature': ambient_temp,
        'traffic_conditions': traffic,
        'road_elevation': elevation,
        'speed': speed,
        'hvac_usage': hvac,
        'payload': payload,
        'driving_style': driving_style,
        'remaining_range': remaining_range
    })
    
    # Save to CSV
    Path('data').mkdir(exist_ok=True)
    df.to_csv('data/ev_dataset.csv', index=False)
    
    print(f"✅ Dataset saved to data/ev_dataset.csv")
    print(f"   Shape: {df.shape}")
    print(f"   Range: {df['remaining_range'].min():.1f} - {df['remaining_range'].max():.1f} km")
    print(f"   Average: {df['remaining_range'].mean():.1f} km")
    
    return df

if __name__ == "__main__":
    df = generate_ev_dataset(10000)
    print("\nFirst 5 rows:")
    print(df.head())