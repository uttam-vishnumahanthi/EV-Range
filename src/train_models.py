"""
Train and save all three ML models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

def load_and_prepare_data():
    """Load dataset and prepare features"""
    df = pd.read_csv('data/ev_dataset.csv')
    
    # Feature engineering
    df['temp_delta'] = df['battery_temperature'] - df['ambient_temperature']
    df['speed_squared'] = df['speed'] ** 2
    df['efficiency_score'] = df['vehicle_efficiency'] * df['soh'] / 100
    
    # Define features
    numeric_features = ['soc', 'soh', 'battery_temperature', 'battery_capacity',
                       'vehicle_efficiency', 'ambient_temperature', 'road_elevation',
                       'speed', 'payload', 'temp_delta', 'speed_squared', 'efficiency_score']
    
    categorical_features = ['traffic_conditions', 'hvac_usage', 'driving_style']
    
    X = df[numeric_features + categorical_features]
    y = df['remaining_range']
    
    return X, y, numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

def train_models():
    """Train all three models"""
    print("="*60)
    print("EV RANGE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n📊 Loading data...")
    X, y, num_features, cat_features = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create preprocessor
    preprocessor = create_preprocessor(num_features, cat_features)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    Path('models').mkdir(exist_ok=True)
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store results
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        # Save model
        model_filename = f'models/{name.lower().replace(" ", "_")}.pkl'
        joblib.dump(pipeline, model_filename)
        
        # Print results
        print(f"  Train RMSE: {train_rmse:.2f} km")
        print(f"  Test RMSE:  {test_rmse:.2f} km")
        print(f"  Train R²:   {train_r2:.4f}")
        print(f"  Test R²:    {test_r2:.4f}")
        print(f"  ✅ Saved to {model_filename}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(results).T
    print(summary_df[['test_rmse', 'test_mae', 'test_r2']])
    
    # Save results
    summary_df.to_csv('models/training_results.csv')
    print("\n✅ Results saved to models/training_results.csv")
    
    return results

if __name__ == "__main__":
    train_models()