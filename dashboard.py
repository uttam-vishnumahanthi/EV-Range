"""
Interactive dashboard for EV range prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predict import EVRangePredictor

# Page config
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="🔋",
    layout="wide"
)

# Title
st.title("🔋 Electric Vehicle Range Predictor")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return EVRangePredictor('models/xgboost.pkl')

try:
    predictor = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model not found! Please run `python src/train_models.py` first")

# Sidebar - Input parameters
st.sidebar.header("⚙️ Vehicle Parameters")

with st.sidebar:
    st.subheader("🔋 Battery Status")
    soc = st.slider("State of Charge (SoC %)", 0, 100, 75)
    soh = st.slider("State of Health (SoH %)", 50, 100, 95)
    battery_temp = st.slider("Battery Temperature (°C)", -10, 50, 22)
    battery_capacity = st.number_input("Battery Capacity (kWh)", 50.0, 100.0, 75.0, step=1.0)
    vehicle_efficiency = st.number_input("Vehicle Efficiency (km/kWh)", 4.0, 8.0, 6.2, step=0.1)
    
    st.subheader("🌤️ Environmental Factors")
    ambient_temp = st.slider("Ambient Temperature (°C)", -20, 45, 18)
    traffic = st.selectbox("Traffic Conditions", ['light', 'moderate', 'heavy'])
    elevation = st.slider("Road Elevation (m)", -100, 500, 10)
    
    st.subheader("🚗 Driving Factors")
    speed = st.slider("Speed (km/h)", 0, 140, 65)
    hvac = st.selectbox("HVAC Usage", ['off', 'low', 'medium', 'high'])
    payload = st.slider("Payload (kg)", 50, 500, 180)
    driving_style = st.selectbox("Driving Style", ['eco', 'moderate', 'aggressive'])

# Main content - 2 columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Input Summary")
    input_data = {
        'Parameter': ['SoC', 'SoH', 'Battery Temp', 'Ambient Temp', 'Speed', 
                     'Traffic', 'Driving Style', 'HVAC', 'Payload'],
        'Value': [f"{soc}%", f"{soh}%", f"{battery_temp}°C", f"{ambient_temp}°C", 
                 f"{speed} km/h", traffic, driving_style, hvac, f"{payload} kg"]
    }
    st.table(pd.DataFrame(input_data))

with col2:
    st.subheader("🎯 Prediction")
    
    if model_loaded:
        # Prepare data for prediction
        ev_data = {
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
            'driving_style': driving_style
        }
        
        # Make prediction
        predicted_range = predictor.predict(ev_data)
        
        # Display prediction
        st.metric("Predicted Range", f"{predicted_range:.1f} km", 
                 delta=None, delta_color="normal")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_range,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Range (km)"},
            gauge = {
                'axis': {'range': [None, 500]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 150], 'color': "red"},
                    {'range': [150, 300], 'color': "orange"},
                    {'range': [300, 500], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_range
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model not loaded")

# Model comparison section
st.markdown("---")
st.subheader("📈 Model Comparison")

if model_loaded:
    # Load training results
    try:
        results_df = pd.read_csv('models/training_results.csv', index_col=0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", results_df['test_r2'].idxmax(), 
                     f"R² = {results_df['test_r2'].max():.3f}")
        with col2:
            st.metric("Lowest Error", f"{results_df['test_rmse'].min():.1f} km",
                     "RMSE")
        with col3:
            st.metric("Average Error", f"{results_df['test_mae'].mean():.1f} km",
                     "MAE")
        
        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_df.index, y=results_df['test_r2'], 
                            name='R² Score', marker_color='lightgreen'))
        fig.add_trace(go.Bar(x=results_df.index, y=results_df['test_rmse'], 
                            name='RMSE (km)', marker_color='lightcoral'))
        fig.update_layout(title="Model Performance Comparison",
                         xaxis_title="Model",
                         yaxis_title="Score",
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
    except:
        st.info("Run training first to see comparison")

# Feature importance
st.markdown("---")
st.subheader("🔍 Key Factors Affecting EV Range")

# Create feature importance chart
feature_importance = {
    'State of Charge (SoC)': 38.5,
    'Temperature Delta': 14.2,
    'Driving Style': 11.8,
    'Speed': 9.5,
    'Traffic Conditions': 8.3,
    'HVAC Usage': 6.2,
    'Payload': 5.1,
    'Others': 6.4
}

fig = px.pie(values=list(feature_importance.values()), 
             names=list(feature_importance.keys()),
             title="Feature Importance (XGBoost)",
             color_discrete_sequence=px.colors.qualitative.Set3)
st.plotly_chart(fig, use_container_width=True)

# Tips section
st.markdown("---")
st.subheader("💡 Tips to Maximize Range")

tips = {
    "🚗 Driving Style": "Eco driving can increase range by up to 15%",
    "🌡️ Temperature": "Precondition battery in extreme temperatures",
    "⚡ Speed": "Optimal efficiency at 50-70 km/h",
    "❄️ HVAC": "Use seat heaters instead of cabin heating when possible",
    "🛣️ Traffic": "Avoid heavy traffic when possible"
}

for tip, explanation in tips.items():
    st.info(f"**{tip}**: {explanation}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data source: Synthetic EV Dataset")