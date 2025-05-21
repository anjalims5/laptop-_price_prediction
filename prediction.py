import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("C:/Users/anjal/OneDrive/Desktop/lap/random_forest_model.pkl")
scaler = joblib.load("C:/Users/anjal/OneDrive/Desktop/lap/scaler.pkl")

# Encoding dictionary for brand (make sure matches your training)
brand_map = {
    'Apple': 0,
    'Dell': 1,
    'HP': 2,
    'Lenovo': 3,
    'Acer': 4,
    'Asus': 5,
    'MSI': 6,
    'Other': 7
}

st.title("Laptop Price Prediction")

brand = st.selectbox("Brand", list(brand_map.keys()))
ram_size = st.number_input("RAM Size (GB)", min_value=1, max_value=128, value=8)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
storage_capacity = st.number_input("Storage Capacity (GB)", min_value=0, max_value=5000, value=256, step=1)
processor_speed = st.number_input("Processor Speed (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
if st.button("Predict Price"):
    try:
        # Encode brand
        brand_enc = brand_map[brand]

        # Prepare input array
        input_data = np.array([[brand_enc, ram_size, weight, screen_size, storage_capacity,processor_speed]])

        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(f"Estimated Laptop Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

