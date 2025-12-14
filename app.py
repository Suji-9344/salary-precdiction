import streamlit as st
import pickle
import numpy as np
import os

st.title("Salary Prediction App")

# Model filename
MODEL_FILE = "salary_data.pkl"

# Check model exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl not found")
    st.info("👉 Upload salary_data.pkl to the SAME folder as app.py")
    st.stop()

# Load model
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Validate model
if not hasattr(model, "predict"):
    st.error("❌ Loaded file is NOT a trained ML model")
    st.stop()

# User input
years = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    try:
        salary = model.predict([[years]])[0]
        st.success(f"💰 Predicted Salary: ₹{salary:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
