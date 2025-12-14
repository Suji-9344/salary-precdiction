import streamlit as st
import pickle
import numpy as np
import os

# App title
st.title("Salary Prediction App")

# Model file name
MODEL_FILE = "salary_data.pkl"

# Check if model file exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl not found")
    st.info("👉 Upload salary_data.pkl to the SAME folder as app.py in GitHub")
    st.stop()

# Load trained model
try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Verify model
if not hasattr(model, "predict"):
    st.error("❌ The .pkl file is not a trained ML model")
    st.stop()

# User input
years_experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    try:
        result = model.predict([[years_experience]])[0]
        st.success(f"💰 Predicted Salary: ₹{result:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
