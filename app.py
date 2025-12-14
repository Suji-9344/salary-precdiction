import streamlit as st
import pickle
import numpy as np
import os

# App title
st.title("Salary Prediction App")

st.write("Predict salary based on Years of Experience")

# Model file name (must be in same folder)
MODEL_FILE = "salary_data.pkl"

# Check model file exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl file not found")
    st.info("👉 Upload salary_data.pkl to the SAME folder as app.py in GitHub")
    st.stop()

# Load trained model
try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Ensure loaded object is a model
if not hasattr(model, "predict"):
    st.error("❌ salary_data.pkl is NOT a trained ML model")
    st.stop()

# User input
years_experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction button
if st.button("Predict Salary"):
    try:
        prediction = model.predict([[years_experience]])[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

   
