import streamlit as st
import pickle
import numpy as np
import os

st.title("Salary Prediction App")
st.write("Predict salary based on Years of Experience")

MODEL_FILE = "salary_data.pkl"

# Check file exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl not found")
    st.stop()

# Load model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Validate model
if not hasattr(model, "predict"):
    st.error("❌ Loaded file is NOT a trained ML model")
    st.stop()

# Input
experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    try:
        prediction = model.predict(np.array([[experience]]))[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

