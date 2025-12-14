import streamlit as st
import pickle
import numpy as np
import os

st.title("Salary Prediction App")

MODEL_FILE = "salary_data.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl not found")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

if not hasattr(model, "predict"):
    st.error("❌ Loaded file is NOT a trained model")
    st.stop()

years = st.number_input("Years of Experience", 0.0, 50.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict([[years]])[0]
    st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

