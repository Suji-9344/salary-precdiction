import streamlit as st
import pickle
import numpy as np
import os

st.title("Salary Prediction App")

MODEL_FILE = "salary_data.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl NOT FOUND")
    st.info("👉 Put salary_data.pkl in the SAME folder as app.py")
    st.stop()

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

years = st.number_input("Years of Experience", 0.0, 50.0, step=0.1)

if st.button("Predict Salary"):
    try:
        prediction = model.predict([[years]])[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

