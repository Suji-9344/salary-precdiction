import streamlit as st
import pickle
import os

st.title("Salary Prediction App")

MODEL_FILE = "salary_data.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("❌ salary_data.pkl not found")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

if not hasattr(model, "predict"):
    st.error("❌ salary_data.pkl is NOT a trained ML model")
    st.stop()

age = st.number_input("Enter Age", 0, 120, step=1)

if st.button("Predict Salary"):
    salary = model.predict([[age]])[0]
    st.success(f"💰 Predicted Salary: ₹{salary:,.2f}")
   
