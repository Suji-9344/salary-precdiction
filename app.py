import streamlit as st
import pickle
import numpy as np
import os

# Try to locate the model file
MODEL_FILE = "salary_data.pkl"  # make sure this file exists in the same folder

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file '{MODEL_FILE}' not found. Please check the folder.")
    st.stop()

# Load the trained model
try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app
st.title("Salary Prediction App")

years_of_experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

if st.button("Predict Salary"):
    try:
        if years_of_experience < 0:
            st.warning("Years of experience cannot be negative.")
        else:
            input_data = np.array([[years_of_experience]])
            predicted_salary = model.predict(input_data)[0]
            st.success(f"Predicted Salary: ₹{predicted_salary:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
prediction[0]:,.2f}")
