import streamlit as st
import pickle
import numpy as np
import os

st.title("Salary Prediction App")

# Path to your model
MODEL_FILE = "salary_data.pkl"

# Step 1: Check if the file exists
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Model file '{MODEL_FILE}' not found! Please put it in the same folder as app.py.")
    st.stop()

# Step 2: Load the trained model
try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Step 3: Get user input
years_of_experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Step 4: Predict salary
if st.button("Predict Salary"):
    if years_of_experience < 0:
        st.warning("⚠️ Years of experience cannot be negative.")
    else:
        try:
            # Ensure the loaded object has a predict method
            if not hasattr(model, "predict"):
                st.error("❌ The loaded file is not a trained model. Please check 'salary_data.pkl'.")
            else:
                input_data = np.array([[years_of_experience]])
                predicted_salary = model.predict(input_data)[0]
                st.success(f"💰 Predicted Salary: ₹{predicted_salary:,.2f}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
