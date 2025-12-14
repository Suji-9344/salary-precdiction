import streamlit as st
import pickle
import numpy as np

# Load salary model
model = pickle.load(open("salary_data.pkl", "rb"))

st.title("Salary Prediction App")

experience = st.number_input("Years of Experience", min_value=0.0, step=0.5)

if st.button("Predict Salary"):
    salary = model.predict([[experience]])
    st.success(f"Predicted Salary: ₹ {salary[0]:,.2f}"
