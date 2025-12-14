import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.title("Salary Prediction App")
st.write("Predict salary based on Years of Experience")

# =========================
# Load dataset from GitHub repo
# =========================
DATA_FILE = "Salary_Data.csv"

if not os.path.exists(DATA_FILE):
    st.error("❌ Salary_Data.csv not found in repository")
    st.info("👉 Upload Salary_Data.csv to the SAME folder as app.py")
    st.stop()

# Read dataset
data = pd.read_csv(DATA_FILE)

# Validate required columns
required_columns = ["YearsExperience", "Salary"]
if not all(col in data.columns for col in required_columns):
    st.error("❌ Dataset must contain 'YearsExperience' and 'Salary' columns")
    st.stop()

# =========================
# Train model
# =========================
X = data[["YearsExperience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

# =========================
# User input
# =========================
experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# =========================
# Prediction
# =========================
if st.button("Predict Salary"):
    prediction = model.predict(np.array([[experience]]))[0]
    st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
