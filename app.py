import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------
# Create Salary Dataset
# ---------------------------
data = {
    "YearsExperience": [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 35000, 38000, 40000, 45000, 50000,
               60000, 70000, 80000, 90000, 100000, 110000, 120000]
}

df = pd.DataFrame(data)

# ---------------------------
# Train Salary Model
# ---------------------------
X = df[["YearsExperience"]]   # Feature
y = df["Salary"]              # Target

model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("💼 Salary Prediction App")
st.write("Predict salary based on years of experience")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.5
)

if st.button("Predict Salary"):
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Salary: ₹{prediction[0]:,.2f}")

