# 1. Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 2. Load dataset
# Make sure Salary_Data.csv is in the same folder
data = pd.read_csv("Salary_Data.csv")

# 3. Select input (X) and output (y)
X = data[["YearsExperience"]]   # Input feature
y = data["Salary"]              # Target

# 4. Train the model
model = LinearRegression()
model.fit(X, y)

# 5. Save the TRAINED MODEL (NOT dataset)
with open("salary_data.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Trained model saved successfully as salary_data.pkl")
