import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("Salary_Data.csv")

# Features and target
X = data[["YearsExperience"]]
y = data["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save TRAINED MODEL (important!)
with open("salary_data.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Trained model saved as salary_data.pkl")
