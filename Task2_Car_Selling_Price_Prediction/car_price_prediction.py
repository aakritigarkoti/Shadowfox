# car_price_prediction.py

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Step 2: Load dataset
data = pd.read_csv('data/car_data.csv')
print("✅ Data Loaded Successfully!")
print(data.head())

# Step 3: Basic info
print("\nDataset Info:")
print(data.info())

# Step 4: Data preprocessing
data['Current_Year'] = 2025
data['Car_Age'] = data['Current_Year'] - data['Year']
data.drop(['Car_Name', 'Year', 'Current_Year'], axis=1, inplace=True)

# Convert categorical to numerical
data = pd.get_dummies(data, drop_first=True)

# Step 5: Define X and y
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
# predict_price.py

import pickle
import numpy as np

# Load trained model
model_path = "model/car_price_model.pkl"
model = pickle.load(open(model_path, 'rb'))

print("✅ Model Loaded Successfully!")

# --- Example Input ---
# Order of inputs:
# [Present_Price, Kms_Driven, Owner, Car_Age, Fuel_Type, Seller_Type, Transmission]

example_input = [5.5, 40000, 0, 5, 0, 0, 1]  # Example values

# Convert to numpy array
final_input = np.array(example_input).reshape(1, -1)

# Predict
predicted_price = model.predict(final_input)[0]
print(f"💰 Predicted Selling Price: ₹{predicted_price:.2f} lakhs")

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 8: Predict
pred = model.predict(X_test)

# Step 9: Evaluate
score = r2_score(y_test, pred)
print(f"\n✅ Model R2 Score: {score:.2f}")

# Step 10: Visualization
plt.figure(figsize=(6,4))
plt.scatter(y_test, pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
