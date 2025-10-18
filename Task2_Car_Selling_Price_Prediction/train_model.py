# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

# 🔹 Load dataset
data_path = "data/car_data.csv"
df = pd.read_csv(data_path)

print("✅ Data Loaded Successfully!")
print(df.head())

# 🔹 Preprocessing
df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

if 'Car_Name' in df.columns:
    df.drop('Car_Name', axis=1, inplace=True)

# 🔹 Split data
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 Evaluate Model
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# 🔹 Save Model
os.makedirs("model", exist_ok=True)
with open("model/car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully at 'model/car_price_model.pkl'")
