# train_model.py

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Load dataset (correct path and filename)
data_path = r"D:\boston_house_price_prediction\data\HousingData.csv"

# Read CSV file
df = pd.read_csv(data_path)

# ✅ Clean data — remove missing (NaN) values
df = df.dropna()

# ✅ Check basic info
print("✅ Data loaded successfully!")
print(f"Shape after cleaning: {df.shape}")
print("\nColumns in dataset:", list(df.columns))

# ✅ Split data into features (X) and target (y)
# Make sure your target column is named 'HousePrice'
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

# ✅ Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ✅ Make predictions
y_pred = model.predict(X_test_scaled)

# ✅ Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model trained successfully!")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# ✅ Save model and scaler for later use
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/house_price_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("\n💾 Model and scaler saved successfully in 'models' folder!")
