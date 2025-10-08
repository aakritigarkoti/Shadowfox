 
# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# File path
data_path = os.path.join("..", "data", "HousingData.csv")

# Step 1: Load dataset
df = pd.read_csv(data_path)

print("✅ Data loaded successfully!")
print(df.head())

# Step 2: Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 3: Split into features (X) and target (y)
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

# Step 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Data preprocessing complete!")
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
