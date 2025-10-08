# predict_price.py

import joblib
import numpy as np

# ✅ Load trained model and scaler
model = joblib.load("../models/house_price_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

print("\n🏠 Boston House Price Prediction System")
print("----------------------------------------")
print("Enter the following details:\n")

# ✅ Take user input for 6 features
crime_rate = float(input("Crime Rate: "))
res_land_zone = float(input("Residential Land Zone: "))
indus_acres = float(input("Industrial Acres: "))
charles_river = int(input("Charles River (1 if bounds river, 0 otherwise): "))
nox_conc = float(input("Nitric Oxides Concentration: "))
avg_rooms = float(input("Average Rooms per Dwelling: "))

# ✅ Prepare the input data for prediction
features = np.array([[crime_rate, res_land_zone, indus_acres, charles_river, nox_conc, avg_rooms]])

# ✅ Scale features (important for accuracy)
features_scaled = scaler.transform(features)

# ✅ Predict price
predicted_price = model.predict(features_scaled)

# ✅ Display result
print("\n💰 Predicted House Price:")
print(f"➡️ ${predicted_price[0]:.2f}k (thousand dollars)")
