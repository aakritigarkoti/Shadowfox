# 🚗 Task 2 - Car Selling Price Prediction

This project aims to **predict the approximate selling price of a car** based on various features such as model year, kilometers driven, fuel type, seller type, transmission, and ownership details.  
It uses **Machine Learning (Linear Regression & Random Forest)** models to analyze the data and make accurate price predictions.

---

## 📁 Project Structure

Task2_Car_Selling_Price_Prediction/
│
├── data/
│ └── car_data.csv # Dataset containing car details
│
├── train_model.py # Script to train and save the ML model
├── predict_price.py # Script to load model and make predictions
└── README.md # Project documentation

yaml
Copy code

---

## 🧩 Features Used

| Feature Name | Description |
|---------------|-------------|
| **Year** | Manufacturing year of the car |
| **Selling_Price** | Price at which the car was sold |
| **Present_Price** | Current ex-showroom price |
| **Kms_Driven** | Total distance covered |
| **Fuel_Type** | Type of fuel (Petrol / Diesel / CNG) |
| **Seller_Type** | Dealer or Individual |
| **Transmission** | Manual or Automatic |
| **Owner** | Number of previous owners |

---

## ⚙️ Libraries Used

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

## 🚀 Steps to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/aakritigarkoti/Shadowfox.git
Navigate to project folder

bash
Copy code
cd Shadowfox/Task2_Car_Selling_Price_Prediction
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run training script

bash
Copy code
python train_model.py
Predict car price

bash
Copy code
python predict_price.py
📊 Insights
Diesel cars generally have higher resale value than Petrol cars.

Older cars (pre-2012) show a steep decline in price.

Manual transmission cars dominate the used car market.

🏁 Output Example
mathematica
Copy code
Enter Year of the Car: 2017  
Enter Present Price (in lakhs): 9.50  
Enter Kms Driven: 69000  
Enter Fuel Type (Petrol/Diesel/CNG): Petrol  
Enter Seller Type (Dealer/Individual): Dealer  
Enter Transmission (Manual/Automatic): Manual  
Enter Owner Count (0/1/2): 0  

✅ Predicted Selling Price: ₹5.85 lakhs


✨ Author
👩‍💻 Aakriti Garkoti
📚 Student | Aspiring Data Scientist
🔗 GitHub Profile

