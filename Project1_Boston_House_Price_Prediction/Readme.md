# 🏠 Task 1 - Boston House Price Prediction

This project predicts **housing prices in Boston** based on multiple features such as crime rate, number of rooms, property tax rate, and accessibility to highways.  
It uses **Machine Learning Regression Models** to estimate the price of houses using the famous **Boston Housing Dataset**.

---

## 📁 Project Structure

Task1_Boston_House_Price_Prediction/
│
├── data/
│ └── HousingData.csv # Dataset used for training & analysis
│
├── car_price_prediction.py # (Main script for data processing, model training & visualization)
├── train_model.py # (Model training and saving script)
└── README.md # Documentation file



---

## 🧩 Features Used

| Feature Name | Description |
|---------------|-------------|
| **CRIM** | Per capita crime rate by town |
| **ZN** | Proportion of residential land zoned for large lots |
| **INDUS** | Proportion of non-retail business acres per town |
| **CHAS** | Charles River dummy variable (1 if tract bounds river) |
| **NOX** | Nitric oxides concentration (parts per 10 million) |
| **RM** | Average number of rooms per dwelling |
| **AGE** | Proportion of owner-occupied units built prior to 1940 |
| **DIS** | Weighted distances to five Boston employment centres |
| **RAD** | Index of accessibility to radial highways |
| **TAX** | Full-value property tax rate per $10,000 |
| **PTRATIO** | Pupil-teacher ratio by town |
| **B** | Proportion of Black population |
| **LSTAT** | % lower status of the population |
| **MEDV** | Median value of owner-occupied homes (target variable) |

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
cd Shadowfox/Task1_Boston_House_Price_Prediction
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the model

bash

python train_model.py
📊 Insights from Data
Houses closer to the Charles River have higher median values.

As the crime rate increases, the housing price decreases.

Number of rooms (RM) strongly correlates with housing prices.

🏁 Example Output
nginx

Predicted Median Value of House: $27.5k
✨ Future Enhancements
Use advanced models like XGBoost or Random Forest for higher accuracy.

Add a web interface using Flask to input values and get real-time predictions.

Deploy the model on a cloud platform (Streamlit / Render / HuggingFace Spaces).

👩‍💻 Author
Aakriti Garkoti
🎓 Student | 💡 Data Science Enthusiast
🔗 GitHub Profile
