# BankWise: Bank Customer Analytics

## Project Overview
BankWise is a machine learning project that analyzes bank customer data using two models: Linear Regression and Logistic Regression. The project includes a simple web interface built with HTML, CSS, and JavaScript.

## Business Problems

1. **Balance Prediction:** Estimate a customer's bank account balance based on selected customer information.
2. **Customer Churn Prediction:** Predict whether a customer is likely to leave the bank.

## Machine Learning Models

### 1. Linear Regression
- **Type:** Regression
- **Purpose:** Predicts a customer's bank account balance.
- **Target variable:** `Balance`

### 2. Logistic Regression
- **Type:** Classification
- **Purpose:** Predicts whether a customer will leave the bank.
- **Target variable:** `Exited`
- **Output:** Likely to leave or likely to stay.

## Dataset
The project uses the `Churn_Modelling.csv` dataset, which contains bank customer information, including age, geography, credit score, account balance, and customer exit status.

## Technologies Used
- **HTML** â€” webpage structure
- **CSS** â€” webpage styling and layout
- **JavaScript** â€” user interactions
- **Python** â€” backend and machine learning
- **Pandas** â€” data processing
- **Scikit-learn** â€” model training
- **FastAPI** â€” backend API

## Project Features
- User-friendly web interface
- Customer balance prediction using Linear Regression
- Customer churn classification using Logistic Regression
- Integration between the frontend and Python backend

## How to Run the Project on macOS

1. Open the project folder in VS Code.
2. Open the terminal in VS Code.
3. Install the required packages:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. Make sure `Churn_Modelling.csv` is in the project folder.
5. Train the two models:

   ```bash
   python3 train_models.py
   ```

6. Start the web application:

   ```bash
   python3 -m uvicorn main:app --reload
   ```

7. Open `http://127.0.0.1:8000` in your browser.

## Conclusion
This project demonstrates how regression and classification can be applied to banking data to support balance estimation and customer churn analysis through a simple web application.
