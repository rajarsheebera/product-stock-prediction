# 📦 Supply Chain Stock Prediction (Ridge Regression + Streamlit)

## Overview

This project is a small experiment to understand how machine learning can help in **inventory planning**.
The goal is simple: based on operational factors like sales, lead time, production volume, and costs, the model estimates how much stock should be maintained.

It’s not meant to be a perfect forecasting system. Instead, it demonstrates how data can support supply-chain decisions such as avoiding overstocking or preventing shortages.

---

## Why This Project?

In real supply chains, companies constantly struggle with:

* Keeping **too much stock** → increases storage and holding costs
* Keeping **too little stock** → leads to missed sales and delays
* Balancing demand, production, and logistics

This project tries to model that relationship using a regression approach.

---

## Dataset

The dataset contains operational supply-chain information such as:

* Price
* Number of products sold
* Lead times
* Order quantities
* Production volumes
* Revenue generated
* Availability
* Shipping times
* Manufacturing lead time and costs
* Defect rates
* Overall costs
* Stock levels (used as the prediction target)

Each row represents a snapshot of supply-chain activity rather than a time series.

---

## Approach

### 1. Data Cleaning

Some text columns were simplified and unnecessary identifiers (like SKU, Supplier, Routes) were removed because they don’t help numeric prediction.

### 2. Feature Selection

Only operational variables that influence inventory were kept.
This helps reduce noise and improves model stability (important since the dataset is small).

### 3. Scaling

Features were standardized using `StandardScaler` so that variables with different units don’t dominate the regression.

### 4. Model Used — Ridge Regression

Ridge Regression was chosen because:

* The dataset is relatively small.
* Many variables are correlated.
* Regularization helps prevent overfitting.
* It provides more stable predictions compared to ordinary linear regression.

### 5. Evaluation Metrics

We used:

* **Mean Squared Error (MSE)** → Measures prediction error magnitude
* **R² Score** → Indicates how well the model explains stock variation

Since this is not a large industrial dataset, extremely high accuracy is not expected.

---

## Streamlit Application

A simple Streamlit app was created to make the model interactive.

Users can input operational values manually and get an estimated stock requirement instantly.
This simulates how such a model might be used in a planning dashboard.

---

## How to Run the Project

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### Train the Model (if needed)

Run the notebook/script to generate:

```
ridge_model.pkl
scaler.pkl
```

### Launch the App

```bash
streamlit run app.py
```

The browser will open automatically with the prediction interface.

---

## Project Structure

```
project/
│
├── app.py                  # Streamlit UI
├── supply_chain_data.csv   # Dataset
├── ridge_model.pkl         # Trained model
├── scaler.pkl              # Feature scaler
└── README.md
```

---

## Limitations

This model has some important limitations:

* The dataset is small (≈100 rows), so predictions are indicative, not definitive.
* It’s not time-series data, so it cannot capture seasonality or demand trends.
* Real supply-chain systems would use much larger historical datasets.

This project should be seen as a **learning prototype**, not a production forecasting tool.

---

## What This Demonstrates

Even with a simple dataset, we can:

* Connect business problems with ML workflows
* Perform preprocessing and feature engineering
* Use regularized regression for stable modeling
* Deploy predictions through a lightweight web interface

---

## Possible Improvements (Future Work)

* Use time-series demand data for better forecasting
* Add classification for low-stock risk alerts
* Integrate visualization dashboards
* Automate retraining with new data

---

## Final Note

This project was built as a practical exercise to bridge theory and application — taking a common logistics problem and exploring how machine learning can assist decision-making in a simple, understandable way.
