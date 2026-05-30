# 🛒 Favorita Sales Forecasting

A machine learning project for time series sales forecasting using the **Corporación Favorita Grocery Sales Forecasting** dataset. This project focuses on predicting future unit sales for thousands of product families across Favorita stores in Ecuador.

## 📌 Overview

This project applies time-series forecasting techniques to analyze and predict grocery sales data. The goal is to build a model that accurately forecasts sales to help with inventory management and supply chain optimization.

## 📁 Project Structure

```
AI_project_2/
├── favorita_forecasting.ipynb   # Main Jupyter notebook with EDA & model training
├── data/
│   └── favorita-grocery-sales-forecasting/
│       ├── train.csv/
│       ├── test.csv/
│       ├── items.csv/
│       ├── stores.csv/
│       ├── transactions.csv/
│       ├── oil.csv/
│       └── holidays_events.csv/
└── notebooks/
```

## 🚀 Features

- Exploratory Data Analysis (EDA) on large-scale grocery sales data
- Time series feature engineering (lags, rolling means, date features)
- Model training and evaluation
- Submission-ready predictions

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — Machine learning utilities
- **Matplotlib / Seaborn** — Visualization
- **Jupyter Notebook** — Interactive development

## 📊 Dataset

Based on the [Kaggle Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) competition dataset.
Most dataset CSVs are tracked with Git LFS because `test.csv` is too large for regular GitHub blobs. The full `train.csv` file is intentionally excluded because it is about 5 GB, which is above GitHub's LFS object limit for this account. Submission output files are also excluded.

Place the Kaggle `train.csv` file at:

```text
data/favorita-grocery-sales-forecasting/train.csv/train.csv
```

## 🏃 Getting Started

```bash
# Clone the repository
git clone https://github.com/1avishek/favorita-sales-forecasting.git
cd favorita-sales-forecasting

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch notebook
jupyter notebook favorita_forecasting.ipynb
```

## 📄 License

MIT License
