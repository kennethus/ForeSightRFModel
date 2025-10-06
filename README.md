# 🧠 ForeSight ML Model

This repository contains the **Machine Learning model API** for [ForeSight](https://fore-sight-eight.vercel.app/) — a finance management system that predicts and visualizes student expenses.  
It combines **Random Forest Regression** and **Exponential Smoothing** to forecast future spending across multiple categories.

---

## 🚀 Overview

The ML model processes user transaction data to:
1. Predict **next month's expenses per category** (Food, Leisure, Academic, etc.) using **Random Forest**.  
2. Forecast **total monthly expenditure** using **Exponential Smoothing**.  
3. Combine both predictions through a **confidence-weighted average** based on recent model performance (R² score).

---

## 🧩 Tech Stack

| Layer | Technology | Description |
|--------|-------------|-------------|
| **Language** | Python | Core programming language for the ML model |
| **Framework** | FastAPI | Lightweight, high-performance web framework for API deployment |
| **Modeling Libraries** | scikit-learn, statsmodels | Used for Random Forest and Exponential Smoothing models |
| **Data Processing** | pandas, numpy | Data wrangling, feature engineering, and numerical computations |
| **Validation** | pydantic | Defines and validates input data schemas |
| **Server** | uvicorn | ASGI server to run the FastAPI app |
| **Version Control** | Git & GitHub | Source control and collaboration platform |



