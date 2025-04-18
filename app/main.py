import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from app import Exponential
from typing import List, Optional
from sklearn.metrics import r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "SocioDemoRFModel.pkl")
rf_model_r2 = 0.85

with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.lower().replace(',', '').strip()
        if 'k' in value:
            return float(value.replace('k', '')) * 1000
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

def preprocess_input(raw_input: dict, reference_columns: list) -> pd.DataFrame:
    df_input = pd.DataFrame([raw_input])

    df_input["Monthly_Allowance"] = df_input["Monthly_Allowance"].apply(convert_to_numeric)

    age_mapping = {"Under 18": 1, "18-20": 2, "21-23": 3, "24-26": 4, "27 and above": 5}
    year_level_mapping = {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4}
    roommates_mapping = {
        "With family": 0, "I live alone": 1, "I live with 1 roommate": 2,
        "I live with 2-3 roommates": 3, "I live with more than 3 roommates": 4
    }
    study_hours_mapping = {
        "Less then 10 hours": 1, "10-20 hours": 2, "21-30 hours": 3,
        "31-40 hours": 4, "More then 40 hours": 5
    }
    income_mapping = {
        "Less than P12,030": 1, "P12,031 - P24,060": 2, "P24,061 - P48,120": 3,
        "P48,121 - P84,210": 4, "P84,211 - P144,360": 5, "P144,361 - P240,600": 6,
        "More than P240,601": 7
    }
    going_home_mapping = {
        "Not at all": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5
    }

    df_input["Age_Group"] = df_input["Age_Group"].map(age_mapping)
    df_input["Year_Level"] = df_input["Year_Level"].map(year_level_mapping)
    df_input["Roommates"] = df_input["Roommates"].map(roommates_mapping)
    df_input["Hours_of_Study_per_Week"] = df_input["Hours_of_Study_per_Week"].map(study_hours_mapping)
    df_input["Family_Monthly_Income"] = df_input["Family_Monthly_Income"].map(income_mapping)
    df_input["Frequency_of_Going_Home"] = df_input["Frequency_of_Going_Home"].map(going_home_mapping)

    yes_no_cols = [col for col in df_input.columns if df_input[col].dropna().isin(["Yes", "No"]).all()]
    df_input[yes_no_cols] = df_input[yes_no_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

    categorical_cols = df_input.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in yes_no_cols]
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    for col in reference_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[reference_columns]

    for col in df_input.select_dtypes(include=['bool']).columns:
        df_input[col] = df_input[col].astype(int)

    return df_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "dataset", "Student-Spending-Habits_PreProcessed.csv")

reference_df = pd.read_csv(csv_path)
reference_columns = reference_df.drop(columns=[
    "Living_Expenses", "Food_and_Dining_Expenses", 
    "Transportation_Expenses", "Leisure_and_Entertainment_Expenses", 
    "Academic_Expenses"
]).columns.tolist()

class UserInput(BaseModel):
    Age_Group: str
    Sex: str
    Year_Level: str
    In_relationship: str
    Personality: str
    Home_Region: str
    Living_Situation: str
    Dorm_Area: str
    Roommates: str
    Degree_Program: str
    In_Organization: str
    Hours_of_Study_per_Week: str
    Monthly_Allowance: str
    Family_Monthly_Income: str
    Have_Scholarship: str
    Have_Job: str
    Meal_Preferences: str
    Frequency_of_Going_Home: str
    Have_Health_Concern: str
    Preferred_Payment_Method: str

class ExpenseItem(BaseModel):
    userId: str
    name: str
    totalAmount: float
    category: str
    type: str
    description: str
    date: str

class PreviousForecast(BaseModel):
    userId: str
    forecasted: List[float]
    dates: List[str]

class CombinedInput(BaseModel):
    user_data: UserInput
    transactions: Optional[List[ExpenseItem]] = None
    previous_forcast: Optional[PreviousForecast] = None

@app.get("/")
def home():
    return {"health_check": "OK"}



@app.post("/predict")
def combined_predict(data: CombinedInput):
    # Predict from RF
    input_df = preprocess_input(data.user_data.model_dump(), reference_columns)
    rf_preds = model.predict(input_df)[0]
    rf_categories = [
        "Living_Expenses", "Food_and_Dining_Expenses", 
        "Transportation_Expenses", "Leisure_and_Entertainment_Expenses", 
        "Academic_Expenses"
    ]
    rf_pred_dict = dict(zip(rf_categories, rf_preds.tolist()))
    rf_total = sum(rf_preds)

    # Prepare transactions for Exponential Smoothing
    txn_dicts = [t.model_dump() for t in data.transactions] if data.transactions else []

    # Forecast from Exponential Smoothing
    es_prediction = Exponential.forecast_expenses(txn_dicts, data.previous_forcast)

    if es_prediction["success"]:
        es_r2 = es_prediction["metrics"].get("r2", 1 - rf_model_r2) #Default is 0.15
        es_total = es_prediction["metrics"]["total_forecasted"]

        # Clamp r2 to [0, 1]
        es_r2 = max(es_r2, 0)

        # Confidence-weighted total
        combined_total = es_r2 * es_total + (1 - es_r2) * rf_total

        # Rescale RF predictions to match combined total
        scaling_factor = combined_total / rf_total if rf_total != 0 else 0
        scaled_rf = {k: v * scaling_factor for k, v in rf_pred_dict.items()}

        return {
            "es_success": True,
            "combined_total": combined_total,
            "categories": scaled_rf,
            "rf_total": rf_total,
            "es_prediction": es_prediction,
            "es_r2_score": es_r2
        }
    else:
        # Fall back to RF-only predictions
        return {
            "es_success": False,
            "categories": rf_pred_dict,
            "rf_total": rf_total
        }
