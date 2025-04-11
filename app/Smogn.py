import numpy as np
import pandas as pd
import smogn

# Load the dataset
file_path = "dataset/Student-Spending-Habits.csv"
df = pd.read_csv(file_path)

print(df.head())

# Step 1: Handle missing values by dropping rows with NaN values
df_cleaned = df.dropna().reset_index(drop=True)

# Step 2: Convert object-type numerical fields to numeric
numeric_cols = ["Monthly_Allowance"]
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

df_cleaned[numeric_cols] = df_cleaned[numeric_cols].applymap(convert_to_numeric)

# Step 5: Convert expenses into percentages of Monthly Allowance
expense_cols = ["Living_Expenses", "Food_and_Dining_Expenses", 
                "Transportation_Expenses", "Leisure_and_Entertainment_Expenses", "Academic_Expenses"]

# Mapping categorical values
age_mapping = {
    "Under 18": 1,
    "18-20": 2,
    "21-23": 3,
    "24-26": 4,
    "27 and above": 5,
}
year_level_mapping = {
    "Freshman": 1,
    "Sophomore": 2,
    "Junior": 3,
    "Senior": 4
}
roommates_mapping = {
    "With family": 0,
    "I live alone": 1,
    "I live with 1 roommate": 2,
    "I live with 2-3 roommates": 3,
    "I live with more than 3 roommates": 4
}
study_hours_mapping = {
    "Less then 10 hours": 1,
    "10-20 hours": 2,
    "21-30 hours": 3,
    "31-40 hours": 4,
    "More then 40 hours": 5
} 
income_mapping = {
    "Less than P12,030": 1,
    "P12,031 - P24,060": 2,
    "P24,061 - P48,120": 3,
    "P48,121 - P84,210": 4,
    "P84,211 - P144,360": 5,
    "P144,361 - P240,600": 6,
    "More than P240,601": 7
}
going_home_mapping = {
    "Not at all": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

df_cleaned["Age_Group"] = df_cleaned["Age_Group"].map(age_mapping)
df_cleaned["Year_Level"] = df_cleaned["Year_Level"].map(year_level_mapping)
df_cleaned["Roommates"] = df_cleaned["Roommates"].map(roommates_mapping)
df_cleaned["Hours_of_Study_per_Week"] = df_cleaned["Hours_of_Study_per_Week"].map(study_hours_mapping)
df_cleaned["Family_Monthly_Income"] = df_cleaned["Family_Monthly_Income"].map(income_mapping)
df_cleaned["Frequency_of_Going_Home"] = df_cleaned["Frequency_of_Going_Home"].map(going_home_mapping)

# Identify and encode binary Yes/No columns
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
yes_no_cols = [col for col in df_cleaned.columns if df_cleaned[col].dropna().isin(["Yes", "No"]).all()]
df_cleaned[yes_no_cols] = df_cleaned[yes_no_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))
categorical_cols = [col for col in categorical_cols if col not in yes_no_cols]
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

# Separate features and targets
X = df_encoded.drop(columns=expense_cols)
y = df_encoded[expense_cols]

# -------------- ✅ SMOGN BALANCING SECTION --------------
# Combine X and y for SMOGN processing
df_combined = pd.concat([X, y], axis=1)

# Drop NaN/infinite and constant columns
df_combined = df_combined.replace([np.inf, -np.inf], np.nan).dropna()
nunique = df_combined.apply(pd.Series.nunique)
df_combined = df_combined.drop(columns=nunique[nunique == 1].index.tolist())

# Ensure all boolean columns are int
for col in df_combined.select_dtypes(include=['bool']).columns:
    df_combined[col] = df_combined[col].astype(int)

# Apply SMOGN per target variable
augmented_dataframes = []
for target in expense_cols:
    print(f"/nApplying SMOGN to: {target}")
    smogn_input = df_combined.copy()
    smogn_input = smogn_input.rename(columns={target: 'y'})
    try:
        smogn_result = smogn.smoter(data=smogn_input, y='y', rel_thres=0.8, rel_method='auto')
        smogn_result = smogn_result.rename(columns={'y': target})
        augmented_dataframes.append(smogn_result)
    except Exception as e:
        print(f"Error applying SMOGN to {target}: {e}")

# Merge all balanced target datasets
from functools import reduce
from operator import or_

# Use index to align rows and drop duplicates
augmented_data = pd.concat(augmented_dataframes, ignore_index=True).drop_duplicates()

# Save to CSV
output_path = "dataset/Student-Spending-Habits_PreProcessed.csv"
augmented_data.to_csv(output_path, index=False)
print(f"✅ Saved combined dataset with synthetic rows at: {output_path}")