import pandas as pd
from dateutil.relativedelta import relativedelta

# Load your CSV
df = pd.read_csv("../dataset/realistic_student_transactions_april_2025.csv")

# Force date parsing and handle invalid formats as NaT
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Check if any dates failed to parse
if df['date'].isnull().any():
    print("⚠️ Warning: Some dates could not be parsed. Please check the input.")

# Apply 1-month shift only to valid dates
df['date'] = df['date'].apply(
    lambda d: d + relativedelta(months=1) if pd.notnull(d) else d
)

# Convert back to string format
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# Save the new CSV
df.to_csv("../dataset/realistic_student_transactions_may_2025.csv", index=False)
