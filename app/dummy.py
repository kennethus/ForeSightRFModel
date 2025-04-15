from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import json

# Your model
class PreviousForecast(BaseModel):
    userId: str
    forecasted: List[float]
    dates: List[str]

# Generate dummy data
np.random.seed(42)  # For reproducibility

# Create timezone-aware datetime index
dates = pd.date_range(start='2024-01-01', end='2024-03-31', tz='UTC')

forecasted = np.random.normal(loc=250, scale=50, size=len(dates)).round(2)  # Simulate around 250 daily expense

# Format dates to "2024-03-26 00:00:00+00:00"
formatted_dates = dates.strftime('%Y-%m-%d %H:%M:%S%z').str.replace(r'(\d{2})(\d{2})$', r'\1:\2', regex=True).tolist()

# Create the object
dummy_forecast = PreviousForecast(
    userId="user123",
    forecasted=forecasted.tolist(),
    dates=formatted_dates
)

# Print JSON with double quotes
print(json.dumps(dummy_forecast.model_dump(), ensure_ascii=False, indent=2))
