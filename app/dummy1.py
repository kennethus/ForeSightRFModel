import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# Parameters
start_date = datetime(2025, 4, 1)
end_date = datetime(2025, 4, 30)
monthly_allowance = 10000
daily_food_range = (100, 200)
rent_amount = 2500
wifi_amount = 350
electricity_range = (250, 400)
water_range = (100, 200)
academic_items = ["Notebook", "Ballpen", "Photocopy", "Lab Manual", "Book"]
leisure_items = ["Netflix", "Movie Ticket", "Mobile Game", "Snack Out", "Spotify"]
transport_items = ["Jeep Fare", "Bus Fare", "Tricycle Fare", "Ride Share"]

transactions = []
current_date = start_date

# Counters for monthly events
academic_days = set()
leisure_days = set()

# Generate random days per month for academic and leisure expenses
for month in [1, 2, 3]:
    academic_days.update(random.sample(range(1, 28), 3))  # 3 academic expenses
    leisure_days.update(random.sample(range(1, 28), 6))   # 6 leisure events

while current_date <= end_date:
    day = current_date.day
    month = current_date.month
    year = current_date.year
    weekday = current_date.weekday()

    date_str = current_date.strftime("%Y-%m-%d")

    # Daily food expense (except Sunday)
    if weekday < 6:
        transactions.append({
            "name": "Daily Food",
            "description": "Meal and drink",
            "totalAmount": round(random.uniform(*daily_food_range), 2),
            "category": "Food and Dining",
            "type": "Expense",
            "date": date_str
        })

    # Transportation: 0 to 2 rides a day, or skip every few days
    if random.random() > 0.2:  # 80% chance of transport today
        for _ in range(random.choice([1, 2])):  # 1 or 2 trips
            transactions.append({
                "name": random.choice(transport_items),
                "description": "Daily commute",
                "totalAmount": round(random.uniform(10, 30), 2),
                "category": "Transportation",
                "type": "Expense",
                "date": date_str
            })

    # Leisure: occasionally
    if day in leisure_days and random.random() < 0.5:
        transactions.append({
            "name": random.choice(leisure_items),
            "description": "Leisure or entertainment",
            "totalAmount": round(random.uniform(50, 300), 2),
            "category": "Leisure and Entertainment",
            "type": "Expense",
            "date": date_str
        })

    # Academic: randomly on selected days
    if day in academic_days and random.random() < 0.5:
        transactions.append({
            "name": random.choice(academic_items),
            "description": "Academic supplies or materials",
            "totalAmount": round(random.uniform(30, 500), 2),
            "category": "Academic",
            "type": "Expense",
            "date": date_str
        })

    # Monthly allowance and bills at end of month
    if day >= 28:
        if day == 28:
            # Allowance
            transactions.append({
                "name": "Monthly Allowance",
                "description": "Allowance from parents",
                "totalAmount": monthly_allowance,
                "category": "Leisure and Entertainment",
                "type": "Income",
                "date": date_str
            })
            # Rent
            transactions.append({
                "name": "Monthly Rent",
                "description": "Boarding house rent",
                "totalAmount": rent_amount,
                "category": "Living",
                "type": "Expense",
                "date": date_str
            })

        if day == 29:
            transactions.append({
                "name": "Electricity Bill",
                "description": "Monthly electric bill",
                "totalAmount": round(random.uniform(*electricity_range), 2),
                "category": "Living",
                "type": "Expense",
                "date": date_str
            })
        elif day == 30:
            transactions.append({
                "name": "WiFi Bill",
                "description": "Internet service fee",
                "totalAmount": wifi_amount,
                "category": "Living",
                "type": "Expense",
                "date": date_str
            })
        elif day == 31 and month in [1, 3]:
            transactions.append({
                "name": "Water Bill",
                "description": "Water utility payment",
                "totalAmount": round(random.uniform(*water_range), 2),
                "category": "Living",
                "type": "Expense",
                "date": date_str
            })

    current_date += timedelta(days=1)

# Save to CSV
df = pd.DataFrame(transactions)
df.to_csv("../dataset/realistic_student_transactions_april_2025.csv", index=False)
print("CSV file created")
