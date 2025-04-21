import random
import pandas as pd
from datetime import datetime, timedelta

# Target daily totals for April 2025
daily_totals = [
    202.39185343922142, 265.7127572843509, 300.9917483577318, 183.66441366280887,
    171.62640696995533, 182.98937193979702, 258.96463679412057, 76.49020557619198,
    175.63283219246554, 252.26973406123048, 235.68428822888995, 169.5599431293777,
    474.63544291556354, 313.78854227024965, 195.40989838642162, 301.07238521368134,
    156.8778067845826, 183.45458610643422, 199.38866090573913, 154.99365323632225,
    375.1928652639365, 153.67617640430262, 301.8646932092109, 203.51918964920247,
    183.67219425920837, 216.6975206520057, 899.5884154698427, 407.10799323788285,
    2054.5538537367797, 279.40583528081436
]

# Categories
categories = ["Food and Dining", "Transportation", "Academic", "Leisure and Entertainment", "Living"]

def generate_transactions_for_day(date, total):
    transactions = []
    remaining = total
    num_transactions = random.randint(2, 5)
    
    for i in range(num_transactions):
        if i == num_transactions - 1:
            amount = round(remaining, 2)
        else:
            max_amount = remaining / (num_transactions - i) * 1.5
            amount = round(random.uniform(0.3 * total / num_transactions, max_amount), 2)
            remaining -= amount

        transactions.append({
            "name":  f"Transaction {i+1}",
            "date": date.strftime("%Y-%m-%d"),
            "category": random.choice(categories),
            "description": f"Auto-generated transaction {i+1}",
            "totalAmount": amount,
            "type": "Expense"
        })
    
    return transactions

# Generate all transactions for April 2025
all_transactions = []
start_date = datetime(2025, 4, 1)

for i, total in enumerate(daily_totals):
    date = start_date + timedelta(days=i)
    all_transactions.extend(generate_transactions_for_day(date, total))

# Save to CSV
df = pd.DataFrame(all_transactions)
df.to_csv("../dataset/synthetic_april_2025_transactions.csv", index=False)
