def generate_adjustment_message(category_label, actual, previous_prediction, current_prediction, adjusted, confidence):
    forecast_error = previous_prediction - actual
    error_percent = (abs(forecast_error) / previous_prediction * 100) if previous_prediction != 0 else 0
    adjustment_diff = adjusted - current_prediction
    adjustment_percent = (abs(adjustment_diff) / current_prediction * 100) if current_prediction != 0 else 0

    actual_str = f"₱{actual:,.2f}"
    previous_str = f"₱{previous_prediction:,.2f}"
    adjusted_str = f"₱{adjusted:,.2f}"
    current_str = f"₱{current_prediction:,.2f}"

    # Base line about current forecast
    base_line = (
        f"Before adjustment, your predicted budget for {category_label} this month was {current_str}. "
    )

    if confidence >= 0.85:
        return (
            f"{base_line}You spent {actual_str} last month, which was close to our forecast of {previous_str} "
            f"(error: {error_percent:.1f}%). Since your spending matched well, we kept this month's forecast at {adjusted_str}."
        )
    elif confidence >= 0.6:
        if adjustment_diff > 0:
            return (
                f"{base_line}You spent {actual_str} on {category_label} last month vs. a forecast of {previous_str}, exceeding it by {error_percent:.1f}%. "
                f"We increased your budget by {adjustment_percent:.1f}% to {adjusted_str}."
            )
        elif adjustment_diff < 0:
            return (
                f"{base_line}Your spending on {category_label} was lower than predicted — {actual_str} vs. {previous_str} "
                f"(error: {error_percent:.1f}%). So we reduced your forecast by {adjustment_percent:.1f}% to {adjusted_str}."
            )
        else:
            return (
                f"{base_line}Your spending on {category_label} was close to predicted ({actual_str} vs. {previous_str}). "
                f"We’ve kept this month's budget at {adjusted_str}."
            )
    else:
        if actual > previous_prediction:
            return (
                f"{base_line}Your spending last month on {category_label} was {actual_str}, significantly higher than the forecast of {previous_str} "
                f"(off by {error_percent:.1f}%). So we adjusted your budget upward to {adjusted_str} — closer to your actual behavior."
            )
        elif actual < previous_prediction:
            return (
                f"{base_line}You spent {actual_str} on {category_label}, much less than our forecast of {previous_str} "
                f"(off by {error_percent:.1f}%). We've reduced this month’s forecast to {adjusted_str} to better align with your actual spending."
            )
        else:
            return (
                f"{base_line}Your {category_label} spending was quite different from our model's prediction. "
                f"We used your actual amount of {actual_str} to adjust this month's forecast to {adjusted_str}."
            )


def budget_adjustment(data_json, previous_forecast=None, current_forecast=None):
    import pandas as pd

    try:
        # 1. Convert input expense data into a DataFrame
        df = pd.DataFrame(data_json)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)

        # 2. Filter only expenses
        df = df[df['type'] == 'Expense']

        # 3. Get the most recent previous month
        latest_date = df['date'].max()
        previous_month = (latest_date - pd.DateOffset(months=1)).month
        previous_month_year = (latest_date - pd.DateOffset(months=1)).year
        prev_month_df = df[(df['date'].dt.month == previous_month + 1) & 
                           (df['date'].dt.year == previous_month_year)]
        
        print("ACTUAL EXPENSE FOR MONTH OF ", previous_month + 1)

        # 4. Group by category and sum the totalAmount
        grouped = prev_month_df.groupby('category')['totalAmount'].sum().reset_index()

        print("GROUPED DATASET: ", grouped) 

        # 5. Define mapping between forecast keys and actual categories
        category_map = {
            "Living_Expenses": "living_expenses",
            "Food_and_Dining_Expenses": "food_and_dining_expenses",
            "Transportation_Expenses": "transportation_expenses",
            "Leisure_and_Entertainment_Expenses": "leisure_and_entertainment_expenses",
            "Academic_Expenses": "academic_expenses"
        }

        category_actual = {
            "Living_Expenses": "Living",
            "Food_and_Dining_Expenses": "Food and Dining",
            "Transportation_Expenses": "Transportation",
            "Leisure_and_Entertainment_Expenses": "Leisure and Entertainment",
            "Academic_Expenses": "Academic"
        }

        # 6. Create a dictionary of actuals for easy access
        actuals = dict(zip(grouped['category'], grouped['totalAmount']))

        print("Actuals: ", actuals)

        confidence_scores = {}
        adjusted_predictions = {}
        personalized_messages = {}


        # 7. Calculate confidence and adjusted prediction
        for category, forecast_key in category_map.items():
            actual = actuals.get(category_actual[category], 0)
            previous_prediction = getattr(previous_forecast, forecast_key, 0)
            current_prediction = current_forecast.get(category, 0)

            if previous_prediction == 0:
                confidence = 0
            else:
                confidence = max(0, 1 - (abs(actual - previous_prediction) / previous_prediction))

            adjusted = (confidence * current_prediction) + ((1 - confidence) * actual)

            confidence_scores[category] = confidence
            adjusted_predictions[category] = adjusted

            # 🆕 Generate the message
            message = generate_adjustment_message(
                category_label=category_actual[category],
                actual=actual,
                previous_prediction=previous_prediction,
                current_prediction=current_prediction,
                adjusted=adjusted,
                confidence=confidence
            )

            personalized_messages[category] = message


        print(confidence_scores)
        print(adjusted_predictions)

        return {
            "success": True,
            "confidence_scores": confidence_scores,
            "adjusted_predictions": adjusted_predictions,
            "messages": personalized_messages
        }


    except Exception as e:
        print("Adjusting error:", e)
        return {
            "success": False
        }

