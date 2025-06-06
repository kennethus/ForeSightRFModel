def forecast_expenses(data_json=None, previous_forecast=None, test_size=30):
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        if(data_json):
            # 1. Convert input expense data into a DataFrame
            df = pd.DataFrame(data_json)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date').reset_index(drop=True)

            # 2. Filter only expenses
            df = df[df['type'] == 'Expense']

            # 3. Group by date and sum
            df_daily_expenses = df.groupby('date')['totalAmount'].sum().reset_index()
            df_daily_expenses.columns = ['date', 'amount']
            df_daily_expenses = df_daily_expenses.sort_values(by='date')
            df_daily_expenses.set_index('date', inplace=True)

            print("TRANSACTIONS GROUPED: ", df_daily_expenses)

            # 4. Prepare the test data (actual expenses of the past N days)
            actual_previous_expense = df_daily_expenses[-test_size:]

            print("LAST MONTH TRANSACTIONS: ", actual_previous_expense)

            # 5. Train the model
            model = ExponentialSmoothing(
                df_daily_expenses['amount'],
                trend='add',
                seasonal='add',
                seasonal_periods=30
            )
            model_fit = model.fit()

            # 6. Forecast
            forecast = model_fit.forecast(test_size)

            # Get the last date in your data
            last_date = df_daily_expenses.index[-1]

            # Calculate the first day of the next month
            first_of_next_month = (last_date + pd.offsets.MonthBegin(1)).normalize()

            # Generate future dates starting from the first of next month
            future_dates = pd.date_range(start=first_of_next_month, periods=test_size)

            print("FUTURE DATES:", future_dates)

            # Assign those to forecast
            forecast.index = future_dates
            
            result = {
                "success": True,
                "forecast": forecast.tolist(),
                "dates": forecast.index.strftime('%Y-%m-%dT%H:%M:%S.000+00:00').tolist(),
                "metrics": {
                    "total_forecasted": forecast.sum()
                }
            }

            print("PREVIOUS FORECAST", previous_forecast)
            # 7. Optional evaluation using previous forecast
            if previous_forecast.forecasted:
                
                forecasted_values = previous_forecast.forecasted
                forecasted_dates = pd.to_datetime(previous_forecast.dates).tz_localize(None)
                previous_forecast_series = pd.Series(forecasted_values, index=forecasted_dates)
                actual_previous_expense.index = actual_previous_expense.index.tz_localize(None)

                previous_forecast_series = previous_forecast_series.loc[actual_previous_expense.index]

                mae = mean_absolute_error(actual_previous_expense['amount'], previous_forecast_series)
                rmse = np.sqrt(mean_squared_error(actual_previous_expense['amount'], previous_forecast_series))
                r2 = r2_score(actual_previous_expense['amount'], previous_forecast_series)

                result["metrics"].update({
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                })

            return result
        else: 
            return {
            "success": False,
            "message": "No transactions"
        }

    except Exception as e:
        print("Forecasting error:", e)
        return {
            "success": False,
            "message": e
        }
