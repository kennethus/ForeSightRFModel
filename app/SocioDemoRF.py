import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os

# Load the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "dataset", "Student-Spending-Habits_PreProcessed.csv")
df = pd.read_csv(csv_path)

# Target columns
expense_cols = ["Living_Expenses", "Food_and_Dining_Expenses", 
                "Transportation_Expenses", "Leisure_and_Entertainment_Expenses", "Academic_Expenses"]

# Separate features and targets
X = df.drop(columns=expense_cols)
y = df[expense_cols]

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None, 10],
}

# Perform Randomized Search with MultiOutputRegressor
base_rf = RandomForestRegressor(n_estimators=250, min_samples_leaf=5, min_samples_split=2, max_features=None, max_depth=None, random_state=2)

#Finding best Hyperparameter
# random_search = RandomizedSearchCV(estimator=base_rf,
#                                    param_distributions=param_grid,
#                                    n_iter=20,
#                                    scoring='r2',
#                                    cv=3,
#                                    verbose=2,
#                                    n_jobs=-1,
#                                    random_state=42)

# multioutput_regressor = MultiOutputRegressor(random_search)

multioutput_regressor = MultiOutputRegressor(base_rf)
multioutput_regressor.fit(X_train, Y_train)

# # Print best parameters for each target
# for i, estimator in enumerate(multioutput_regressor.estimators_):
#     print(f"Best parameters for target {i+1}:", estimator.best_params_)

#Save model

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "SocioDemoRFModel.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(multioutput_regressor, f)
print("Model trained and saved successfully")

# # Predict and evaluate
# Y_pred = multioutput_regressor.predict(X_test)

# r2 = r2_score(Y_test, Y_pred)
# mse = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
# mae = mean_absolute_error(Y_test, Y_pred)


# print("\nEvaluation:")
# print("R2 Score:", r2)
# print("Mean Squared Error:", mse)
# print("Mean Absolute Error:", mae)

# # Actual vs predicted
# print("\nActual values:\n", Y_test.head())
# print("\nPredicted values:\n", pd.DataFrame(Y_pred, columns=y.columns).head())

# Feature Importance Plots
# for i, estimator in enumerate(multioutput_regressor.estimators_):
#     best_rf = estimator
#     importances = best_rf.feature_importances_
#     feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
#     top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head()
    
#     plt.figure(figsize=(8, 5))
#     plt.bar(top_features['Feature'], top_features['Importance'], color='skyblue')
#     plt.xticks(rotation=45, ha="right")
#     plt.xlabel("Feature")
#     plt.ylabel("Importance")
#     plt.title(f"Top 5 Features for Target {expense_cols[i]}")
#     plt.tight_layout()
#     plt.show()