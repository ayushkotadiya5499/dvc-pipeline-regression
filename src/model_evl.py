import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data
def load_data(test_path):
    return pd.read_csv(test_path)

# Drop Unnecessary Columns
def drop_column(test_df, name):
    return test_df.drop(columns=[name], axis=1)

# Split X and Y
def x_y_values(test_df, target_column):
    x_test = test_df.drop(columns=[target_column], axis=1)
    y_test = test_df[target_column]
    return x_test, y_test

# Define Paths
test_path = './data/processed/test_pre.csv'
model_path = 'model.pkl'
metrics_path = 'metrics.json'

# Main Function
def main():
    test_df = load_data(test_path)

    # Drop Unnamed Column if Exists
    test_df = drop_column(test_df, 'Unnamed: 0')

    # Extract X and Y
    x_test, y_test = x_y_values(test_df, 'tip')

    # Load Model
    model = joblib.load(model_path)

    # Make Predictions
    y_pred = model.predict(x_test)

    # Compute Metrics
    metrics = {
        'mean_squared_error': mean_squared_error(y_test, y_pred),
        'mean_absolute_error': mean_absolute_error(y_test, y_pred),
        
    }

    # Save Metrics as JSON
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=3)

    print(f"✅ Metrics saved to {metrics_path}")

# Run Main
if __name__ == '__main__':
    main()
