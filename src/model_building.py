import pandas as pd
from xgboost import XGBRegressor
import joblib

# Load Data
def load_data(train_path):
    return pd.read_csv(train_path)

# Drop Unnecessary Columns
def drop_column(train_df, name):
    return train_df.drop(columns=[name], axis=1)  # Fixed axis

# Split X and Y
def x_y_values(train_df, target_column):
    x_train = train_df.drop(columns=[target_column], axis=1)
    y_train = train_df[target_column]
    return x_train, y_train

# Train and Save Model
def train_and_save_model(model_obj, x_train, y_train, model_path):
    model_obj.fit(x_train, y_train)  # Train model
    joblib.dump(model_obj, model_path)  # Save model
    print(f"✅ Model saved at {model_path}")

# Define Paths
train_path = './data/processed/train_pre.csv'
model_path = 'model.pkl'

# Main Function
def main():
    train_df = load_data(train_path)
    train_df = drop_column(train_df, 'Unnamed: 0')
    x_train, y_train = x_y_values(train_df, 'tip')
    model_obj = XGBRegressor()
    train_and_save_model(model_obj, x_train, y_train, model_path)

if __name__ == '__main__':
    main()
