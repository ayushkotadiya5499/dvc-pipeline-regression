import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Load Data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Split X and Y
def x_y_values(train_df, test_df,name):
    x_train = train_df.drop(columns=[name], axis=1)  # Fixed axis
    y_train = train_df[[name]]  # Keep as DataFrame for MinMaxScaler
    x_test = test_df.drop(columns=[name], axis=1)
    y_test = test_df[[name]]
    return x_train, y_train, x_test, y_test

# Processing on y_train & y_test
def processing_on_y(y_train, y_test):
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train)  # Fit on train
    y_test = scaler.transform(y_test)  # Only transform test
    return pd.DataFrame(y_train, columns=['tip']), pd.DataFrame(y_test, columns=['tip'])

# ColumnTransformer for x_train & x_test
categorical_features = ['gender', 'smoker', 'day', 'time']
numerical_features = ['total_bill', 'size']

compus = ColumnTransformer(transformers=[
    ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
    ('range', MinMaxScaler(), numerical_features)
], remainder='passthrough')  # Pass other columns through

# Processing on X
def processing_on_train(compus, x_train, x_test):
    x_train_array = compus.fit_transform(x_train)
    x_test_array = compus.transform(x_test)  # Use `transform()`, NOT `fit_transform()`

    # Get new column names after transformation
    one_hot_columns = compus.named_transformers_['one_hot'].get_feature_names_out(categorical_features)
    transformed_columns = np.concatenate((one_hot_columns, numerical_features))

    # **Fix: Handle Passthrough Columns**
    passthrough_columns = [col for col in x_train.columns if col not in categorical_features + numerical_features]
    final_columns = np.concatenate((transformed_columns, passthrough_columns))  # Include all columns

    # Create DataFrames with proper column names
    x_train_df = pd.DataFrame(x_train_array, columns=final_columns)
    x_test_df = pd.DataFrame(x_test_array, columns=final_columns)

    return x_train_df, x_test_df

# Combine X and Y
def combine_data(x_train, y_train, x_test, y_test):
    train_pre_data = pd.concat([x_train, y_train], axis=1)
    test_pre_data = pd.concat([x_test, y_test], axis=1)
    return train_pre_data, test_pre_data

# Save Data
def save_data(train_pre_data, test_pre_data, data_path):
    os.makedirs(data_path, exist_ok=True)  # Use exist_ok=True to prevent errors
    train_pre_data.to_csv(os.path.join(data_path, 'train_pre.csv'), index=False)
    test_pre_data.to_csv(os.path.join(data_path, 'test_pre.csv'), index=False)

# Paths
train_path = 'data/raw/train.csv'
test_path = 'data/raw/test.csv'

# Main Function
def main():
    train_df, test_df = load_data(train_path, test_path)
    x_train, y_train, x_test, y_test = x_y_values(train_df, test_df,'tip')
    y_train, y_test = processing_on_y(y_train, y_test)
    x_train, x_test = processing_on_train(compus, x_train, x_test)
    train_pre_data, test_pre_data = combine_data(x_train, y_train, x_test, y_test)
    save_data(train_pre_data, test_pre_data, 'data/processed')
    print("✅ Preprocessing completed! Files saved in 'data/processed'.")

if __name__ == '__main__':
    main()
