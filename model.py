import pandas as pd
from prophet import Prophet
import pickle
import os

def train_and_save_model(data_path='index_1.csv', model_save_path='prophet_model.pkl'):
    """
    Loads data, preprocesses it, trains a Prophet model, and saves the model.

    Args:
        data_path (str): Path to the input CSV dataset.
        model_save_path (str): Path where the trained model will be saved.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset '{data_path}' not found. Please ensure it's in the same directory.")
        return

    # Convert 'datetime' column to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Aggregate data by date and sum the 'money' column
    # This prepares the data into 'ds' (datestamp) and 'y' (value) format required by Prophet
    df_prophet = df.groupby(df['datetime'].dt.date)['money'].sum().reset_index()

    # Rename columns to 'ds' and 'y'
    df_prophet.columns = ['ds', 'y']

    # Ensure 'ds' is in datetime format after aggregation
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    print("Data preprocessed successfully. First 5 rows for Prophet:")
    print(df_prophet.head())

    # Initialize and train the Prophet model
    print("Training Prophet model...")
    m = Prophet()
    m.fit(df_prophet)
    print("Prophet model trained successfully.")

    # Save the trained model using pickle
    with open(model_save_path, 'wb') as f:
        pickle.dump(m, f)
    print(f"Prophet model saved as '{model_save_path}'.")

if __name__ == '__main__':
    # Ensure index_1.csv is in the same directory as model.py
    train_and_save_model()
