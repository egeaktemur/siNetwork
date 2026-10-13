import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ['day_of_year', 'hour', 'day', 'week', 'month', 'week_of_year', 'am_pm']


def load_data(file_path='data/AirQualityUCI.csv'):
    # Read the CSV file
    df = pd.read_csv(file_path, sep=';')

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Convert date and time columns to datetime
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')

    # Extract new columns
    df['day'] = df['datetime'].dt.day
    df['week'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['hour'] = df['datetime'].dt.hour
    df['am_pm'] = df['hour'] < 12

    # Convert 'T' (temperature) column to float (replacing commas with dots)
    df['T'] = df['T'].str.replace(',', '.').astype(float)

    df = df[df['T'] != -200]

    df = df[FEATURES + ['datetime', 'T']]

    return df.dropna()


def make_loader(df, device, batch_size=32):
    input_features = df[FEATURES].astype(float).values
    target = df['T'].values

    # Convert to PyTorch tensors and move to the device
    input_features = torch.tensor(input_features, dtype=torch.float32).to(device)
    target = torch.tensor(target, dtype=torch.float32).to(device)

    # Create a dataset and DataLoader
    dataset = TensorDataset(input_features, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
