import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def plot_waves(model, df, out_path):
    device = next(model.parameters()).device

    # Generate an array of hours and days
    all_datetimes = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='h')
    all_days = all_datetimes.dayofyear.values.reshape(-1, 1)
    all_hours = all_datetimes.hour.values.reshape(-1, 1)

    hours = np.arange(24).reshape(-1, 1)
    days = np.arange(1, 366).reshape(-1, 1)  # Assuming 365 days for simplicity

    # Convert numpy arrays to PyTorch tensors
    hours_tensor = torch.tensor(hours).float().to(device)
    days_tensor = torch.tensor(days).float().to(device)

    # Calculate sinusoidal values
    with torch.no_grad():
        a_day, b_day, c_day, a_hour, b_hour, c_hour = model.get_sinusoidal_wave()
        sinusoidal_values_hour = (a_hour * torch.sin(b_hour * hours_tensor + c_hour)).cpu().numpy()
        sinusoidal_values_day = (a_day * torch.sin(b_day * days_tensor + c_day)).cpu().numpy()
        sinusoidal_sum_day = (a_day * torch.sin(b_day * torch.tensor(all_days).to(device) + c_day)).sum(1).cpu().numpy()
        sinusoidal_sum_hour = (a_hour * torch.sin(b_hour * torch.tensor(all_hours).to(device) + c_hour)).sum(1).cpu().numpy()
        total_sinusoidal = sinusoidal_sum_day + sinusoidal_sum_hour

    fig = plt.figure(figsize=(40, 20))
    grid = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    # Temperature and sinusoidal values spanning the top row
    ax_big = fig.add_subplot(grid[0, :])
    ax_big.scatter(df['datetime'], df['T'], label='Temperature (T)', s=10, color='blue')
    ax_big.plot(all_datetimes, total_sinusoidal, label='Total Sinusoidal Wave', color='orange')
    ax_big.plot(all_datetimes, sinusoidal_sum_day, label='Day Sinusoidal Wave', color='red')
    ax_big.set_xlabel('Datetime')
    ax_big.set_ylabel('Value')
    ax_big.set_title('Temperature and Sinusoidal Waves Over Time')
    ax_big.legend()
    ax_big.grid(True)

    # Hourly sinusoidal wave plots
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(hours.flatten(), sinusoidal_values_hour.sum(axis=1), label='Sum of Hourly Sinusoidal Waves', color='green')
    ax.set_title('Sum of Hourly Sinusoidal Waves')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Sinusoidal Value')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(hours.flatten(), sinusoidal_values_hour, color='blue')
    ax.set_title('Hourly Sinusoidal Waves')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Sinusoidal Value')
    ax.grid(True)

    # Daily sinusoidal wave plots
    ax = fig.add_subplot(grid[2, 0])
    ax.plot(days.flatten(), sinusoidal_values_day.sum(axis=1), label='Sum of Daily Sinusoidal Waves', color='purple')
    ax.set_title('Sum of Daily Sinusoidal Waves')
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Sinusoidal Value')
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(grid[2, 1])
    ax.plot(days.flatten(), sinusoidal_values_day, color='red')
    ax.set_title('Daily Sinusoidal Waves')
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Sinusoidal Value')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_correlation(df, out_path):
    fig = plt.figure()
    sns.heatmap(df.corr(numeric_only=True))
    fig.savefig(out_path)
    plt.close(fig)


def plot_temperature(df, out_path):
    # Plot temperature over time
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['T'], label='Temperature (T)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (T)')
    plt.title('Temperature over Time')
    plt.legend()
    plt.grid(True)
    fig.savefig(out_path)
    plt.close(fig)
