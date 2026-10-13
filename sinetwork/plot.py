import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sinetwork.model import period_days


def plot_fit(model, train_df, test_df, scaling, label, out_path):
    """Data and fitted waves over the whole time range."""
    device = next(model.parameters()).device
    slowest = max(model.components, key=period_days)

    # Hourly grid over the whole data range
    all_datetimes = pd.date_range(start=train_df['datetime'].min(), end=test_df['datetime'].max(), freq='h')
    all_t = torch.tensor(scaling.days(all_datetimes), dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, offset, slope = model.get_params()
        total = scaling.unscale(model(all_t).cpu().numpy())
        slow = offset + slope * all_t / model.span + model.waves(all_t)[slowest].sum(1)
        slow = scaling.unscale(slow.cpu().numpy())

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.scatter(train_df['datetime'], train_df['y'], label=f'{label} (train)', s=4, color='blue')
    ax.scatter(test_df['datetime'], test_df['y'], label=f'{label} (test)', s=4, color='gray')
    ax.plot(all_datetimes, total, label='Total Sinusoidal Wave', color='orange')
    ax.plot(all_datetimes, slow, label=f'Trend + {slowest.capitalize()} Sinusoidal Wave', color='red')
    ax.set_xlabel('Datetime')
    ax.set_ylabel(label)
    ax.set_title(f'{label} and Sinusoidal Waves Over Time')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_components(model, scaling, label, out_path):
    """One row per component over its first period: the sum of its sinusoids and each sinusoid."""
    device = next(model.parameters()).device
    components = model.components
    fig = plt.figure(figsize=(16, 4 * len(components)))
    grid = fig.add_gridspec(len(components), 2)

    for row, period in enumerate(components):
        days = np.linspace(0, period_days(period), 500)
        with torch.no_grad():
            waves = model.waves(torch.tensor(days, dtype=torch.float32).to(device))[period]
            waves = (waves * scaling.y_std).cpu().numpy()
        x, xlabel = period_axis(period, days)
        name = period.capitalize()

        ax = fig.add_subplot(grid[row, 0])
        ax.plot(x, waves.sum(axis=1), label=f'Sum of {name} Sinusoidal Waves', color='green')
        ax.set_title(f'Sum of {name} Sinusoidal Waves')
        ax.legend()

        ax2 = fig.add_subplot(grid[row, 1])
        ax2.plot(x, waves, color='blue')
        ax2.set_title(f'{name} Sinusoidal Waves')

        for a in (ax, ax2):
            a.set_xlabel(xlabel)
            a.set_ylabel(label)
            a.grid(True)
            if period == 'week':
                a.set_xticks(range(7), [(scaling.t0 + pd.Timedelta(days=d)).day_name() for d in range(7)])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def period_axis(period, days):
    """x values and axis label for plotting one period of a component."""
    if period == 'day':
        return days * 24, 'Hour of Day'
    if period == 'year':
        return days + 1, 'Day of Year'
    if period_days(period) < 2:
        return days * 24, 'Hours'
    return days, 'Days'


def plot_series(df, label, out_path):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['y'], label=label)
    plt.xlabel('Time')
    plt.ylabel(label)
    plt.title(f'{label} over Time')
    plt.legend()
    plt.grid(True)
    fig.savefig(out_path)
    plt.close(fig)
