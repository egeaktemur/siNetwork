from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset


@dataclass
class Scaling:
    """Maps datetimes to days since t0 and standardizes the target, or divides it by its mean if not centered."""
    t0: pd.Timestamp
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, df, center=True):
        # Start of the first year, so day and year components line up with hour of day and day of year
        t0 = pd.Timestamp(year=df['datetime'].min().year, month=1, day=1)
        if not center:  # keeps the target non-negative, for the Poisson loss
            return cls(t0, 0.0, df['y'].mean())
        return cls(t0, df['y'].mean(), df['y'].std())

    def days(self, datetimes):
        return ((pd.Series(datetimes) - self.t0) / pd.Timedelta(days=1)).to_numpy(float)

    def unscale(self, y):
        return y * self.y_std + self.y_mean


def holiday_of_day(scaling, end, country, subdivision=None):
    """For each day from t0 to `end`, the 1-based index of its holiday (numbered by name) or 0."""
    dates = pd.date_range(scaling.t0, end, freq='D')
    calendar = holidays.country_holidays(country, subdiv=subdivision, years=range(dates[0].year, dates[-1].year + 1))
    names = [calendar.get(date) for date in dates]
    index = {name: i + 1 for i, name in enumerate(sorted({name for name in names if name}))}
    return np.array([index.get(name, 0) for name in names])


def split(df, test_fraction):
    """Chronological split: the last `test_fraction` of the time range is held out."""
    start, end = df['datetime'].min(), df['datetime'].max()
    cutoff = start + (end - start) * (1 - test_fraction)
    return df[df['datetime'] <= cutoff], df[df['datetime'] > cutoff]


def make_loader(df, scaling, device, batch_size=None, shuffle=True):
    t = torch.tensor(scaling.days(df['datetime']), dtype=torch.float32).to(device)
    y = torch.tensor(((df['y'] - scaling.y_mean) / scaling.y_std).values, dtype=torch.float32).to(device)
    dataset = TensorDataset(t, y)
    # Index whole batches at once rather than stacking single samples, which is far faster for tensors on the GPU
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size or len(dataset), drop_last=False), batch_size=None)
