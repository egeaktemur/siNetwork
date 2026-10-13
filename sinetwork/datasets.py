"""Dataset registry.

A dataset is a reader function that takes its folder (data/<folder>) and
returns a DataFrame with a `datetime` column and a float target column `y`.
Register it with `@register(...)`, giving the archive to download its files
from and the periodic components it is fit with by default:

    @register('mydata', url='https://example.com/mydata.zip', files=('mydata.csv',),
              label='Load (kW)', components={'day': 6, 'week': 3, 'year': 4})
    def read_mydata(folder):
        df = pd.read_csv(folder / 'mydata.csv', parse_dates=['timestamp'])
        return df.rename(columns={'timestamp': 'datetime', 'load': 'y'})

Any other CSV can be used without registering it, see `csv_dataset`.
"""
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    name: str
    read: Callable[[Path], pd.DataFrame]  # folder -> DataFrame with `datetime` and `y`
    files: tuple  # files in the folder that `read` needs, as named in the archive
    label: str  # name of the target, used in plots
    components: dict  # default {period: sinusoid count}, see model.PERIODS
    url: Optional[str] = None  # zip archive containing `files`
    folder: Optional[str] = None  # subfolder of the data directory, defaults to `name`

    def load(self, data_dir='data'):
        folder = Path(data_dir) / (self.folder or self.name)
        missing = [f for f in self.files if not (folder / f).exists()]
        if missing:
            if self.url is None:
                raise FileNotFoundError(f'{self.name}: missing {missing} in {folder}')
            download(self.url, missing, folder)
        df = self.read(folder)
        df = df[['datetime', 'y']].dropna()
        return df.astype({'y': float}).sort_values('datetime').reset_index(drop=True)


DATASETS = {}


def register(name, **kwargs):
    def decorator(read):
        DATASETS[name] = Dataset(name=name, read=read, **kwargs)
        return read
    return decorator


def download(url, members, folder):
    """Download the zip archive at `url` and extract `members` into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}')
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)
        with zipfile.ZipFile(tmp) as archive:
            for member in members:
                with archive.open(member) as src, open(folder / member, 'wb') as dst:
                    shutil.copyfileobj(src, dst)


def csv_dataset(path, time_col, target_col, components, sep=','):
    """A dataset read from an arbitrary CSV file with a timestamp and a target column."""
    path = Path(path)

    def read(folder):
        df = pd.read_csv(path, sep=sep, usecols=[time_col, target_col])
        df['datetime'] = pd.to_datetime(df[time_col])
        df['y'] = pd.to_numeric(df[target_col], errors='coerce')
        return df

    return Dataset(name=path.stem, read=read, files=(), label=target_col, components=components)


@register('airquality', url='https://archive.ics.uci.edu/static/public/360/air+quality.zip',
          files=('AirQualityUCI.csv',), label='Temperature (°C)', components={'day': 5, 'year': 5})
def read_airquality(folder):
    df = pd.read_csv(folder / 'AirQualityUCI.csv', sep=';', decimal=',')
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Date', 'Time'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df['y'] = df['T']
    return df[df['y'] != -200]  # -200 marks a missing reading


@register('jena', url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
          files=('jena_climate_2009_2016.csv',), label='Temperature (°C)', components={'day': 5, 'year': 5})
def read_jena(folder):
    df = pd.read_csv(folder / 'jena_climate_2009_2016.csv', usecols=['Date Time', 'T (degC)'])
    df['datetime'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    # 10-minute readings, averaged to hourly
    return df.set_index('datetime')['T (degC)'].resample('h').mean().rename('y').reset_index()


@register('bike', url='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
          files=('hour.csv',), label='Bike rentals per hour', components={'day': 6, 'week': 3, 'year': 3})
def read_bike(folder):
    df = pd.read_csv(folder / 'hour.csv')
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df['y'] = df['cnt']
    return df


@register('power', url='https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',
          files=('household_power_consumption.txt',), label='Global active power (kW)',
          components={'day': 6, 'week': 3, 'year': 4})
def read_power(folder):
    df = pd.read_csv(folder / 'household_power_consumption.txt', sep=';', na_values='?',
                     usecols=['Date', 'Time', 'Global_active_power'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    # 1-minute readings, averaged to hourly
    return df.set_index('datetime')['Global_active_power'].resample('h').mean().rename('y').reset_index()
