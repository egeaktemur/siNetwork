"""Dataset registry.

A dataset is one CSV file, data/<folder>/<folder>.csv, and a reader function
that takes its path and returns a DataFrame with a `datetime` column and a
float target column `y`. Register it with `@register(...)`, giving the zip
archive and member to download the file from and the periodic components it is
fit with by default:

    @register('mydata', url='https://example.com/mydata.zip', member='raw/mydata.csv',
              label='Load (kW)', components={'day': 6, 'week': 3, 'year': 4})
    def read_mydata(path):
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return df.rename(columns={'timestamp': 'datetime', 'load': 'y'})

Any other CSV can be used without registering it, see `csv_dataset`.
"""
import gzip
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
    read: Callable[[Path], pd.DataFrame]  # CSV path -> DataFrame with `datetime` and `y`
    label: str  # name of the target, used in plots
    components: dict  # default {period: sinusoid count}, see model.PERIODS
    url: Optional[str] = None  # zip archive containing the CSV
    member: Optional[str] = None  # the CSV's name in the archive; a .gz member is decompressed
    folder: Optional[str] = None  # stored as data/<folder>/<folder>.csv, folder defaults to `name`
    path: Optional[Path] = None  # explicit CSV path, instead of the data directory

    def load(self, data_dir='data'):
        folder = self.folder or self.name
        path = self.path or Path(data_dir) / folder / f'{folder}.csv'
        if not path.exists():
            if self.url is None:
                raise FileNotFoundError(f'{self.name}: missing {path}')
            download(self.url, self.member, path)
        df = self.read(path)
        df = df[['datetime', 'y']].dropna()
        return df.astype({'y': float}).sort_values('datetime').reset_index(drop=True)


DATASETS = {}


def register(name, **kwargs):
    def decorator(read):
        DATASETS[name] = Dataset(name=name, read=read, **kwargs)
        return read
    return decorator


def download(url, member, path):
    """Download the zip archive at `url` and extract `member` to `path`, decompressing it if gzipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}')
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)
        with zipfile.ZipFile(tmp) as archive, archive.open(member) as src, open(path, 'wb') as dst:
            shutil.copyfileobj(gzip.open(src) if member.endswith('.gz') else src, dst)


def csv_dataset(path, time_col, target_col, components, sep=','):
    """A dataset read from an arbitrary CSV file with a timestamp and a target column."""
    path = Path(path)

    def read(path):
        df = pd.read_csv(path, sep=sep, usecols=[time_col, target_col])
        df['datetime'] = pd.to_datetime(df[time_col])
        df['y'] = pd.to_numeric(df[target_col], errors='coerce')
        return df

    return Dataset(name=path.stem, read=read, label=target_col, components=components, path=path)


@register('airquality', url='https://archive.ics.uci.edu/static/public/360/air+quality.zip',
          member='AirQualityUCI.csv', label='Temperature (°C)', components={'day': 5, 'year': 5})
def read_airquality(path):
    df = pd.read_csv(path, sep=';', decimal=',')
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Date', 'Time'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
    df['y'] = df['T']
    return df[df['y'] != -200]  # -200 marks a missing reading


@register('jena', url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
          member='jena_climate_2009_2016.csv', label='Temperature (°C)', components={'day': 5, 'year': 5})
def read_jena(path):
    df = pd.read_csv(path, usecols=['Date Time', 'T (degC)'])
    df['datetime'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    # 10-minute readings, averaged to hourly
    return df.set_index('datetime')['T (degC)'].resample('h').mean().rename('y').reset_index()


@register('bike', url='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
          member='hour.csv', label='Bike rentals per hour', components={'day': 6, 'week': 3, 'year': 3})
def read_bike(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df['y'] = df['cnt']
    return df


@register('power', url='https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',
          member='household_power_consumption.txt', label='Global active power (kW)',
          components={'day': 6, 'week': 3, 'year': 4})
def read_power(path):
    df = pd.read_csv(path, sep=';', na_values='?', usecols=['Date', 'Time', 'Global_active_power'])
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    # 1-minute readings, averaged to hourly
    return df.set_index('datetime')['Global_active_power'].resample('h').mean().rename('y').reset_index()


@register('metro', url='https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip',
          member='Metro_Interstate_Traffic_Volume.csv.gz', label='Traffic volume (cars per hour)',
          components={'day': 6, 'week': 3, 'year': 3})
def read_metro(path):
    df = pd.read_csv(path, usecols=['date_time', 'traffic_volume'])
    df['datetime'] = pd.to_datetime(df['date_time'])
    # Hours with several weather conditions appear once per condition
    return df.groupby('datetime', as_index=False)['traffic_volume'].first().rename(columns={'traffic_volume': 'y'})


def read_beijing(path, column):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['y'] = df[column]
    return df


BEIJING = dict(url='https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip',
               member='PRSA_data_2010.1.1-2014.12.31.csv', folder='beijing')


@register('beijing-temp', label='Temperature (°C)', components={'day': 5, 'year': 5}, **BEIJING)
def read_beijing_temp(path):
    return read_beijing(path, 'TEMP')


@register('beijing-pm25', label='PM2.5 (µg/m³)', components={'day': 5, 'week': 3, 'year': 4}, **BEIJING)
def read_beijing_pm25(path):
    return read_beijing(path, 'pm2.5')
