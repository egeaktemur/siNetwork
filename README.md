# siNetwork

Fits a time series as a learned sum of sinusoids over periodic components (day, week, year, ...) plus a linear trend:

    y(t) = offset + slope·t/span + Σ_components Σ_k a_k·sin(b_k·t + c_k)

`t` is time in days. A component with period P and n sinusoids starts its frequencies at the harmonics 2πk/P (k = 1..n) and learns a deviation from them. The amplitudes, frequencies, phases, offset and slope are output by an MLP fed a constant input.

## Datasets

| Name | Target | Span | Default components |
|---|---|---|---|
| `airquality` | Temperature, [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) | 2004–2005, hourly | `day:5 year:5` |
| `jena` | Temperature, [Jena Climate](https://www.bgc-jena.mpg.de/wetter/) (Max Planck Institute) | 2009–2016, 10 min → hourly | `day:5 year:5` |
| `bike` | Bike rentals per hour, [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 2011–2012, hourly | `day:6 week:3 year:3` |
| `power` | Household active power, [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | 2006–2010, 1 min → hourly | `day:6 week:3 year:4` |
| `metro` | Traffic volume on I-94, [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | 2012–2018, hourly | `day:6 week:3 year:3` |
| `beijing-temp` | Temperature, [UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) | 2010–2014, hourly | `day:5 year:5` |
| `beijing-pm25` | PM2.5 concentration, same source | 2010–2014, hourly | `day:5 week:3 year:4` |

`beijing-pm25` is mostly driven by weather episodes rather than cycles, so expect a poor fit; it is there as a hard case.

## Folders

Each dataset is one CSV file, `data/<dataset>/<dataset>.csv`, downloaded the first time the dataset is used (the source's own separator and columns are kept; gzipped files are decompressed). Plots go into one folder per dataset, with a subfolder per component setting so runs with different components don't overwrite each other. Both folders are not tracked in git.

    data/
      airquality/airquality.csv
      beijing/beijing.csv                            # shared by beijing-temp and beijing-pm25
      bike/bike.csv
      jena/jena.csv
      metro/metro.csv
      power/power.csv
    plots/
      bike/
        data.png                                     # the raw series
        day6-week3-year3/                            # one folder per --components setting
          fit_epoch_0010.png                         # fitted waves, every 10 epochs
          fit_epoch_0020.png

## Usage

Requires [uv](https://docs.astral.sh/uv/).

    uv sync
    uv run train --dataset airquality --epochs 500

Options:

- `--dataset` — one of the datasets above (default `airquality`)
- `--components` — sinusoids per period, e.g. `--components day:5 week:3 year:4`. A period is `day`, `week`, `year` or a number of days (`0.5:2` for a 12-hour cycle). Defaults to the dataset's.
- `--epochs` (default 500), `--batch-size` (default 256), `--seed` (default 0)
- `--test-fraction` — final fraction of the time range held out for testing (default 0.2)
- `--data-dir` (default `data`), `--plot-dir` (default `plots`)

The target is standardized for training. Every 10 epochs the loss and the train/test RMSE (in the target's units) are printed and a plot of the fitted waves is saved to `plots/<dataset>/<components>/fit_epoch_NNNN.png`.

## Using your own data

Any CSV with a timestamp column and a numeric target column:

    uv run train --csv my_data.csv --time-col timestamp --target-col load --components day:5 week:3

`--sep` sets the separator (default `,`). Timestamps are parsed with `pandas.to_datetime`, so ISO format (`2024-01-31 13:00`) is safest.

To add a dataset permanently, register a reader in [sinetwork/datasets.py](sinetwork/datasets.py). It takes the path of the dataset's CSV and returns a DataFrame with a `datetime` column and a numeric `y` column:

```python
@register('mydata', url='https://example.com/mydata.zip', member='raw/mydata.csv',
          label='Load (kW)', components={'day': 6, 'week': 3, 'year': 4})
def read_mydata(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.rename(columns={'timestamp': 'datetime', 'load': 'y'})
```

If `data/mydata/mydata.csv` is missing, `member` is extracted to it from the zip at `url` (`url` and `member` are optional if you place the file there yourself). Datasets read from the same file can share it with `folder='...'`. Missing values in `y` are dropped. The dataset then becomes available as `--dataset mydata`.
