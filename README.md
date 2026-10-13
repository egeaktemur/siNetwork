# siNetwork

Fits a time series as a learned sum of sinusoids over periodic components (day, week, year, ...) plus a linear trend:

    y(t) = offset + slope·t/span + Σ_components Σ_k a_k·sin(b_k·t + c_k)

`t` is time in days. A component with period P and n sinusoids has its frequencies fixed at the harmonics 2πk/P (k = 1..n). The amplitudes, phases, offset and slope are output by an MLP fed a constant input.

Each sinusoid has an amplitude and a phase, so n sinusoids give a component 2n free numbers: it can follow a shape with about one value per P/2n. The default, `week:84 year:2`, is:

- `week:84`: 168 numbers for the 168 hours of a week, so any hourly weekly profile (Monday mornings can differ from Sunday mornings). The daily cycle is included, since the 7th weekly harmonic is one cycle per day.
- `year:2`: 4 numbers, about one per 3 months: a smooth seasonal swing. More yearly detail (`year:6` for monthly) needs several years of training data; with `airquality`'s single year it extrapolates wildly.

## Datasets

| Name | Target | Span |
|---|---|---|
| `airquality` | Temperature, [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) | 2004–2005, hourly |
| `jena` | Temperature, [Jena Climate](https://www.bgc-jena.mpg.de/wetter/) (Max Planck Institute) | 2009–2016, 10 min → hourly |
| `bike` | Bike rentals per hour, [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 2011–2012, hourly |
| `power` | Household active power, [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | 2006–2010, 1 min → hourly |
| `metro` | Traffic volume on I-94, [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | 2012–2018, hourly |
| `beijing-temp` | Temperature, [UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) | 2010–2014, hourly |
| `beijing-pm25` | PM2.5 concentration, same source | 2010–2014, hourly |

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
        week84-year2/                                # one folder per --components setting
          fit_epoch_0050.png                         # data and fitted waves over time, every --plot-every epochs
          components_epoch_0050.png                  # each component's sinusoids over one period
          metrics.csv                                # loss and train/test RMSE every 10 epochs

## Usage

Requires [uv](https://docs.astral.sh/uv/).

    uv sync
    uv run train --dataset airquality --epochs 500

Options:

- `--dataset` — one of the datasets above (default `airquality`)
- `--components` — sinusoids per period (default `week:84 year:2`, see above). A period is `day`, `week`, `year` or a number of days (`0.5:2` for a 12-hour cycle).
- `--epochs` (default 500), `--batch-size` (default 256), `--seed` (default 0)
- `--test-fraction` — final fraction of the time range held out for testing (default 0.2)
- `--data-dir` (default `data`), `--plot-dir` (default `plots`), `--plot-every` (default 50)

The target is standardized for training. Every 10 epochs the loss and the train/test RMSE (in the target's units) are printed and appended to `plots/<dataset>/<components>/metrics.csv`. Every `--plot-every` epochs and after the last one, two plots are saved to `plots/<dataset>/<components>/`: `fit_epoch_NNNN.png` (data and fitted waves over time) and `components_epoch_NNNN.png` (each component's sinusoids over one period).

## Using your own data

Any CSV with a timestamp column and a numeric target column:

    uv run train --csv my_data.csv --time-col timestamp --target-col load

`--sep` sets the separator (default `,`). Timestamps are parsed with `pandas.to_datetime`, so ISO format (`2024-01-31 13:00`) is safest.

To add a dataset permanently, register a reader in [sinetwork/datasets.py](sinetwork/datasets.py). It takes the path of the dataset's CSV and returns a DataFrame with a `datetime` column and a numeric `y` column:

```python
@register('mydata', url='https://example.com/mydata.zip', member='raw/mydata.csv',
          label='Load (kW)')
def read_mydata(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df.rename(columns={'timestamp': 'datetime', 'load': 'y'})
```

If `data/mydata/mydata.csv` is missing, `member` is extracted to it from the zip at `url` (`url` and `member` are optional if you place the file there yourself). Datasets read from the same file can share it with `folder='...'`. Missing values in `y` are dropped. The dataset then becomes available as `--dataset mydata`.
