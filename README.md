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

Data files are downloaded into `data/` (not tracked in git) the first time a dataset is used.

## Usage

Requires [uv](https://docs.astral.sh/uv/).

    uv sync
    uv run train --dataset airquality --epochs 500

Options:

- `--dataset` — one of the datasets above (default `airquality`)
- `--components` — sinusoids per period, e.g. `--components day:5 week:3 year:4`. A period is `day`, `week`, `year` or a number of days (`0.5:2` for a 12-hour cycle). Defaults to the dataset's.
- `--epochs` (default 500), `--batch-size` (default 32), `--seed` (default 0)
- `--test-fraction` — final fraction of the time range held out for testing (default 0.2)
- `--data-dir` (default `data`), `--plot-dir` (default `plots`)

The target is standardized for training. Every 10 epochs the loss and the train/test RMSE (in the target's units) are printed and a plot of the fitted waves is saved to `plots/<dataset>/`.

For the larger datasets (`jena`, `power`) a bigger batch, e.g. `--batch-size 256`, keeps epochs short.

## Using your own data

Any CSV with a timestamp column and a numeric target column:

    uv run train --csv my_data.csv --time-col timestamp --target-col load --components day:5 week:3

`--sep` sets the separator (default `,`). Timestamps are parsed with `pandas.to_datetime`, so ISO format (`2024-01-31 13:00`) is safest.

To add a dataset permanently, register a reader in [sinetwork/datasets.py](sinetwork/datasets.py). It takes the data directory and returns a DataFrame with a `datetime` column and a numeric `y` column:

```python
@register('mydata', url='https://example.com/mydata.zip', files=('mydata.csv',),
          label='Load (kW)', components={'day': 6, 'week': 3, 'year': 4})
def read_mydata(data_dir):
    df = pd.read_csv(data_dir / 'mydata.csv', parse_dates=['timestamp'])
    return df.rename(columns={'timestamp': 'datetime', 'load': 'y'})
```

`files` are extracted from the zip at `url` into `data/` if missing (`url` is optional for files you place there yourself). Missing values in `y` are dropped. The dataset then becomes available as `--dataset mydata`.
