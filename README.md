# siNetwork

Fits hourly temperature from the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality) as a sum of learned sinusoids over day-of-year and hour of day:

    T(day, hour) = Σ a_day·sin(b_day·day + c_day) + Σ a_hour·sin(b_hour·hour + c_hour)

The amplitudes, frequencies and phases are output by an MLP fed a constant input.

## Data

The dataset (`AirQualityUCI.csv`, semicolon-separated hourly readings from March 2004 to April 2005) is not tracked in git. Download it into `data/`:

    mkdir -p data
    curl -L -o data/air.zip 'https://archive.ics.uci.edu/static/public/360/air+quality.zip'
    unzip -o data/air.zip AirQualityUCI.csv -d data && rm data/air.zip

## Usage

Requires [uv](https://docs.astral.sh/uv/).

    uv sync
    uv run train --epochs 500

Options: `--data` (CSV path, default `data/AirQualityUCI.csv`), `--epochs` (default 500), `--plot-dir` (default `plots`), `--seed` (default 0).

Every 10 epochs the loss is printed and a plot of the fitted waves is saved to `plots/`.
