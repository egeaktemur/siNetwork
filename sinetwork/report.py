"""Compare the trained variants with each other and with their least-squares optimum.

Reads plots/<dataset>/<run>/summary.json and model.pt written by `train`, prints
a markdown table of test RMSE and saves the README figures to docs/:

    uv run report
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sinetwork.datasets import DATASETS
from sinetwork.model import SinusoidalNetwork

DATASET_ORDER = ['metro', 'bike', 'power', 'jena', 'beijing-temp', 'airquality', 'beijing-pm25']
NON_NEGATIVE = {'metro', 'bike', 'power', 'beijing-pm25'}  # can use the Poisson loss

# Each variant adds to the previous one; datasets with negative values keep the MSE loss in the last
VARIANTS = [
    ('Base', lambda d: 'week84-year2'),
    ('+ holidays', lambda d: 'week84-year2-holidays'),
    ('+ seasonal modulation', lambda d: 'week84-year2-mod1-holidays'),
    ('+ Poisson loss', lambda d: 'week84-year2-mod1-holidays' + ('-poisson' if d in NON_NEGATIVE else '')),
]

# Test RMSE of the first version (learned frequency deviations, amplitude and phase, batches of 256),
# commit b6f01c0, which the current code no longer reproduces
ORIGINAL = {'airquality': 7.52, 'bike': 120.2, 'metro': 519.8, 'power': 0.645, 'beijing-temp': 3.77,
            'beijing-pm25': 93.3, 'jena': 4.51}

# Light chart tokens and the first categorical slots of the reference palette
SURFACE, INK, INK_2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']


def load_summaries(plot_dir):
    return {(d, run): json.loads((plot_dir / d / run / 'summary.json').read_text())
            for d in DATASET_ORDER for _, name in VARIANTS for run in [name(d)]}


def markdown_table(summaries):
    header = ['Dataset', 'First version', 'Least squares (base)'] + [label for label, _ in VARIANTS]
    rows = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for d in DATASET_ORDER:
        network = [summaries[d, name(d)]['network']['test']['rmse'] for _, name in VARIANTS]
        best = min(network)
        cells = [f'`{d}`', fmt(ORIGINAL[d]), fmt(summaries[d, VARIANTS[0][1](d)]['lstsq']['test']['rmse'])]
        for i, value in enumerate(network):
            text = fmt(value)
            if i == len(VARIANTS) - 1 and d not in NON_NEGATIVE:
                text += '¹'
            cells.append(f'**{text}**' if value == best else text)
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def fmt(value):
    return f'{value:.3f}' if value < 10 else f'{value:.1f}' if value < 1000 else f'{value:.0f}'


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_comparison(summaries, out_path):
    """Test RMSE of each variant relative to the least-squares fit of the base model, network vs least squares."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    style(ax)
    width = 0.16
    for g, d in enumerate(DATASET_ORDER):
        reference = summaries[d, VARIANTS[0][1](d)]['lstsq']['test']['rmse']
        ax.bar(g - 2 * width, ORIGINAL[d] / reference, width - 0.02, color='none', edgecolor=MUTED,
               linewidth=1.2, label='First version' if g == 0 else None)
        for i, (label, name) in enumerate(VARIANTS):
            summary = summaries[d, name(d)]
            x = g + (i - 1) * width
            skipped = i == len(VARIANTS) - 1 and d not in NON_NEGATIVE
            ax.bar(x, summary['network']['test']['rmse'] / reference, width - 0.02, color=SERIES[i],
                   alpha=0.35 if skipped else 1, label=label if g == 0 else None)
            ax.plot([x - width / 2 + 0.01, x + width / 2 - 0.01], [summary['lstsq']['test']['rmse'] / reference] * 2,
                    color=INK, linewidth=2, solid_capstyle='butt', label='Least squares, same model' if g == i == 0 else None)
        final = summaries[d, VARIANTS[-1][1](d)]['network']['test']['rmse']
        ax.annotate(f'{100 * (final / ORIGINAL[d] - 1):+.0f}%', (g + 2 * width, final / reference),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9, color=INK_2)
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.set_xticks(range(len(DATASET_ORDER)), DATASET_ORDER, color=INK, fontsize=10)
    ax.set_ylim(0.6, 1.45)
    ax.set_ylabel('Test RMSE / least squares of the base model', color=INK_2, fontsize=9)
    ax.set_title('Test error by variant: the network (bars) matches the least-squares optimum (ticks) of each model',
                 color=INK, fontsize=11, loc='left', pad=34)
    ax.legend(ncols=6, frameon=False, fontsize=9, labelcolor=INK_2, loc='lower left', bbox_to_anchor=(0, 1.0),
              handlelength=1.5, columnspacing=1.2)
    fig.text(0.01, 0.01, 'Labels: change of the last variant against the first version. '
             'Faded: target has negative values, so the last variant keeps the MSE loss.',
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def predict(plot_dir, dataset, run, datetimes):
    """The saved network's prediction at the given datetimes, in the target's units."""
    saved = torch.load(plot_dir / dataset / run / 'model.pt', weights_only=False)
    model = SinusoidalNetwork(**saved['kwargs'])
    model.load_state_dict(saved['state_dict'])
    scaling = saved['scaling']
    with torch.no_grad():
        y = model(torch.tensor(scaling.days(datetimes), dtype=torch.float32)).numpy()
    return scaling.unscale(y)


def plot_window(ax, plot_dir, data, dataset, runs, start, end, title, ylabel):
    hours = pd.date_range(start, end, freq='h')
    window = data[(data['datetime'] >= hours[0]) & (data['datetime'] <= hours[-1])]
    style(ax)
    ax.scatter(window['datetime'], window['y'], s=9, color=MUTED, label='Data (test period)', zorder=3)
    for color, (label, run) in zip(SERIES, runs):
        ax.plot(hours, predict(plot_dir, dataset, run, hours), color=color, linewidth=2, label=label)
    ax.set_xlim(hours[0], hours[-1])
    ax.set_title(title, color=INK, fontsize=10, loc='left', pad=22)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=max(1, (hours[-1] - hours[0]).days // 7)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%a %d' if len(hours) > 8 * 24 else '%a %d %b'))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3,
              borderaxespad=0.2)  # between the title and the plot, off the data
    return window


def plot_fit_example(plot_dir, data_dir, out_path, start='2017-08-11', end='2017-09-01'):
    """Three ordinary weeks of metro's test period, chosen as the median three-week window by error."""
    dataset, run = 'metro', VARIANTS[-1][1]('metro')
    data = DATASETS[dataset].load(data_dir)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
    window = plot_window(ax, plot_dir, data, dataset, [('Prediction, all options', run)], start, end, '', 'Cars per hour')
    error = predict(plot_dir, dataset, run, window['datetime']) - window['y'].to_numpy()
    test_rmse = json.loads((plot_dir / dataset / run / 'summary.json').read_text())['network']['test']['rmse']
    ax.set_title(f'metro, three weeks of the test period, never seen in training: RMSE {np.sqrt(np.mean(error ** 2)):.0f} '
                 f'cars per hour here, {test_rmse:.0f} over the whole test period', color=INK, fontsize=10, loc='left', pad=22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def plot_showcase(plot_dir, data_dir, out_path):
    fig = plt.figure(figsize=(12, 10), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2)
    data = {d: DATASETS[d].load(data_dir) for d in ('metro', 'bike', 'jena')}

    plot_window(fig.add_subplot(grid[0, :]), plot_dir, data['metro'], 'metro',
                [('Base', 'week84-year2'), ('+ holidays', 'week84-year2-holidays')],
                '2017-11-19', '2017-11-27', 'metro, Thanksgiving week: holidays remove the rush hours of Thursday',
                'Cars per hour')
    plot_window(fig.add_subplot(grid[1, :]), plot_dir, data['bike'], 'bike',
                [('Base (MSE)', 'week84-year2'),
                 ('+ holidays, modulation, Poisson loss', 'week84-year2-mod1-holidays-poisson')],
                '2012-12-17', '2012-12-30',
                'bike, Christmas in winter: the Poisson loss scales the daily swing with the level, and never goes negative',
                'Rentals per hour')
    for col, (start, end, season) in enumerate([('2016-01-11', '2016-01-15', 'January'),
                                                ('2016-07-11', '2016-07-15', 'July')]):
        ax = fig.add_subplot(grid[2, col])
        plot_window(ax, plot_dir, data['jena'], 'jena',
                    [('Base', 'week84-year2'), ('+ holidays, seasonal modulation', 'week84-year2-mod1-holidays')],
                    start, end, f'jena, {season}: modulation lets the daily swing change with the season',
                    'Temperature (°C)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare the trained variants and save the README figures.')
    parser.add_argument('--plot-dir', type=Path, default=Path('plots'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out-dir', type=Path, default=Path('docs'))
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    summaries = load_summaries(args.plot_dir)
    print(markdown_table(summaries))
    plot_comparison(summaries, args.out_dir / 'comparison.png')
    plot_fit_example(args.plot_dir, args.data_dir, args.out_dir / 'fit.png')
    plot_showcase(args.plot_dir, args.data_dir, args.out_dir / 'showcase.png')
    shutil.copy(args.plot_dir / 'metro' / 'week84-year2' / 'components_epoch_0500.png',
                args.out_dir / 'metro_components.png')


if __name__ == '__main__':
    main()
