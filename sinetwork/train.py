import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sinetwork.data import Scaling, make_loader, split
from sinetwork.datasets import DATASETS, csv_dataset
from sinetwork.model import PERIODS, SinusoidalNetwork, parse_components
from sinetwork.plot import plot_components, plot_fit, plot_series


METRICS = ['rmse', 'mae', 'bias', 'r2']


def evaluate(model, loader, scaling):
    """RMSE, MAE and bias (mean of prediction - target) in the target's units, and R²."""
    model.eval()
    with torch.no_grad():
        predictions, targets = zip(*[(model(inputs), batch_targets) for inputs, batch_targets in loader])
    error = torch.cat(predictions) - torch.cat(targets)
    targets = torch.cat(targets)
    mse = (error ** 2).mean().item()
    return {'rmse': mse ** 0.5 * scaling.y_std, 'mae': error.abs().mean().item() * scaling.y_std,
            'bias': error.mean().item() * scaling.y_std, 'r2': 1 - mse / targets.var(unbiased=False).item()}


def train_model(model, dataset, train_df, test_df, scaling, train_loader, test_loader, criterion, optimizer,
                plot_dir, num_epochs=1000, plot_every=50):
    metrics_path = plot_dir / 'metrics.csv'
    metrics_path.write_text(','.join(['epoch', 'loss'] + [f'{split}_{m}' for split in ('train', 'test') for m in METRICS]) + '\n')
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_inputs)  # Forward pass
            loss = criterion(outputs, batch_targets)  # Calculate loss
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            optimizer.step()  # Optimize
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            loss = epoch_loss / len(train_loader)
            train, test = evaluate(model, train_loader, scaling), evaluate(model, test_loader, scaling)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Train RMSE: {train["rmse"]:.4f}, '
                  f'Test RMSE: {test["rmse"]:.4f}, MAE: {test["mae"]:.4f}, Bias: {test["bias"]:+.4f}, '
                  f'R²: {test["r2"]:.3f}')
            with open(metrics_path, 'a') as f:
                f.write(','.join([str(epoch + 1), f'{loss:.6f}'] + [f'{m[k]:.6f}' for m in (train, test) for k in METRICS]) + '\n')
        if (epoch + 1) % plot_every == 0 or epoch + 1 == num_epochs:
            plot_fit(model, train_df, test_df, scaling, dataset.label, plot_dir / f'fit_epoch_{epoch+1:04d}.png')
            plot_components(model, scaling, dataset.label, plot_dir / f'components_epoch_{epoch+1:04d}.png')


def main():
    parser = argparse.ArgumentParser(description='Fit a time series as a learned sum of sinusoids.')
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='airquality')
    parser.add_argument('--csv', type=Path, help='fit a CSV file instead of a registered dataset')
    parser.add_argument('--time-col', help='timestamp column of --csv')
    parser.add_argument('--target-col', help='target column of --csv')
    parser.add_argument('--sep', default=',', help='separator of --csv')
    parser.add_argument('--components', nargs='+', metavar='PERIOD:COUNT', default=['week:84', 'year:2'],
                        help='sinusoids per period (period name or days); n sinusoids resolve the period '
                             'to about period/2n, so the default is hourly over the week and seasonal over the year')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, help='default: the whole training set in one batch')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--modulate', type=int, default=0, metavar='N',
                        help='let the faster components\' coefficients vary with the first N harmonics of the slowest')
    parser.add_argument('--loss', choices=['mse', 'poisson'], default='mse',
                        help='poisson predicts exp(sum of sinusoids), for non-negative targets such as counts')
    parser.add_argument('--test-fraction', type=float, default=0.2, help='final fraction of the time range held out')
    parser.add_argument('--plot-dir', type=Path, default=Path('plots'))
    parser.add_argument('--plot-every', type=int, default=50, help='save plots every N epochs and after the last')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.csv:
        if not (args.time_col and args.target_col):
            parser.error('--csv requires --time-col and --target-col')
        dataset = csv_dataset(args.csv, args.time_col, args.target_col, args.sep)
    else:
        dataset = DATASETS[args.dataset]
    components = parse_components(args.components)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = dataset.load(args.data_dir)
    print(df.describe())
    if args.loss == 'poisson' and (df['y'] < 0).any():
        parser.error(f'--loss poisson needs a non-negative target, {dataset.name} has values down to {df["y"].min()}')

    # plots/<dataset>/data.png, and one folder of fits per component setting, e.g. plots/bike/day6-week3-year3/
    dataset_plot_dir = args.plot_dir / dataset.name
    run_name = '-'.join(f'{period if period in PERIODS else period + "d"}{count}' for period, count in components.items())
    if args.modulate:
        run_name += f'-mod{args.modulate}'
    if args.loss != 'mse':
        run_name += f'-{args.loss}'
    plot_dir = dataset_plot_dir / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_series(df, dataset.label, dataset_plot_dir / 'data.png')

    train_df, test_df = split(df, args.test_fraction)
    scaling = Scaling.fit(train_df, center=args.loss == 'mse')
    train_loader = make_loader(train_df, scaling, device, args.batch_size)
    test_loader = make_loader(test_df, scaling, device, args.batch_size, shuffle=False)

    model = SinusoidalNetwork(components, span=scaling.days([train_df['datetime'].max()])[0],
                              log_link=args.loss == 'poisson', modulation=args.modulate).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss() if args.loss == 'mse' else nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, fused=True)

    train_model(model, dataset, train_df, test_df, scaling, train_loader, test_loader, criterion, optimizer,
                plot_dir, args.epochs, args.plot_every)

    print("Training complete.")


if __name__ == '__main__':
    main()
