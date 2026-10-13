import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sinetwork.data import Scaling, make_loader, split
from sinetwork.datasets import DATASETS, csv_dataset
from sinetwork.model import PERIODS, SinusoidalNetwork, parse_components
from sinetwork.plot import plot_components, plot_fit, plot_series


def evaluate_rmse(model, loader, scaling):
    """Root mean squared error in the target's original units."""
    model.eval()
    squared_error, count = 0.0, 0
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            squared_error += ((model(batch_inputs) - batch_targets) ** 2).sum().item()
            count += len(batch_targets)
    return (squared_error / count) ** 0.5 * scaling.y_std


def train_model(model, dataset, train_df, test_df, scaling, train_loader, test_loader, criterion, optimizer,
                plot_dir, num_epochs=1000, plot_every=50):
    metrics_path = plot_dir / 'metrics.csv'
    metrics_path.write_text('epoch,loss,train_rmse,test_rmse\n')
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
            train_rmse = evaluate_rmse(model, train_loader, scaling)
            test_rmse = evaluate_rmse(model, test_loader, scaling)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, '
                  f'Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
            with open(metrics_path, 'a') as f:
                f.write(f'{epoch+1},{loss:.6f},{train_rmse:.6f},{test_rmse:.6f}\n')
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

    # plots/<dataset>/data.png, and one folder of fits per component setting, e.g. plots/bike/day6-week3-year3/
    dataset_plot_dir = args.plot_dir / dataset.name
    run_name = '-'.join(f'{period if period in PERIODS else period + "d"}{count}' for period, count in components.items())
    plot_dir = dataset_plot_dir / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_series(df, dataset.label, dataset_plot_dir / 'data.png')

    train_df, test_df = split(df, args.test_fraction)
    scaling = Scaling.fit(train_df)
    train_loader = make_loader(train_df, scaling, device, args.batch_size)
    test_loader = make_loader(test_df, scaling, device, args.batch_size, shuffle=False)

    model = SinusoidalNetwork(components, span=scaling.days([train_df['datetime'].max()])[0]).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, fused=True)

    train_model(model, dataset, train_df, test_df, scaling, train_loader, test_loader, criterion, optimizer,
                plot_dir, args.epochs, args.plot_every)

    print("Training complete.")


if __name__ == '__main__':
    main()
