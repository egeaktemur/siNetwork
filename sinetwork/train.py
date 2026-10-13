import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sinetwork.data import load_data, make_loader
from sinetwork.model import SinusoidalNetwork
from sinetwork.plot import plot_correlation, plot_temperature, plot_waves


def train_model(model, df, train_loader, criterion, optimizer, plot_dir, num_epochs=1000):
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')
            plot_waves(model, df, plot_dir / f'waves_epoch_{epoch+1:04d}.png')


def main():
    parser = argparse.ArgumentParser(description='Fit a sum of sinusoids to AirQualityUCI temperature.')
    parser.add_argument('--data', default='data/AirQualityUCI.csv')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--plot-dir', type=Path, default=Path('plots'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data(args.data)
    print(df.describe())

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    plot_correlation(df, args.plot_dir / 'correlation.png')
    plot_temperature(df, args.plot_dir / 'temperature.png')

    train_loader = make_loader(df, device)

    model = SinusoidalNetwork(64, 128, 256, 128).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, df, train_loader, criterion, optimizer, args.plot_dir, args.epochs)

    print("Training complete.")


if __name__ == '__main__':
    main()
