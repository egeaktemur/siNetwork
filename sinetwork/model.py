import math

import torch
import torch.nn as nn

# Named periods in days
PERIODS = {'day': 1.0, 'week': 7.0, 'year': 365.25}


def parse_components(specs):
    """Parse ['day:5', 'year:4', '0.5:2'] into {period: count}; a period is a name in PERIODS or days."""
    components = {}
    for spec in specs:
        period, count = spec.split(':')
        components[period] = int(count)
    return components


def period_days(period):
    return PERIODS[period] if period in PERIODS else float(period)


class SinusoidalNetwork(nn.Module):
    """Predicts y(t) = offset + slope·t/span + Σ a·sin(b·t + c) over periodic components.

    t is time in days and span is the length of the training range in days. A
    component with period P and n sinusoids has its frequencies fixed at the
    harmonics 2πk/P, k = 1..n. The parameters (amplitude a, phase c, offset,
    slope) are produced by an MLP fed a constant input.
    """

    def __init__(self, components, span, input_size=64, hidden_size1=128, hidden_size2=256):
        super(SinusoidalNetwork, self).__init__()
        self.components = dict(components)
        self.span = span
        harmonics = [2 * math.pi * k / period_days(period)
                     for period, count in self.components.items() for k in range(1, count + 1)]
        self.register_buffer('harmonics', torch.tensor(harmonics))
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2 * len(harmonics) + 2)

    def forward(self, t):
        a, b, c, offset, slope = self.get_params()
        return offset + slope * t / self.span + (a * torch.sin(b * t.unsqueeze(1) + c)).sum(dim=1)

    def get_params(self):
        x = torch.relu(self.fc1(torch.ones(self.fc1.in_features, device=self.fc1.weight.device)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        n = len(self.harmonics)
        a, c = x[:2 * n].view(2, n)
        offset, slope = x[2 * n:]
        return a, self.harmonics, c, offset, slope

    def waves(self, t):
        """Each sinusoid evaluated at t, as {period: tensor of shape (len(t), count)}."""
        a, b, c, _, _ = self.get_params()
        values = a * torch.sin(b * t.unsqueeze(1) + c)
        return dict(zip(self.components, values.split(list(self.components.values()), dim=1)))
