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
    """Predicts y(t) = offset + slope·t/span + Σ α·sin(ω·t) + β·cos(ω·t) over periodic components.

    t is time in days and span is the length of the training range in days. A
    component with period P and n sinusoids has its frequencies fixed at the
    harmonics ω = 2πk/P, k = 1..n. The parameters (sine and cosine
    coefficients α and β, offset, slope) are produced by an MLP fed a constant
    input. α·sin + β·cos is a sinusoid with amplitude √(α²+β²); unlike
    amplitude and phase, it makes the squared error convex in α and β.

    With modulation N, the α and β of every component but the slowest vary
    over the slowest one's period Q (the year by default) as
    α(t) = α + Σ_j γ_j·cos(2πjt/Q) + δ_j·sin(2πjt/Q), j = 1..N, so the
    weekly shape can differ between summer and winter.

    `holiday_of_day` gives, for each day since t = 0, the index (1, 2, ...) of
    the holiday on it, or 0. A holiday adds a learned value per holiday plus a
    daily profile shared by all holidays, a sum of HOLIDAY_HARMONICS daily
    harmonics, e.g. to remove the rush hours of a holiday on a weekday.
    """

    HOLIDAY_HARMONICS = 12

    def __init__(self, components, span, log_link=False, modulation=0, holiday_of_day=None,
                 input_size=64, hidden_size1=128, hidden_size2=256):
        super(SinusoidalNetwork, self).__init__()
        self.components = dict(components)
        self.span = span
        self.log_link = log_link  # predict exp(sum): the components multiply instead of adding up
        harmonics = [2 * math.pi * k / period_days(period)
                     for period, count in self.components.items() for k in range(1, count + 1)]
        self.register_buffer('harmonics', torch.tensor(harmonics))
        slowest = max(self.components, key=period_days)
        self.register_buffer('modulated', torch.tensor(
            [period != slowest for period, count in self.components.items() for _ in range(count)]))
        self.register_buffer('seasons', torch.tensor(
            [2 * math.pi * j / period_days(slowest) for j in range(1, modulation + 1)]))
        self.n_modulation = 2 * int(self.modulated.sum()) * 2 * modulation
        self.register_buffer('holiday_of_day', torch.zeros(0, dtype=torch.long) if holiday_of_day is None
                             else torch.as_tensor(holiday_of_day, dtype=torch.long))
        self.n_holidays = int(self.holiday_of_day.max()) if len(self.holiday_of_day) else 0
        n_holiday = self.n_holidays + 2 * self.HOLIDAY_HARMONICS if self.n_holidays else 0
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2 * len(harmonics) + 2 + self.n_modulation + n_holiday)

    def forward(self, t):
        _, _, offset, slope, _, holiday = self.get_params()
        y = offset + slope * t / self.span + self.sinusoids(t).sum(dim=1)
        if self.n_holidays:
            y = y + self.holiday_effect(t, holiday)
        return y.exp() if self.log_link else y

    def holiday_effect(self, t, holiday):
        index = self.holiday_of_day[t.floor().long().clamp(0, len(self.holiday_of_day) - 1)]
        value, coefficients = holiday[:self.n_holidays], holiday[self.n_holidays:].view(2, -1)
        phase = 2 * math.pi * torch.arange(1, self.HOLIDAY_HARMONICS + 1, device=t.device) * t.unsqueeze(1)
        profile = (coefficients[0] * torch.sin(phase) + coefficients[1] * torch.cos(phase)).sum(dim=1)
        return torch.where(index > 0, value[(index - 1).clamp(min=0)] + profile, 0.0)

    def get_params(self):
        x = torch.relu(self.fc1(torch.ones(self.fc1.in_features, device=self.fc1.weight.device)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        n = len(self.harmonics)
        alpha, beta = x[:2 * n].view(2, n)
        offset, slope = x[2 * n:2 * n + 2]
        # γ and δ for α and β of each modulated sinusoid: (2, modulated sinusoids, 2·modulation)
        modulation = x[2 * n + 2:2 * n + 2 + self.n_modulation].view(2, int(self.modulated.sum()), 2 * len(self.seasons))
        holiday = x[2 * n + 2 + self.n_modulation:]
        return alpha, beta, offset, slope, modulation, holiday

    def sinusoids(self, t):
        """Every sinusoid evaluated at t, shape (len(t), number of harmonics)."""
        alpha, beta, _, _, modulation, _ = self.get_params()
        if len(self.seasons):
            season = self.seasons * t.unsqueeze(1)
            season = torch.cat([torch.cos(season), torch.sin(season)], dim=1)  # (len(t), 2·modulation)
            alpha = alpha.expand(len(t), -1).clone()
            beta = beta.expand(len(t), -1).clone()
            alpha[:, self.modulated] += season @ modulation[0].T
            beta[:, self.modulated] += season @ modulation[1].T
        phase = self.harmonics * t.unsqueeze(1)
        return alpha * torch.sin(phase) + beta * torch.cos(phase)

    def waves(self, t):
        """Each sinusoid evaluated at t, as {period: tensor of shape (len(t), count)}."""
        values = self.sinusoids(t)
        return dict(zip(self.components, values.split(list(self.components.values()), dim=1)))
