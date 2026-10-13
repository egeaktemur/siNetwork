import torch
import torch.nn as nn


class SinusoidalNetwork(nn.Module):
    """Predicts T as a sum of learned sinusoids over day-of-year and hour.

    The sinusoid parameters (amplitude a, frequency b, phase c) are produced by
    an MLP fed a constant input; only columns 0 (day_of_year) and 1 (hour) of x are used.
    """

    def __init__(self, input_size=7, hidden_size1=128, hidden_size2=128, sinusoidal_count=5):
        super(SinusoidalNetwork, self).__init__()
        self.sinusoidal_count = sinusoidal_count
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, sinusoidal_count * 6)

    def forward(self, x):
        a_day, b_day, c_day, a_hour, b_hour, c_hour = self.get_sinusoidal_wave()
        day = x[:, 0].unsqueeze(1)
        hour = x[:, 1].unsqueeze(1)
        sinusoidal_sum_day = torch.sum(a_day * torch.sin(b_day * day + c_day), dim=1)
        sinusoidal_sum_hour = torch.sum(a_hour * torch.sin(b_hour * hour + c_hour), dim=1)
        sinusoidal_sum = sinusoidal_sum_day + sinusoidal_sum_hour
        return sinusoidal_sum

    def get_sinusoidal_wave(self):
        x = torch.relu(self.fc1(torch.ones(self.fc1.in_features, device=self.fc1.weight.device)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        params = x.view(-1, 6, self.sinusoidal_count)
        a_day, b_day, c_day = params[:, 0], params[:, 1], params[:, 2]
        a_hour, b_hour, c_hour = params[:, 3], params[:, 4], params[:, 5]
        return a_day, b_day, c_day, a_hour, b_hour, c_hour
