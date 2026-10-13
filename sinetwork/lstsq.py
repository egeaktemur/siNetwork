"""Least-squares reference fits of a SinusoidalNetwork's model.

Whatever its options, the network's prediction is linear in the numbers its MLP
outputs (or its logarithm is, with a log link), so the best fit of the same
model can be computed directly: by linear least squares for the MSE loss, and by
iteratively reweighted least squares (Newton's method) for the Poisson loss.
Comparing the network with it shows how close training gets to the optimum.
"""
import math

import numpy as np


def design_matrix(model, t):
    """One column per output of the model's MLP, so that prediction (or its log) = X @ outputs."""
    t = np.asarray(t, dtype=float)
    harmonics = model.harmonics.cpu().numpy().astype(float)
    phase = np.outer(t, harmonics)
    columns = [np.sin(phase), np.cos(phase), np.ones((len(t), 1)), (t / model.span)[:, None]]
    if len(model.seasons):
        modulated = model.modulated.cpu().numpy()
        season = np.outer(t, model.seasons.cpu().numpy().astype(float))
        season = np.hstack([np.cos(season), np.sin(season)])
        for wave in (np.sin(phase[:, modulated]), np.cos(phase[:, modulated])):  # γ, δ of α, then of β
            columns.append((wave[:, :, None] * season[:, None, :]).reshape(len(t), -1))
    if model.n_holidays:
        holiday_of_day = model.holiday_of_day.cpu().numpy()
        index = holiday_of_day[np.clip(np.floor(t).astype(int), 0, len(holiday_of_day) - 1)]
        columns.append((index[:, None] == np.arange(1, model.n_holidays + 1)).astype(float))
        daily = 2 * math.pi * np.outer(t, np.arange(1, model.HOLIDAY_HARMONICS + 1))
        columns += [np.sin(daily) * (index > 0)[:, None], np.cos(daily) * (index > 0)[:, None]]
    return np.hstack(columns)


def fit(model, t, y, iterations=30):
    """The coefficients of design_matrix(model, t) that best fit the scaled target y."""
    X = design_matrix(model, t)
    if not model.log_link:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    coefficients = np.zeros(X.shape[1])  # exp(0) = 1, the mean of the target scaled by its mean
    for _ in range(iterations):
        eta = X @ coefficients
        mu = np.exp(eta)
        weight = np.sqrt(mu)
        coefficients = np.linalg.lstsq(X * weight[:, None], weight * (eta + (y - mu) / mu), rcond=None)[0]
    return coefficients


def predict(model, coefficients, t):
    eta = design_matrix(model, t) @ coefficients
    return np.exp(eta) if model.log_link else eta
