"""Miscellaneous helper utilities: plotting, scaler persistence, etc."""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "plot_train_curve",
    "save_scalers_and_vector",
]


def plot_train_curve(losses: list[float], save_path: str | Path = "train_loss.png") -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_scalers_and_vector(
    freq: np.ndarray,
    Re: np.ndarray,
    Im: np.ndarray,
    *,
    target_N: int,
    scaler_f_path: str | Path = "scaler_f.pkl",
    scaler_z_path: str | Path = "scaler_z.pkl",
) -> np.ndarray:
    """Create (and persist) frequency/impedance scalers, return feature vector."""
    log_freq = np.log10(freq)
    z_abs = np.sqrt(Re ** 2 + Im ** 2)

    sort_idx = np.argsort(log_freq)
    log_freq_sorted, z_abs_sorted = log_freq[sort_idx], z_abs[sort_idx]

    target_log_freq = np.linspace(log_freq_sorted.min(), log_freq_sorted.max(), target_N)
    z_abs_interp = interp.interp1d(
        log_freq_sorted, z_abs_sorted, kind="linear", fill_value="extrapolate"
    )(target_log_freq)

    scaler_f = MinMaxScaler()
    scaler_z = MinMaxScaler()

    freq_scaled = scaler_f.fit_transform(target_log_freq.reshape(-1, 1)).flatten()
    z_scaled = scaler_z.fit_transform(z_abs_interp.reshape(-1, 1)).flatten()

    joblib.dump(scaler_f, scaler_f_path)
    joblib.dump(scaler_z, scaler_z_path)

    return np.concatenate([freq_scaled, z_scaled])
