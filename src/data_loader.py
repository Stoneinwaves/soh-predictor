"""Data loading utilities for SOH estimation project.

This module provides helpers to:
- Parse SOH (capacity‑fade) CSVs and build a cycle→SOH dictionary.
- Parse old EIS CSVs whose filenames follow the pattern ``EIS_state_V_*.csv``.
- Parse new Excel EIS files whose filenames embed SOH in the pattern ``Cell*_*SOH_*degC_95SOC_*.xls``.
- Assemble PyTorch ``DataLoader`` objects ready for training / evaluation.

All loaders normalise frequency (log‑scaled) and |Z| independently into [0,1] via ``MinMaxScaler``.
The final feature vector for a single spectrum is the concatenation of the scaled log‑frequencies
and the scaled |Z| values.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Tuple, List, Dict, Sequence

import numpy as np
import pandas as pd
import scipy.interpolate as interp
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split

__all__ = [
    "load_soh_data",
    "load_eis_data",
    "load_new_excel_data",
    "build_dataloaders",
]


def load_soh_data(pattern: str, capacity_base: float = 45.0) -> Dict[int, float]:
    """Return a mapping {cycle_index: soh} parsed from *capacity* CSVs.

    The expected CSV layout (without header) is::
        …,cycle_index,…,measurement_capacity,…
    Only *cycle_index* (col 1) and measured capacity (col 3) are used.
    ``capacity_base`` gives the rated capacity (mAh) used to convert to SOH.
    """
    soh_dict: Dict[int, float] = {}
    for file in glob.glob(pattern):
        df = pd.read_csv(file, header=None, skiprows=1)
        for _, row in df.iterrows():
            try:
                cycle = int(row[1])
                soh = float(row[3]) / capacity_base
                soh_dict[cycle] = soh
            except (ValueError, IndexError):
                continue
    if not soh_dict:
        raise FileNotFoundError(f"No SOH data matched pattern: {pattern}")
    return soh_dict


def _scale_pair(arr_x: np.ndarray, arr_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two arrays independently scaled to [0,1] with *MinMaxScaler*."""
    sx = MinMaxScaler().fit_transform(arr_x.reshape(-1, 1)).flatten()
    sy = MinMaxScaler().fit_transform(arr_y.reshape(-1, 1)).flatten()
    return sx, sy


def load_eis_data(pattern: str, soh_dict: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Load *old* EIS CSVs and build *X*, *y* arrays suitable for model input."""
    data_X: List[np.ndarray] = []
    data_y: List[float] = []

    for file in glob.glob(pattern):
        df = pd.read_csv(file, header=None, skiprows=1)[[1, 2, 3, 4]]
        df.columns = ["cycle", "freq", "Re", "Im"]
        df["log_freq"] = np.log10(df["freq"].values)
        df["Z_abs"] = np.sqrt(df["Re"] ** 2 + df["Im"] ** 2)

        for cycle, grp in df.groupby("cycle"):
            if cycle not in soh_dict:
                continue
            grp_sorted = grp.sort_values("log_freq")
            freq_scaled, z_scaled = _scale_pair(
                grp_sorted["log_freq"].values, grp_sorted["Z_abs"].values
            )
            feature_vec = np.concatenate([freq_scaled, z_scaled])
            data_X.append(feature_vec)
            data_y.append(soh_dict[cycle])

    if not data_X:
        raise FileNotFoundError(f"No EIS data matched pattern: {pattern}")

    return np.asarray(data_X), np.asarray(data_y)


def load_new_excel_data(
    pattern: str, *, target_N: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """Load *new* Excel EIS spectra and return (*X*, *y*)."""
    features: List[np.ndarray] = []
    labels: List[float] = []

    for file in glob.glob(pattern):
        filename = Path(file).name
        try:
            soh_pct = int(filename.split("_")[1][:-3])  # extract e.g. "95" from "Cell1_95SOH_…"
        except (IndexError, ValueError):
            print(f"[WARN] Failed to parse SOH from filename: {filename}")
            continue
        soh_val = soh_pct / 100.0

        df = pd.read_excel(file, header=None)
        if df.shape[1] < 3:
            print(f"[WARN] Insufficient columns in {filename}")
            continue

        freq, Re, Im = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values
        log_freq = np.log10(freq)
        z_abs = np.sqrt(Re ** 2 + Im ** 2)

        sort_idx = np.argsort(log_freq)
        log_freq_sorted, z_abs_sorted = log_freq[sort_idx], z_abs[sort_idx]

        target_log_freq = np.linspace(log_freq_sorted.min(), log_freq_sorted.max(), target_N)
        z_interp = interp.interp1d(
            log_freq_sorted, z_abs_sorted, kind="linear", fill_value="extrapolate"
        )(target_log_freq)

        freq_scaled, z_scaled = _scale_pair(target_log_freq, z_interp)
        features.append(np.concatenate([freq_scaled, z_scaled]))
        labels.append(soh_val)

    if not features:
        raise FileNotFoundError(f"No Excel data matched pattern: {pattern}")

    return np.asarray(features), np.asarray(labels)


def build_dataloaders(
    *,
    soh_pattern: str = "processed_data_Capacity_*.csv",
    eis_pattern: str = "EIS_state_V_*.csv",
    excel_pattern: str = "Cell*_*SOH_*degC_95SOC_*.xls",
    batch_size: int = 32,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    capacity_base: float = 45.0,
    target_N: int = 60,
) -> Tuple[DataLoader, DataLoader, int]:
    """Return (train_loader, test_loader, input_dim)."""
    soh_dict = load_soh_data(soh_pattern, capacity_base)
    X_old, y_old = load_eis_data(eis_pattern, soh_dict)
    X_new, y_new = load_new_excel_data(excel_pattern, target_N=target_N)

    ds_old = TensorDataset(torch.tensor(X_old, dtype=torch.float32),
                           torch.tensor(y_old, dtype=torch.float32).view(-1, 1))
    ds_new = TensorDataset(torch.tensor(X_new, dtype=torch.float32),
                           torch.tensor(y_new, dtype=torch.float32).view(-1, 1))

    full_ds = ConcatDataset([ds_old, ds_new])

    torch.manual_seed(random_seed)
    test_len = int(len(full_ds) * test_ratio)
    train_len = len(full_ds) - test_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_old.shape[1]  # freq_scaled + z_scaled
    return train_loader, test_loader, input_dim
