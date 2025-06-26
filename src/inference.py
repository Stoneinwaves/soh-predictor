"""Single-sample inference script using trained SOH model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

from model import SOHFCNN


def load_and_process_eis(file_path: str, target_N: int) -> torch.Tensor:
    data = pd.read_csv(file_path)
    freq = data["Frequency(Hz)"].values
    Re = data["IC(Re)"].values
    Im = data["IC(-Im)"].values

    log_freq = np.log10(freq)
    z_abs = np.sqrt(Re**2 + Im**2)
    sort_idx = np.argsort(log_freq)

    log_freq_sorted = log_freq[sort_idx]
    z_abs_sorted = z_abs[sort_idx]

    target_log_freq = np.linspace(log_freq_sorted.min(), log_freq_sorted.max(), target_N)
    z_interp = np.interp(target_log_freq, log_freq_sorted, z_abs_sorted)

    from sklearn.preprocessing import MinMaxScaler
    scaler_f = MinMaxScaler()
    scaler_z = MinMaxScaler()
    freq_scaled = scaler_f.fit_transform(target_log_freq.reshape(-1, 1)).flatten()
    z_scaled = scaler_z.fit_transform(z_interp.reshape(-1, 1)).flatten()

    feature_vector = np.concatenate([freq_scaled, z_scaled])
    return torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)



def infer_soh(model_path: str, eis_path: str, input_dim: int) -> float:
    model = SOHFCNN(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x_new = load_and_process_eis(eis_path, target_N=input_dim // 2)

    with torch.no_grad():
        pred = model(x_new).item()

    print(f"预测 SOH ≈ {pred * 100:.2f}%")
    return pred


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a single EIS file")
    parser.add_argument("--model_path", default="merged_model.pth")
    parser.add_argument("--eis_path", required=True)
    parser.add_argument("--input_dim", type=int, required=True)
    args = parser.parse_args()

    infer_soh(model_path=args.model_path, eis_path=args.eis_path, input_dim=args.input_dim)


if __name__ == "__main__":
    main()
