"""Evaluation script for trained SOH model."""
from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import build_dataloaders
from model import SOHFCNN


def evaluate(
    *,
    model_path: str,
    soh_pattern: str,
    eis_pattern: str,
    excel_pattern: str,
    batch_size: int = 32,
):
    train_loader, test_loader, input_dim = build_dataloaders(
        soh_pattern=soh_pattern, eis_pattern=eis_pattern, excel_pattern=excel_pattern, batch_size=batch_size
    )

    model = SOHFCNN(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for feats, target in test_loader:
            output = model(feats)
            preds.extend(output.squeeze().tolist())
            labels.extend(target.squeeze().tolist())

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOH model")
    parser.add_argument("--model_path", default="merged_model.pth")
    parser.add_argument("--soh_pattern", default="processed_data_Capacity_*.csv")
    parser.add_argument("--eis_pattern", default="EIS_state_V_*.csv")
    parser.add_argument("--excel_pattern", default="Cell*_*SOH_*degC_95SOC_*.xls")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        soh_pattern=args.soh_pattern,
        eis_pattern=args.eis_pattern,
        excel_pattern=args.excel_pattern,
    )


if __name__ == "__main__":
    main()
