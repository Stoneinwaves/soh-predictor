"""Training script for SOH estimation model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import build_dataloaders
from model import SOHFCNN, BEST_PARAMS
from util import plot_train_curve


def _train_one_epoch(model: SOHFCNN, loader: torch.utils.data.DataLoader, criterion, optimizer) -> float:
    model.train()
    running_loss = 0.0
    for feats, labels in loader:
        optimizer.zero_grad()
        output = model(feats)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * feats.size(0)
    return running_loss / len(loader.dataset)


def train_model(
    *,
    soh_pattern: str,
    eis_pattern: str,
    excel_pattern: str,
    epochs: int = 100,
    batch_size: int = 32,
    model_path: str | Path = "merged_model.pth",
) -> Tuple[SOHFCNN, list[float]]:
    train_loader, test_loader, input_dim = build_dataloaders(
        soh_pattern=soh_pattern, eis_pattern=eis_pattern, excel_pattern=excel_pattern, batch_size=batch_size
    )

    model = SOHFCNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS["learning_rate"])

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = _train_one_epoch(model, train_loader, criterion, optimizer)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved → {model_path}")

    plot_train_curve(losses)
    return model, losses


def main():
    parser = argparse.ArgumentParser(description="Train SOH model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", default="merged_model.pth")
    parser.add_argument("--soh_pattern", default="processed_data_Capacity_*.csv")
    parser.add_argument("--eis_pattern", default="EIS_state_V_*.csv")
    parser.add_argument("--excel_pattern", default="Cell*_*SOH_*degC_95SOC_*.xls")
    args = parser.parse_args()

    train_model(
        soh_pattern=args.soh_pattern,
        eis_pattern=args.eis_pattern,
        excel_pattern=args.excel_pattern,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
