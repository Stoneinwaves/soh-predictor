"""Neural network model definition for SOH estimation."""
from __future__ import annotations

import torch.nn as nn

BEST_PARAMS = {
    "hidden_size": 1024,
    "learning_rate": 1e-3,
    "dropout_rate": 0.17,
}


class SOHFCNN(nn.Module):
    """Fully‑connected baseline model used for impedance‑to‑SOH regression."""

    def __init__(self, input_dim: int, hidden_size: int | None = None, dropout: float | None = None):
        super().__init__()
        hidden_size = hidden_size or BEST_PARAMS["hidden_size"]
        dropout = dropout or BEST_PARAMS["dropout_rate"]

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # noqa: D401 (simple return)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
