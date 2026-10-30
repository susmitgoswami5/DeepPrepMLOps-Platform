#!/usr/bin/env python
"""Scheduled retraining: trains a *challenger* model on freshly simulated orders.

Run by the GitHub Actions workflow, followed by scripts/validate_model.py, which
compares the challenger with the current champion on a fixed held-out set.
"""
import json
import os
import sys
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.core import build_dataset, train_model  # noqa: E402

CHALLENGER_PATH = "models/challenger_model.pt"
EPOCHS = 40
N_ORDERS = 15000


def main():
    # A new simulation seed each day stands in for "new data arriving".
    seed = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    print(f"Retraining challenger | seed={seed} | {N_ORDERS} simulated orders | {EPOCHS} epochs")
    df = build_dataset(N_ORDERS, seed=seed)
    model = train_model(df, epochs=EPOCHS, seed=seed, verbose=True)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), CHALLENGER_PATH)
    with open("models/training_metadata.json", "w") as f:
        json.dump({"trained_at": datetime.now(timezone.utc).isoformat(), "data_seed": seed,
                   "num_orders": df.height, "epochs": EPOCHS, "loss": "quantile (P50/P90/P95)",
                   "model_path": CHALLENGER_PATH}, f, indent=2)
    print(f"Challenger saved to {CHALLENGER_PATH}")


if __name__ == "__main__":
    main()
