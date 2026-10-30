"""Shared training / evaluation logic.

Used by the Prefect pipeline, the nightly retraining scripts and the evaluation
script, so that every part of the system trains and scores the model the same way.
"""
import random

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.data.mock_generator import generate_mock_logs
from src.data.restaurant_profiles import RESTAURANT_IDS
from src.features.engineering import engineer_features
from src.model.architecture import DeepPrepModel, quantile_loss
from src.training.dataset import DeepPrepDataset

QUANTILES = [0.5, 0.9, 0.95]
QUEUE_FEATURE_SLICE = slice(7, 10)  # cook_utilization, queue_wait_time_m, orders_in_queue

# Stable restaurant -> embedding index mapping (sorted, so it never changes between runs)
RESTAURANT_TO_IDX = {r: i for i, r in enumerate(sorted(RESTAURANT_IDS))}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(n_orders: int = 6000, seed: int = 0) -> pl.DataFrame:
    """Simulate orders and engineer features, sorted by creation time."""
    return engineer_features(generate_mock_logs(n_orders, seed=seed)).sort("created_at")


def chronological_split(df: pl.DataFrame, train_frac: float = 0.8):
    """Train on the past, test on the future (no look-ahead leakage)."""
    cut = int(df.height * train_frac)
    return df[:cut], df[cut:]


def _dataset(df: pl.DataFrame, drop_queue: bool) -> DeepPrepDataset:
    ds = DeepPrepDataset(df, RESTAURANT_TO_IDX)
    if drop_queue:  # ablation: remove the queueing-theory signal
        ds.wide_features[:, QUEUE_FEATURE_SLICE] = 0.0
    return ds


def train_model(train_df: pl.DataFrame, epochs: int = 40, lr: float = 3e-3, batch_size: int = 64,
                seed: int = 0, drop_queue: bool = False, verbose: bool = False) -> DeepPrepModel:
    set_seed(seed)
    ds = _dataset(train_df, drop_queue)
    model = DeepPrepModel(num_restaurants=len(RESTAURANT_TO_IDX))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for r, w, s, t in loader:
            opt.zero_grad()
            loss = quantile_loss(model(r, w, s), t, QUANTILES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(r)
        sched.step()
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            print(f"  epoch {epoch + 1:3d}/{epochs}  quantile loss {total / len(ds):.4f}")
    model.eval()
    return model


@torch.no_grad()
def predict(model: DeepPrepModel, df: pl.DataFrame, drop_queue: bool = False) -> np.ndarray:
    """Returns an array of shape (n, 3) with p50, p90, p95 predictions in minutes."""
    model.eval()
    out = [model(r, w, s).numpy() for r, w, s, _ in DataLoader(_dataset(df, drop_queue), batch_size=1024)]
    return np.concatenate(out)


def quantile_metrics(p50: np.ndarray, p95: np.ndarray, y: np.ndarray) -> dict:
    return {
        "mae_p50_min": float(np.mean(np.abs(p50 - y))),
        "p95_coverage": float(np.mean(y <= p95)),                     # ideal 0.95
        "pinball_p95": float(np.mean(np.maximum(0.95 * (y - p95), -0.05 * (y - p95)))),
    }


def evaluate(model: DeepPrepModel, df: pl.DataFrame, drop_queue: bool = False) -> dict:
    p = predict(model, df, drop_queue)
    return quantile_metrics(p[:, 0], p[:, 2], df["target_fpt_minutes"].to_numpy())
