"""Held-out evaluation and ablation for DeepPrep.

Questions answered:
  1. How accurate are the P50 / P95 predictions on future (held-out) orders?
  2. Do the M/M/c queueing features help?  (ablation: queue features set to 0)
  3. How does the neural model compare with simple baselines?

Protocol: simulate orders -> chronological 80/20 split (train on the past, test on
the future) -> repeat for 3 random seeds -> report mean and standard deviation.

Run from the repository root:
    python -m evaluation.evaluate            # full run (3 seeds)
    python -m evaluation.evaluate --quick    # 1 seed, fewer orders (smoke test)
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.training.core import (build_dataset, chronological_split, evaluate,
                               quantile_metrics, train_model)

WIDE = ["hour_sin", "hour_cos", "day_sin", "day_cos", "embed_0", "embed_1", "embed_2",
        "cook_utilization", "queue_wait_time_m", "orders_in_queue"]
LOADS = ["load_15m", "load_30m", "load_60m"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def gbm_quantiles(tr, te, extra_item_count: bool, seed: int) -> dict:
    """Gradient-boosted quantile regression on the same inputs (+ restaurant id)."""
    def X(d):
        cols = [d.select(WIDE + LOADS).to_numpy(),
                d["restaurant_id"].str.slice(1).cast(int).to_numpy()[:, None]]
        if extra_item_count:
            cols.append(d["items"].list.len().to_numpy()[:, None])
        return np.hstack(cols)

    y = tr["target_fpt_minutes"].to_numpy()
    preds = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                              learning_rate=0.05, random_state=seed)
             .fit(X(tr), y).predict(X(te)) for q in (0.5, 0.95)}
    return quantile_metrics(preds[0.5], preds[0.95], te["target_fpt_minutes"].to_numpy())


def run(seeds, n_orders, epochs):
    runs = []
    for seed in seeds:
        df = build_dataset(n_orders, seed=seed)
        tr, te = chronological_split(df)
        y_tr, y_te = tr["target_fpt_minutes"].to_numpy(), te["target_fpt_minutes"].to_numpy()
        res = {
            "seed": seed, "n_train": tr.height, "n_test": te.height,
            "constant_baseline": quantile_metrics(np.full_like(y_te, np.quantile(y_tr, .5)),
                                                  np.full_like(y_te, np.quantile(y_tr, .95)), y_te),
            "deepprep_full": evaluate(train_model(tr, epochs=epochs, seed=seed), te),
            "deepprep_without_queue_features": evaluate(
                train_model(tr, epochs=epochs, seed=seed, drop_queue=True), te, drop_queue=True),
            "gbm_same_inputs": gbm_quantiles(tr, te, False, seed),
            "gbm_plus_item_count": gbm_quantiles(tr, te, True, seed),
        }
        print(json.dumps(res, indent=1), flush=True)
        runs.append(res)

    models = [k for k in runs[0] if isinstance(runs[0][k], dict)]
    summary = {m: {k: {"mean": round(float(np.mean([r[m][k] for r in runs])), 3),
                       "std": round(float(np.std([r[m][k] for r in runs])), 3)}
                   for k in runs[0][m]} for m in models}
    return runs, summary


def to_markdown(summary, n_orders, seeds) -> str:
    names = {
        "constant_baseline": "Constant baseline (training-set quantiles)",
        "deepprep_full": "**DeepPrep (Wide & Deep + queue features)**",
        "deepprep_without_queue_features": "DeepPrep without queue features (ablation)",
        "gbm_same_inputs": "Gradient-boosted quantile regression, same inputs",
        "gbm_plus_item_count": "Gradient-boosted quantile regression + item count",
    }
    lines = [f"Simulated orders: {n_orders} per seed · chronological 80/20 split · seeds {list(seeds)} · mean ± std",
             "", "| Model | P50 MAE (min) ↓ | P95 coverage (target 0.95) | P95 pinball loss ↓ |",
             "|---|---|---|---|"]
    for k, label in names.items():
        s = summary[k]
        lines.append(f"| {label} | {s['mae_p50_min']['mean']:.2f} ± {s['mae_p50_min']['std']:.2f} "
                     f"| {s['p95_coverage']['mean']:.3f} ± {s['p95_coverage']['std']:.3f} "
                     f"| {s['pinball_p95']['mean']:.3f} ± {s['pinball_p95']['std']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds, n, ep = ((0,), 3000, 5) if a.quick else ((0, 1, 2), 15000, 40)
    runs, summary = run(seeds, n, ep)
    table = to_markdown(summary, n, seeds)
    print(table)
    if not a.quick:
        json.dump({"runs": runs, "summary": summary}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=1)
        open(os.path.join(OUT_DIR, "results.md"), "w").write(table)
