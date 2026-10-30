#!/usr/bin/env python
"""Validation gate: promote the challenger only if it beats the champion.

Both models are scored on the same fixed held-out set (seed 12345, never used for
training) using P95 pinball loss, the metric that matters for late-order risk.
Writes ``deploy=true|false`` to $GITHUB_OUTPUT when running in GitHub Actions.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.architecture import DeepPrepModel  # noqa: E402
from src.training.core import RESTAURANT_TO_IDX, build_dataset, evaluate  # noqa: E402

CHAMPION_PATH = "models/deep_prep_weights.pth"
CHALLENGER_PATH = "models/challenger_model.pt"
HOLDOUT_SEED = 12345
MIN_RELATIVE_IMPROVEMENT = 0.01  # require at least a 1% gain to avoid churn


def load(path):
    model = DeepPrepModel(num_restaurants=len(RESTAURANT_TO_IDX))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def write_output(deploy: bool):
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"deploy={'true' if deploy else 'false'}\n")


def main():
    if not os.path.exists(CHALLENGER_PATH):
        sys.exit("No challenger model found. Run scripts/retrain.py first.")

    holdout = build_dataset(4000, seed=HOLDOUT_SEED)
    challenger = evaluate(load(CHALLENGER_PATH), holdout)
    print("Challenger:", json.dumps(challenger))

    if not os.path.exists(CHAMPION_PATH):
        print("No champion found: challenger is promoted.")
        return write_output(True)

    champion = evaluate(load(CHAMPION_PATH), holdout)
    print("Champion:  ", json.dumps(champion))

    gain = (champion["pinball_p95"] - challenger["pinball_p95"]) / champion["pinball_p95"]
    deploy = gain >= MIN_RELATIVE_IMPROVEMENT
    print(f"P95 pinball improvement: {gain:+.1%} -> {'PROMOTE' if deploy else 'KEEP CHAMPION'}")
    write_output(deploy)


if __name__ == "__main__":
    main()
