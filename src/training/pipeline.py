"""Training pipeline: simulate orders, (optionally) enrich via Feast, train, save."""
import json
import os

from src.training.core import (RESTAURANT_TO_IDX, build_dataset, chronological_split,
                               evaluate, train_model as fit)

MODEL_DIR = "models"


def _try_feast_enrichment(df):
    """Demonstrates an offline Feast lookup; skipped if the feature store isn't set up."""
    try:
        from feast import FeatureStore
        fs = FeatureStore(repo_path="src/feature_store")
        entity_df = (df.select(["restaurant_id", "created_at"])
                       .rename({"created_at": "event_timestamp"}).to_pandas())
        enriched = fs.get_historical_features(
            entity_df=entity_df,
            features=["restaurant_features:load_15m", "restaurant_features:load_30m",
                      "restaurant_features:load_60m"],
        ).to_df()
        print(f"Feast enrichment columns: {list(enriched.columns)}")
    except Exception as e:  # noqa: BLE001 - optional in local development
        print(f"Feast offline store unavailable, continuing without it: {e}")


def train_model(epochs: int = 40, n_orders: int = 15000, seed: int = 0):
    print("Simulating orders and engineering features...")
    df = build_dataset(n_orders, seed=seed)
    _try_feast_enrichment(df)

    train_df, val_df = chronological_split(df)
    print(f"Training on {train_df.height} orders, validating on {val_df.height} later orders")
    model = fit(train_df, epochs=epochs, seed=seed, verbose=True)
    print("Validation metrics:", json.dumps(evaluate(model, val_df), indent=1))

    import torch
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "deep_prep_weights.pth"))
    with open(os.path.join(MODEL_DIR, "restaurant_mapping.json"), "w") as f:
        json.dump(RESTAURANT_TO_IDX, f, indent=1)
    print(f"Model and restaurant mapping saved to `{MODEL_DIR}/`")
    return model


if __name__ == "__main__":
    train_model()
