import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import polars as pl

# Use relative imports when running from root
from src.data.mock_generator import generate_mock_logs
from src.features.engineering import engineer_features
from src.model.architecture import DeepPrepModel, quantile_loss
from src.training.dataset import DeepPrepDataset, prepare_data

from feast import FeatureStore

def train_model(epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    print("Generating Mock Logs...")
    # 1. Generate Fake Data
    raw_df = generate_mock_logs(2000)
    
    # 1.5 Feast Offline Enrichment
    print("Enriching via Feast Offline Store...")
    fs = FeatureStore(repo_path="src/feature_store")
    # In real world, entity_df contains just ID and Timestamp. We mock extract it here.
    entity_df = raw_df.select(["restaurant_id", "created_at"]).rename({"created_at": "event_timestamp"}).to_pandas()
    try:
        enriched_df = fs.get_historical_features(
            entity_df=entity_df,
            features=[
                "restaurant_features:load_15m",
                "restaurant_features:load_30m",
                "restaurant_features:load_60m"
            ]
        ).to_df()
        print(f"Feast Enriched Columns: {enriched_df.columns}")
    except Exception as e:
        print(f"Feast Offline Warning (skipping strict check for local dev): {e}")
    
    print("Engineering Features...")
    # 2. Engineer Final Pipeline Features
    feat_df = engineer_features(raw_df)
    
    # 3. Setup Dataset
    train_df, val_df, r2i = prepare_data(feat_df)
    num_restaurants = len(r2i)
    
    train_dataset = DeepPrepDataset(train_df, r2i)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = DeepPrepDataset(val_df, r2i)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Init Model
    model = DeepPrepModel(num_restaurants=num_restaurants)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    quantiles = [0.5, 0.9, 0.95]
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            r_idx, wide_f, seq_f, target = batch
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(r_idx, wide_f, seq_f)
            # Match target sequence shape
            target = target.unsqueeze(-1)
            
            # Loss
            loss = quantile_loss(preds, target, quantiles)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(r_idx)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                r_idx, wide_f, seq_f, target = batch
                preds = model(r_idx, wide_f, seq_f)
                target = target.unsqueeze(-1)
                loss = quantile_loss(preds, target, quantiles)
                val_loss += loss.item() * len(r_idx)
                
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    print("Training Complete! Saving Model...")
    os.makedirs("models", exist_ok=True)
    
    # Save model weights and the restaurant mapping
    torch.save(model.state_dict(), "models/deep_prep_weights.pth")
    # In a real system, we'd save `r2i` to disk (e.g. JSON/Pickle) to guarantee inference mapping
    import json
    with open("models/restaurant_mapping.json", "w") as f:
        json.dump(r2i, f)
        
    print("Model and Mapping Saved to `models/`")
    return model

if __name__ == "__main__":
    train_model(epochs=5)
