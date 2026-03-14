#!/usr/bin/env python
"""
DeepPrep Nightly Retraining Script
===================================
This script is executed by the GitHub Actions CI/CD pipeline.
It re-trains the PyTorch DeepPrepModel on updated synthetic data,
saves the new weights as a challenger model, and logs training metrics.
"""
import os
import sys
import torch
import torch.nn as nn
import random
import json
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.architecture import DeepPrepModel


def generate_synthetic_training_data(num_samples=500):
    """
    Generate synthetic training data simulating restaurant order patterns.
    In production, this would pull from a data warehouse or Kafka topic replay.
    """
    data = []
    for _ in range(num_samples):
        hour = random.uniform(0, 24)
        day = random.randint(0, 6)
        
        import math
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        day_sin = math.sin(2 * math.pi * day / 7.0)
        day_cos = math.cos(2 * math.pi * day / 7.0)
        
        # Simulated item embeddings
        emb = [random.uniform(-1, 1) for _ in range(3)]
        
        # Load features
        load_15 = random.uniform(0, 50)
        load_30 = random.uniform(0, 60)
        load_60 = random.uniform(0, 80)
        
        # Queue engine features
        cook_utilization = random.uniform(0.1, 0.99)
        queue_wait_time_m = random.uniform(0, 15)
        orders_in_queue = random.uniform(0, 10)
        
        rest_idx = random.randint(0, 19)
        
        # Ground truth: a non-linear function mimicking real kitchen behavior
        base_time = 10 + (load_60 * 0.3) + (hour_sin * 5) + random.gauss(0, 3)
        p50_true = max(5, base_time)
        p90_true = max(7, base_time * 1.3)
        p95_true = max(8, base_time * 1.5)
        
        data.append({
            "rest_idx": rest_idx,
            "features": [hour_sin, hour_cos, day_sin, day_cos, emb[0], emb[1], emb[2],
                         cook_utilization, queue_wait_time_m, orders_in_queue],
            "seq": [[load_15], [load_30], [load_60]],
            "targets": [p50_true, p90_true, p95_true]
        })
    
    return data


def train_model(data, epochs=20, lr=0.001):
    """
    Train a fresh DeepPrepModel on the synthetic data.
    """
    model = DeepPrepModel(num_restaurants=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    model.train()
    
    print(f"🧠 Starting Nightly Retraining | {len(data)} samples | {epochs} epochs")
    print("=" * 60)
    
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(data)
        
        for sample in data:
            r_tensor = torch.tensor([sample["rest_idx"]], dtype=torch.long)
            w_tensor = torch.tensor([sample["features"]], dtype=torch.float32)
            s_tensor = torch.tensor([sample["seq"]], dtype=torch.float32)
            target = torch.tensor([sample["targets"]], dtype=torch.float32)
            
            optimizer.zero_grad()
            pred = model(r_tensor, w_tensor, s_tensor)
            loss = loss_fn(pred, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Avg MSE Loss: {avg_loss:.4f}")
    
    model.eval()
    print("=" * 60)
    print(f"✅ Training Complete | Final Loss: {avg_loss:.4f}")
    
    return model, avg_loss


def main():
    print("=" * 60)
    print(f"🔄 DEEPPREP NIGHTLY RETRAINING PIPELINE")
    print(f"   Timestamp: {datetime.now(timezone.utc).isoformat()}Z")
    print("=" * 60)
    
    # Step 1: Generate fresh data
    print("\n📊 Step 1: Generating synthetic training data...")
    data = generate_synthetic_training_data(num_samples=500)
    print(f"   Generated {len(data)} samples")
    
    # Step 2: Train new model
    print("\n🏋️ Step 2: Training challenger model...")
    new_model, final_loss = train_model(data, epochs=20, lr=0.001)
    
    # Step 3: Save challenger weights
    os.makedirs("models", exist_ok=True)
    challenger_path = "models/challenger_model.pt"
    torch.save(new_model.state_dict(), challenger_path)
    print(f"\n💾 Step 3: Challenger weights saved to {challenger_path}")
    
    # Step 4: Save training metadata
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(data),
        "epochs": 20,
        "final_mse_loss": round(final_loss, 4),
        "model_path": challenger_path
    }
    
    meta_path = "models/training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Step 4: Training metadata saved to {meta_path}")
    
    print(f"\n🎉 RETRAINING PIPELINE COMPLETE")
    print(f"   Next step: validate_model.py will compare this against the champion.")


if __name__ == "__main__":
    main()
