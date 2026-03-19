#!/usr/bin/env python
"""
DeepPrep Model Validation Gate
================================
This script compares the newly trained Challenger model against
the current Champion to decide whether auto-deployment is safe.

The gate passes ONLY if the Challenger's loss is LOWER than
the Champion's loss on a shared holdout set.
"""
import os
import sys
import torch
import torch.nn as nn
import random
import json
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.architecture import DeepPrepModel


def generate_holdout_data(num_samples=100):
    """
    Generate a fixed holdout set for fair comparison.
    Uses a fixed seed for deterministic evaluation.
    """
    random.seed(42)
    data = []
    for _ in range(num_samples):
        hour = random.uniform(0, 24)
        day = random.randint(0, 6)
        
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        day_sin = math.sin(2 * math.pi * day / 7.0)
        day_cos = math.cos(2 * math.pi * day / 7.0)
        
        emb = [random.uniform(-1, 1) for _ in range(3)]
        load_15 = random.uniform(0, 50)
        load_30 = random.uniform(0, 60)
        load_60 = random.uniform(0, 80)
        
        # Queue engine features
        cook_utilization = random.uniform(0.1, 0.99)
        queue_wait_time_m = random.uniform(0, 15)
        orders_in_queue = random.uniform(0, 10)
        
        rest_idx = random.randint(0, 19)
        
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


def evaluate_model(model, holdout):
    """Compute average MSE on the holdout set."""
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for sample in holdout:
            r = torch.tensor([sample["rest_idx"]], dtype=torch.long)
            w = torch.tensor([sample["features"]], dtype=torch.float32)
            s = torch.tensor([sample["seq"]], dtype=torch.float32)
            t = torch.tensor([sample["targets"]], dtype=torch.float32)
            
            pred = model(r, w, s)
            total_loss += loss_fn(pred, t).item()
    
    return total_loss / len(holdout)


def main():
    print("=" * 60)
    print("🧪 DEEPPREP MODEL VALIDATION GATE")
    print("=" * 60)
    
    champion_path = "models/deep_prep_weights.pth"
    challenger_path = "models/challenger_model.pt"
    
    if not os.path.exists(challenger_path):
        print("❌ FAIL: No challenger model found. Run retrain.py first.")
        sys.exit(1)
    
    # Load holdout data
    holdout = generate_holdout_data(num_samples=100)
    print(f"📊 Holdout set: {len(holdout)} samples (seed=42)")
    
    # Load Champion
    champion = DeepPrepModel(num_restaurants=20)
    if os.path.exists(champion_path):
        champion.load_state_dict(torch.load(champion_path, map_location="cpu", weights_only=True))
        print("🏆 Champion model loaded")
    else:
        print("⚠️ No existing champion — challenger auto-wins")
    
    # Load Challenger
    challenger = DeepPrepModel(num_restaurants=20)
    challenger.load_state_dict(torch.load(challenger_path, map_location="cpu", weights_only=True))
    print("🥊 Challenger model loaded")
    
    # Evaluate both
    champ_loss = evaluate_model(champion, holdout)
    chall_loss = evaluate_model(challenger, holdout)
    
    print(f"\n📈 Results:")
    print(f"   Champion MSE:   {champ_loss:.4f}")
    print(f"   Challenger MSE: {chall_loss:.4f}")
    print(f"   Delta:          {champ_loss - chall_loss:.4f}")
    
    # Decision gate
    env_file = os.getenv("GITHUB_OUTPUT")
    
    if chall_loss < champ_loss:
        improvement = ((champ_loss - chall_loss) / champ_loss) * 100
        print(f"\n✅ GATE PASSED: Challenger is {improvement:.1f}% better!")
        print("   -> Safe to promote to Champion.")
        if env_file:
            with open(env_file, "a") as f:
                f.write("deploy=true\n")
        sys.exit(0)
    else:
        degradation = ((chall_loss - champ_loss) / champ_loss) * 100
        print(f"\n❌ GATE FAILED: Challenger is {degradation:.1f}% worse.")
        print("   -> Keeping current Champion. No deployment.")
        if env_file:
            with open(env_file, "a") as f:
                f.write("deploy=false\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
