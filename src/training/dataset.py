import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np

class DeepPrepDataset(Dataset):
    def __init__(self, df: pl.DataFrame, restaurant_to_idx: dict[str, int]):
        """
        PyTorch Dataset for DeepPrep Model.
        Expects a DataFrame that has already passed through `engineer_features`.
        """
        self.df = df
        
        # Static mappings
        self.restaurant_to_idx = restaurant_to_idx
        
        # Extract necessary columns as numpy arrays for fast indexing
        self.restaurant_ids = self.df["restaurant_id"].to_list()
        
        # Wide features
        self.wide_features = self.df.select(["hour_sin", "hour_cos", "day_sin", "day_cos", "embed_0", "embed_1", "embed_2", "cook_utilization", "queue_wait_time_m", "orders_in_queue"]).to_numpy().astype(np.float32)
        
        # Deep sequence features (rolling loads)
        self.seq_features = self.df.select(["load_15m", "load_30m", "load_60m"]).to_numpy().astype(np.float32)
        
        # Targets
        self.targets = self.df.select("target_fpt_minutes").to_numpy().astype(np.float32)
        
    def __len__(self):
        return self.df.height
    
    def __getitem__(self, idx):
        # 1. Restaurant ID mapping
        rest_id = self.restaurant_ids[idx]
        rest_idx = self.restaurant_to_idx.get(rest_id, 0) # default to 0 if unseen
        
        # 2. Wide features
        wide_f = torch.tensor(self.wide_features[idx])
        
        # 3. Seq features - Needs to be shape (seq_len=3, feature_dim=1)
        seq_f = torch.tensor(self.seq_features[idx]).unsqueeze(-1)
        
        # 4. Target
        target = torch.tensor(self.targets[idx])
        
        return torch.tensor(rest_idx, dtype=torch.long), wide_f, seq_f, target

def prepare_data(df: pl.DataFrame):
     # Create mapping
    unique_restaurants = df["restaurant_id"].unique().to_list()
    rest_to_idx = {r: i for i, r in enumerate(unique_restaurants)}
    
    # Split
    train_df = df.sample(fraction=0.8, seed=42)
    val_df = df.filter(~pl.col("order_id").is_in(train_df["order_id"]))
    
    return train_df, val_df, rest_to_idx

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from data.mock_generator import generate_mock_logs
    from src.features.engineering import engineer_features
    
    raw = generate_mock_logs(100)
    feat = engineer_features(raw)
    
    train_df, val_df, r2i = prepare_data(feat)
    
    dataset = DeepPrepDataset(train_df, r2i)
    r_idx, w_f, s_f, t = dataset[0]
    
    print("Sample Output:")
    print("Rest IDX:", r_idx)
    print("Wide F:", w_f.shape)
    print("Seq F:", s_f.shape)
    print("Target F:", t.shape)
