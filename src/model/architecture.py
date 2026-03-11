import torch
import torch.nn as nn

class WideSide(nn.Module):
    def __init__(self, num_restaurants: int, restaurant_emb_dim: int = 8, other_wide_features: int = 6):
        """
        Wide side for memorizing specific restaurant characteristics and handling simple linear features.
        other_wide_features includes: hour_sin, hour_cos, day_sin, day_cos, and 3 item embedding dims.
        """
        super().__init__()
        self.restaurant_embedding = nn.Embedding(num_embeddings=num_restaurants, embedding_dim=restaurant_emb_dim)
        # We also pass other features (cyclic time, averaged item embeddings)
        self.linear = nn.Linear(restaurant_emb_dim + other_wide_features, 16)
        
    def forward(self, restaurant_idx: torch.Tensor, wide_features: torch.Tensor) -> torch.Tensor:
        # restaurant_idx: (batch_size)
        # wide_features: (batch_size, other_wide_features)
        r_emb = self.restaurant_embedding(restaurant_idx)
        wide_input = torch.cat([r_emb, wide_features], dim=-1)
        return torch.relu(self.linear(wide_input))


class DeepSide(nn.Module):
    def __init__(self, seq_feature_dim: int = 1, hidden_size: int = 32, num_layers: int = 1):
        """
        Deep side using LSTM over the recent kitchen load features.
        We treat the [15m, 30m, 60m] loads as a sequence of length 3 for the LSTM to interpret trends.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_feature_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 16)
        
    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        # seq_features: (batch_size, seq_len=3, seq_feature_dim=1)
        # We are padding the rolling features [load_15m, load_30m, load_60m] as a sequence dimension
        out, (hn, cn) = self.lstm(seq_features)
        # Take the last hidden state
        last_out = hn[-1] # Shape: (batch_size, hidden_size)
        return torch.relu(self.fc(last_out))


class DeepPrepModel(nn.Module):
    def __init__(self, num_restaurants: int):
        super().__init__()
        # 4 cyclic + 3 emb = 7 features (weights expect 15: 8 emb_dim + 7)
        self.wide = WideSide(num_restaurants=num_restaurants, restaurant_emb_dim=8, other_wide_features=7) 
        self.deep = DeepSide(seq_feature_dim=1, hidden_size=32)
        
        # Combine the wide and deep outputs
        self.combiner = nn.Sequential(
            nn.Linear(16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Quantile Predictor (Output heads for 50th, 90th, 95th percentiles)
        # Size 3 output representing [p50, p90, p95]
        self.output_head = nn.Linear(16, 3)
        
    def forward(self, restaurant_idx: torch.Tensor, wide_features: torch.Tensor, seq_features: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(restaurant_idx, wide_features)
        deep_out = self.deep(seq_features)
        
        combined = torch.cat([wide_out, deep_out], dim=-1)
        features = self.combiner(combined)
        
        # The output dimensions are (batch_size, 3) corresponding to the quantiles
        return self.output_head(features)

# Loss Function
def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list[float] = [0.5, 0.9, 0.95]) -> torch.Tensor:
    """
    Compute Quantile Loss (Pinball Loss).
    preds: (batch_size, num_quantiles)
    target: (batch_size, 1)
    """
    assert preds.size(1) == len(quantiles), "Number of predictions must match number of quantiles"
    
    losses = []
    for i, q in enumerate(quantiles):
        pred_q = preds[:, i:i+1] # Shape: (batch_size, 1)
        errors = target - pred_q
        # Pinball loss logic: max(q * e, (q - 1) * e)
        loss_q = torch.max(q * errors, (q - 1) * errors)
        losses.append(loss_q)
        
    # Sum over quantiles and mean over batch
    return torch.cat(losses, dim=1).sum(dim=1).mean()

if __name__ == "__main__":
    # Test shapes
    batch_size = 4
    rest_idx = torch.tensor([0, 15, 2, 7])
    wide_f = torch.randn(batch_size, 10) # 4 cyclic + 3 emb + 3 queue
    seq_f = torch.randn(batch_size, 3, 1) # [15m, 30m, 60m] loads as sequence
    
    model = DeepPrepModel(num_restaurants=20)
    preds = model(rest_idx, wide_f, seq_f)
    print(f"Predictions shape: {preds.shape} (Expected: {batch_size}, 3)")
    
    target = torch.randn(batch_size, 1)
    loss = quantile_loss(preds, target)
    print(f"Computed Quantile Loss: {loss.item()}")
