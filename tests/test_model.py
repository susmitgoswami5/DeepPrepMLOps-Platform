import torch

from src.model.architecture import DeepPrepModel, quantile_loss
from src.training.core import build_dataset, chronological_split, evaluate, train_model


def test_output_shape():
    m = DeepPrepModel(num_restaurants=20)
    out = m(torch.tensor([0, 5]), torch.randn(2, 10), torch.rand(2, 3, 1) * 10)
    assert out.shape == (2, 3)


def test_quantile_loss_does_not_broadcast_targets():
    """Regression test: a (B,1,1) target used to broadcast into a (B,B,1) loss."""
    preds = torch.tensor([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]])
    target = torch.tensor([[1.0], [10.0]])
    a = quantile_loss(preds, target)
    b = quantile_loss(preds, target.unsqueeze(-1))
    assert torch.isclose(a, b)
    assert a.item() < 3.0


def test_training_beats_constant_prediction():
    df = build_dataset(3000, seed=0)
    tr, te = chronological_split(df)
    model = train_model(tr, epochs=8, seed=0)
    y = te["target_fpt_minutes"].to_numpy()
    import numpy as np
    constant_mae = float(np.mean(np.abs(y - np.median(tr["target_fpt_minutes"].to_numpy()))))
    assert evaluate(model, te)["mae_p50_min"] < constant_mae
