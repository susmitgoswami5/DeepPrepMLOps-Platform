import numpy as np

from src.data.mock_generator import generate_mock_logs
from src.features.engineering import engineer_features


def test_schema_order_and_size():
    df = generate_mock_logs(500, seed=1)
    assert df.height == 500
    assert df.columns == ["order_id", "restaurant_id", "created_at", "items", "target_fpt_minutes"]
    assert df["created_at"].is_sorted()
    assert (df["target_fpt_minutes"] > 0).all()


def test_seed_is_reproducible():
    a, b = generate_mock_logs(300, seed=7), generate_mock_logs(300, seed=7)
    assert a.equals(b)


def test_prep_time_depends_on_kitchen_load():
    """The point of the simulator: busier kitchens should mean longer waits."""
    df = engineer_features(generate_mock_logs(4000, seed=3))
    r = np.corrcoef(df["load_60m"].to_numpy(), df["target_fpt_minutes"].to_numpy())[0, 1]
    assert r > 0.2
