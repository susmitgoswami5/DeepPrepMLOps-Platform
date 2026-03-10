import polars as pl
import numpy as np
from datetime import timedelta

# Import the new queue simulation engine
from src.features.queue_engine import append_queue_features

# A simple mock embedding dictionary for items
# In real life, Word2Vec or BERT embeddings would be pre-calculated and loaded.
MOCK_EMBEDDINGS = {
    "Pizza": [0.9, 0.1, 0.0],
    "Burger": [0.8, 0.2, 0.1],
    "Fries": [0.1, 0.8, 0.1],
    "Bread": [0.2, 0.7, 0.1],
    "Chicken": [0.3, 0.1, 0.9],
    "Naan": [0.2, 0.6, 0.2],
    "Ice Cream": [0.0, 0.0, 0.1],
    "Cake": [0.0, 0.1, 0.1],
    "Soda": [0.0, 0.9, 0.9],
}

def get_mock_embedding(item_name: str) -> list[float]:
    """Return a mock embedding for an item based on keywords."""
    for key, emb in MOCK_EMBEDDINGS.items():
        if key in item_name:
            return emb
    return [0.0, 0.0, 0.0]

def add_cyclic_features(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Add cyclic sine/cosine transforms for hour and day."""
    return df.with_columns([
        (pl.col(time_col).dt.hour() * (2 * np.pi / 24)).sin().alias("hour_sin"),
        (pl.col(time_col).dt.hour() * (2 * np.pi / 24)).cos().alias("hour_cos"),
        (pl.col(time_col).dt.weekday() * (2 * np.pi / 7)).sin().alias("day_sin"),
        (pl.col(time_col).dt.weekday() * (2 * np.pi / 7)).cos().alias("day_cos")
    ])

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """End-to-end feature engineering pipeline."""
    # Ensure creation time is sorted
    df = df.sort("created_at")
    
    # 1. Cyclic Features
    df = add_cyclic_features(df, "created_at")
    
    # 2. Kitchen Load (Rolling window of active orders)
    # We will use Polars rolling aggregation over the created_at column by restaurant_id
    # We approximate "active orders" by just counting orders received in the last N mins
    
    # Convert 'created_at' to be part of the index or just group by_dynamic
    
    # To use rolling_by or sort by 'created_at' and do rolling counts
    # We want: for each row, count how many orders for the same restaurant were created in the last 15, 30, 60 mins.
    df = df.sort(["restaurant_id", "created_at"])
    
    # Count orders in rolling windows
    df = df.rolling(
        index_column="created_at",
        period="15m",
        group_by="restaurant_id"
    ).agg(
        pl.len().alias("load_15m")
    ) \
    .join(df, on=["restaurant_id", "created_at"], how="inner")
    
    df = df.sort(["restaurant_id", "created_at"])
    df = df.rolling(
        index_column="created_at",
        period="30m",
        group_by="restaurant_id"
    ).agg(
        pl.len().alias("load_30m")
    ).join(df, on=["restaurant_id", "created_at"], how="inner")
    
    df = df.sort(["restaurant_id", "created_at"])
    df = df.rolling(
        index_column="created_at",
        period="60m",
        group_by="restaurant_id"
    ).agg(
        pl.len().alias("load_60m")
    ).join(df, on=["restaurant_id", "created_at"], how="inner")

    # 3. Item Embeddings
    # Since items is a list of strings, we want to map each to its embedding and average them.
    # We can do this with a python UDF or mapping, since it's a list.
    def get_avg_embedding(items: list[str]) -> list[float]:
        embs = [get_mock_embedding(i) for i in items]
        if not embs:
            return [0.0, 0.0, 0.0]
        return np.mean(embs, axis=0).tolist()
    
    df = df.with_columns(
        pl.col("items").map_elements(get_avg_embedding, return_dtype=pl.List(pl.Float64)).alias("item_embeddings")
    )
    
    # Flatten the embeddings into 3 separate columns for easier Wide&Deep processing later
    df = df.with_columns([
        pl.col("item_embeddings").list.get(0).alias("embed_0"),
        pl.col("item_embeddings").list.get(1).alias("embed_1"),
        pl.col("item_embeddings").list.get(2).alias("embed_2"),
    ]).drop("item_embeddings")
    
    # 4. Integrate Queue Engine Features
    df = append_queue_features(df)
    
    return df

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.mock_generator import generate_mock_logs
    df_raw = generate_mock_logs(100)
    df_feat = engineer_features(df_raw)
    print("Engineered Features:")
    print(df_feat.head())
    print("\nColumns:", df_feat.columns)
