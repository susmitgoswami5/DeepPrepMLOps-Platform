import pandas as pd
from datetime import datetime, timedelta
import random
import os

os.makedirs("data", exist_ok=True)
base_time = datetime.now() - timedelta(days=10)

data = {
    "restaurant_id": [f"R{str(i).zfill(3)}" for i in range(1, 21)],
    "event_timestamp": [base_time for _ in range(20)],
    "load_15m": [random.randint(1, 10) for _ in range(20)],
    "load_30m": [random.randint(5, 20) for _ in range(20)],
    "load_60m": [random.randint(10, 40) for _ in range(20)],
    "average_prep_time_ms": [random.uniform(5.0, 30.0) * 60 * 1000 for _ in range(20)]
}

df = pd.DataFrame(data)
df["restaurant_id"] = df["restaurant_id"].astype("string")
df.to_parquet("data/restaurant_stats.parquet", engine="pyarrow")
print("Wrote mock Parquet to data/restaurant_stats.parquet with clean PyArrow string types")
