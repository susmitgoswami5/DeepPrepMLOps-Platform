from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# Data Sources
# In production, this would point to Parquet files or Snowflake/BigQuery tables
# populated by our Kafka consumers. For now, it points to local parquet files.
restaurant_stats_source = FileSource(
    name="restaurant_hourly_stats",
    path="data/restaurant_stats.parquet",
    timestamp_field="event_timestamp",
)

from feast import ValueType

# Entities
restaurant = Entity(
    name="restaurant_id",
    join_keys=["restaurant_id"],
    description="Restaurant identifier",
)

# Feature Views
restaurant_features_view = FeatureView(
    name="restaurant_features",
    entities=[restaurant],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="load_15m", dtype=Int64),
        Field(name="load_30m", dtype=Int64),
        Field(name="load_60m", dtype=Int64),
        Field(name="average_prep_time_ms", dtype=Float32),
    ],
    source=restaurant_stats_source,
    tags={"team": "logistics_ml"},
)
