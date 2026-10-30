"""Synthetic order log generator for Food Preparation Time (FPT).

Each restaurant is simulated as a multi-cook, first-come-first-served kitchen:

* Orders arrive as a non-homogeneous Poisson process with lunch and dinner peaks.
* Each order needs a service time that depends on its items and on the kitchen's
  speed, with log-normal noise.
* An order waits until one of the restaurant's cooks is free.

The target, ``target_fpt_minutes``, is the time from order creation to completion
(waiting time + service time). Because waiting time comes from the queue, the
preparation time genuinely depends on how busy the kitchen is. This is what the
queueing-theory features are meant to capture.
"""
import heapq
import math
import random
from datetime import datetime, timedelta

import polars as pl

from src.data.restaurant_profiles import RESTAURANT_IDS, get_profile

MENU_ITEMS = [
    "Margherita Pizza", "Pepperoni Pizza", "Garlic Bread", "Cheese Burger",
    "Veggie Burger", "French Fries", "Tandoori Chicken", "Butter Chicken",
    "Naan", "Ice Cream", "Chocolate Cake", "Coke", "Sprite",
]

# Approximate hands-on preparation time (minutes) of the slowest item type.
INTRINSIC_PREP = {
    "Pizza": 15, "Burger": 10, "Fries": 5, "Bread": 5, "Chicken": 20,
    "Naan": 5, "Ice Cream": 2, "Cake": 3, "Coke": 1, "Sprite": 1,
}


def demand_multiplier(hour: float) -> float:
    """Relative order rate by hour of day: quiet nights, lunch and dinner peaks."""
    lunch = 1.3 * math.exp(-((hour - 13.0) ** 2) / (2 * 1.0 ** 2))
    dinner = 1.7 * math.exp(-((hour - 20.5) ** 2) / (2 * 1.3 ** 2))
    night = 0.15 if (hour < 7 or hour >= 23.5) else 0.5
    return night + lunch + dinner


def item_service_minutes(items: list[str], speed_factor: float, rng: random.Random) -> float:
    slowest = max(next((v for k, v in INTRINSIC_PREP.items() if k in it), 5) for it in items)
    base = slowest + 1.5 * (len(items) - 1)                 # extra items add assembly time
    return max(1.0, base * speed_factor * rng.lognormvariate(0.0, 0.25))


def _simulate_restaurant(rid: str, start: datetime, hours: float, rng: random.Random) -> list[dict]:
    prof = get_profile(rid)
    peak_rate = prof["base_orders_per_hour"] * 3.0          # upper bound for thinning
    cooks_free_at = [0.0] * prof["cooks"]                   # minutes since start
    heapq.heapify(cooks_free_at)
    t, out = 0.0, []
    while True:
        t += rng.expovariate(peak_rate) * 60.0              # candidate arrival (minutes)
        if t >= hours * 60:
            break
        hour = ((start.hour * 60 + t) / 60.0) % 24
        if rng.random() > prof["base_orders_per_hour"] * demand_multiplier(hour) / peak_rate:
            continue                                        # thinning step
        items = rng.choices(MENU_ITEMS, k=rng.choices([1, 2, 3, 4, 5], [0.4, 0.3, 0.15, 0.1, 0.05])[0])
        service = item_service_minutes(items, prof["speed_factor"], rng)
        free = heapq.heappop(cooks_free_at)
        begin = max(t, free)
        done = begin + service
        heapq.heappush(cooks_free_at, done)
        out.append({"restaurant_id": rid, "created_at": start + timedelta(minutes=t),
                    "items": items, "target_fpt_minutes": done - t})
    return out


def generate_mock_logs(n_orders: int = 5000, start_date_str: str = "2024-01-01",
                       seed: int | None = None) -> pl.DataFrame:
    """Generate ``n_orders`` simulated orders across all restaurants, sorted by time."""
    rng = random.Random(seed)
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    expected_per_day = sum(get_profile(r)["base_orders_per_hour"] for r in RESTAURANT_IDS) \
        * sum(demand_multiplier(h + 0.5) for h in range(24))
    hours = 24.0 * (n_orders / expected_per_day) * 1.25 + 24.0   # simulate a margin, then trim

    orders = []
    for rid in RESTAURANT_IDS:
        orders += _simulate_restaurant(rid, start, hours, rng)
    orders.sort(key=lambda o: o["created_at"])
    orders = orders[:n_orders]
    for i, o in enumerate(orders, 1):
        o["order_id"] = f"ORD{str(i).zfill(8)}"
    cols = ["order_id", "restaurant_id", "created_at", "items", "target_fpt_minutes"]
    return pl.DataFrame([{c: o[c] for c in cols} for o in orders])


if __name__ == "__main__":
    df = generate_mock_logs(10000, seed=0)
    print(df.head())
    print(df["target_fpt_minutes"].describe())
    df.write_parquet("mock_orders.parquet")
