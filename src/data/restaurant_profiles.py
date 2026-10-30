"""Static restaurant profiles shared by the simulator, the queue engine and the API.

In a real deployment these values (number of cooks on shift, typical service
speed, baseline demand) would come from restaurant onboarding data or a
feature store. Here they are generated deterministically from the restaurant ID
so that every component of the system sees the same kitchen.
"""
import random

RESTAURANT_IDS = [f"R{str(i).zfill(3)}" for i in range(1, 21)]

DEFAULT_PROFILE = {"cooks": 3, "speed_factor": 1.0, "base_orders_per_hour": 4.0}


def _make_profile(restaurant_id: str) -> dict:
    rng = random.Random(f"profile-{restaurant_id}")
    return {
        "cooks": rng.choice([2, 3, 3, 4]),               # parallel servers (c)
        "speed_factor": round(rng.uniform(0.8, 1.25), 3),  # >1 means a slower kitchen
        "base_orders_per_hour": round(rng.uniform(2.5, 5.5), 2),
    }


RESTAURANT_PROFILES = {rid: _make_profile(rid) for rid in RESTAURANT_IDS}

# Mean service time of one order by one cook, in minutes, before the speed factor.
MEAN_SERVICE_MINUTES = 14.0


def get_profile(restaurant_id: str) -> dict:
    return RESTAURANT_PROFILES.get(restaurant_id, DEFAULT_PROFILE)


def service_rate_per_cook(restaurant_id: str) -> float:
    """Orders one cook completes per hour (mu in the M/M/c model)."""
    return 60.0 / (MEAN_SERVICE_MINUTES * get_profile(restaurant_id)["speed_factor"])
