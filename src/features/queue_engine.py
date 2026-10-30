"""M/M/c (Erlang C) queueing features for kitchen congestion."""
import math

import polars as pl

from src.data.restaurant_profiles import get_profile, service_rate_per_cook

MAX_UTILIZATION = 0.99  # the analytic queue is unstable at rho >= 1, so cap it


def calculate_mmc_metrics(arrival_rate_lambda: float, service_rate_mu: float, num_servers_c: int):
    """Analytic M/M/c metrics.

    Args:
        arrival_rate_lambda: orders arriving per hour.
        service_rate_mu: orders one cook completes per hour.
        num_servers_c: number of cooks working in parallel.

    Returns:
        (cook_utilization, expected_queue_wait_minutes, expected_orders_in_queue)
    """
    if num_servers_c <= 0 or service_rate_mu <= 0 or arrival_rate_lambda <= 0:
        return 0.0, 0.0, 0.0

    c = num_servers_c
    rho = min(arrival_rate_lambda / (c * service_rate_mu), MAX_UTILIZATION)
    a = c * rho  # offered load in Erlangs (after capping)

    p0_sum = sum(a ** n / math.factorial(n) for n in range(c))
    p0_c_term = (a ** c / math.factorial(c)) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (p0_sum + p0_c_term)

    erlang_c = p0_c_term * p0                       # probability an order has to wait
    wq_hours = erlang_c / (c * service_rate_mu * (1.0 - rho))
    lq = erlang_c * rho / (1.0 - rho)               # mean queue length (consistent with capped rho)
    return rho, wq_hours * 60.0, lq


def queue_features(load_60m: float, restaurant_id: str):
    """Queue features for one restaurant, using the last hour's order count as lambda."""
    return calculate_mmc_metrics(
        arrival_rate_lambda=float(load_60m),
        service_rate_mu=service_rate_per_cook(restaurant_id),
        num_servers_c=get_profile(restaurant_id)["cooks"],
    )


def append_queue_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cook_utilization, queue_wait_time_m and orders_in_queue columns."""
    rows = [queue_features(l, r) for l, r in zip(df["load_60m"].to_list(), df["restaurant_id"].to_list())]
    util, wait, lq = zip(*rows) if rows else ([], [], [])
    return df.with_columns(
        pl.Series("cook_utilization", util, dtype=pl.Float64),
        pl.Series("queue_wait_time_m", wait, dtype=pl.Float64),
        pl.Series("orders_in_queue", lq, dtype=pl.Float64),
    )
