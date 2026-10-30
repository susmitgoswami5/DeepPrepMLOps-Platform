import math

import pytest

from src.features.queue_engine import calculate_mmc_metrics, queue_features


def erlang_c_reference(lam, mu, c):
    """Erlang C via the Erlang B recursion (independent of the implementation)."""
    a = lam / mu
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    rho = a / c
    return b / (1 - rho * (1 - b))


@pytest.mark.parametrize("lam,mu,c", [(2.0, 1.5, 2), (10.0, 4.0, 3), (3.0, 5.0, 1)])
def test_wait_time_matches_erlang_c(lam, mu, c):
    rho, wq_min, lq = calculate_mmc_metrics(lam, mu, c)
    pw = erlang_c_reference(lam, mu, c)
    assert rho == pytest.approx(lam / (c * mu))
    assert wq_min == pytest.approx(pw / (c * mu - lam) * 60.0)
    assert lq == pytest.approx(lam * wq_min / 60.0)  # Little's law


def test_mm1_closed_form():
    lam, mu = 3.0, 5.0
    _, wq_min, _ = calculate_mmc_metrics(lam, mu, 1)
    assert wq_min == pytest.approx(lam / (mu * (mu - lam)) * 60.0)


def test_unstable_queue_is_capped_and_finite():
    rho, wq, lq = calculate_mmc_metrics(100.0, 1.0, 2)
    assert rho == pytest.approx(0.99)
    assert math.isfinite(wq) and math.isfinite(lq)


def test_zero_load_has_no_queue():
    assert calculate_mmc_metrics(0.0, 4.0, 3) == (0.0, 0.0, 0.0)


def test_more_load_means_longer_wait():
    waits = [queue_features(l, "R001")[1] for l in (1, 5, 10, 15)]
    assert waits == sorted(waits)
