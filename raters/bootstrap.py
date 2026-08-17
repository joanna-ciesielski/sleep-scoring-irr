"""Bootstrap confidence intervals (resample units)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ._core import UndefinedStatistic


def bootstrap_ci(stat: Callable[[list[int]], float], n_units: int,
                 n_boot: int = 1000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for any unit-level statistic.

    `stat` receives a list of unit indices (sampled with replacement) and returns
    the statistic computed on that resample. A point estimate without a CI is
    noise wearing a suit — a kappa from 50 items can swing wildly.
    """
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units).tolist()
        try:
            boots.append(stat(idx))
        except UndefinedStatistic:
            # Degenerate resample where the statistic is undefined (e.g. it
            # collapsed to a single category). Excluded from the distribution —
            # counting it as 1.0 would inflate the interval. Only this specific
            # exception is swallowed, so a genuine bug in `stat` still surfaces.
            continue
    if not boots:
        raise ValueError("all bootstrap resamples degenerate")
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
