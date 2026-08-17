"""Shared internals for the raters package: types, exceptions, weight matrices."""
from __future__ import annotations

import numpy as np

Category = str | int


class UndefinedStatistic(ValueError):
    """Raised when a coefficient is mathematically undefined for the given data.

    This is the "no information" case — a single category present, or zero
    variance in a continuous measure — where chance-corrected agreement has no
    meaning (it is 0/0, not 1.0). It subclasses ValueError so that
    ``bootstrap_ci`` skips such resamples: a resample that collapses to one
    category must be *excluded* from the bootstrap distribution, not silently
    counted as perfect agreement (which would inflate the interval).
    """


def _weight_matrix(k: int, weights: str | None) -> np.ndarray:
    """Disagreement weights w[i, j] in [0, 1]; 0 on the diagonal.

    None      -> all-or-nothing (unweighted kappa)
    'linear'  -> |i - j| / (k - 1)
    'quadratic' -> ((i - j) / (k - 1)) ** 2
    """
    if k < 2:
        # One (or zero) categories: no disagreement is representable.
        return np.zeros((k, k), dtype=float)
    idx = np.arange(k, dtype=float)
    diff = np.abs(idx[:, None] - idx[None, :])
    if weights is None:
        return (diff > 0).astype(float)
    if weights == "linear":
        return diff / (k - 1)
    if weights == "quadratic":
        return (diff / (k - 1)) ** 2
    raise ValueError(f"unknown weights: {weights!r}")
