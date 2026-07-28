"""Inter-rater reliability statistics, implemented from first principles.

Every statistic here is hand-implemented (numpy only) so the math is inspectable
and testable against hand-computed reference values — the posture a regulated
clinical system needs: deterministic, versioned, and traceable to a reference.
"""
from __future__ import annotations

from typing import Callable, Sequence

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


# ---------------------------------------------------------------------------
# Basic agreement
# ---------------------------------------------------------------------------

def confusion_matrix(a: Sequence[Category], b: Sequence[Category],
                     categories: Sequence[Category]) -> np.ndarray:
    """Counts matrix M[i, j] = #items rater A labeled categories[i] and B labeled categories[j]."""
    if len(a) != len(b):
        raise ValueError("raters must score the same items")
    index = {c: i for i, c in enumerate(categories)}
    m = np.zeros((len(categories), len(categories)), dtype=float)
    for x, y in zip(a, b):
        if x not in index or y not in index:
            raise ValueError(f"label not in categories: {x!r} or {y!r}")
        m[index[x], index[y]] += 1
    return m


def percent_agreement(a: Sequence[Category], b: Sequence[Category]) -> float:
    """Raw agreement — simple, but blind to chance. Report it, never rely on it alone."""
    if len(a) != len(b):
        raise ValueError("raters must score the same items")
    if not a:
        raise ValueError("empty input")
    return sum(x == y for x, y in zip(a, b)) / len(a)


# ---------------------------------------------------------------------------
# Cohen's kappa (2 raters), unweighted or weighted (ordinal categories)
# ---------------------------------------------------------------------------

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


def cohen_kappa(a: Sequence[Category], b: Sequence[Category],
                categories: Sequence[Category], weights: str | None = None) -> float:
    """Chance-corrected agreement for two raters.

    Unweighted for nominal labels; 'linear'/'quadratic' for ordinal categories
    (e.g. sleep stages), where near-misses (N2 vs N3) should cost less than far
    misses (N2 vs REM). `categories` order defines ordinal distance.
    """
    m = confusion_matrix(a, b, categories)
    n = m.sum()
    if n == 0:
        raise ValueError("empty input")
    p = m / n
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    expected = np.outer(row, col)
    w = _weight_matrix(len(categories), weights)
    observed_disagreement = (w * p).sum()
    expected_disagreement = (w * expected).sum()
    if expected_disagreement == 0:
        # No information: both raters used a single category -> kappa is 0/0,
        # not perfect agreement. Undefined.
        raise UndefinedStatistic("kappa undefined: only one category present")
    return 1.0 - observed_disagreement / expected_disagreement


# ---------------------------------------------------------------------------
# Fleiss' kappa (3+ raters, counts table)
# ---------------------------------------------------------------------------

def fleiss_kappa(counts: np.ndarray) -> float:
    """Chance-corrected agreement for many raters.

    `counts` is an (n_items x n_categories) table where counts[i, c] is how many
    raters put item i in category c; every row must sum to the same rater count.
    """
    counts = np.asarray(counts, dtype=float)
    n_items, _ = counts.shape
    raters = counts.sum(axis=1)
    if n_items == 0:
        raise ValueError("empty input")
    if not np.all(raters == raters[0]):
        raise ValueError("all items must be scored by the same number of raters")
    m = raters[0]
    if m < 2:
        raise ValueError("need at least 2 raters")
    p_i = ((counts ** 2).sum(axis=1) - m) / (m * (m - 1))
    p_bar = p_i.mean()
    p_c = (counts.sum(axis=0) / (n_items * m)) ** 2
    p_e = p_c.sum()
    if p_e == 1.0:
        # All ratings in a single category -> undefined, not perfect agreement.
        raise UndefinedStatistic("Fleiss' kappa undefined: only one category present")
    return (p_bar - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Krippendorff's alpha (any #raters, missing data)
# ---------------------------------------------------------------------------

def krippendorff_alpha(data: Sequence[Sequence[Category | None]],
                       level: str = "nominal",
                       categories: Sequence[Category] | None = None) -> float:
    """The most general agreement coefficient.

    `data` is raters x units; None marks a missing score, and units with fewer
    than two scores are ignored. `level` is 'nominal' (0/1 distance) or
    'interval' (squared distance on category index — also the standard practical
    stand-in for ordinal scales like sleep stages).

    For 'interval'/ordinal data, pass `categories` in scale order. The distance
    between two labels is the gap between their *positions in `categories`*, so
    the full scale (including any category absent from the data) must be supplied
    or the spacing collapses. When `categories` is omitted the scale is inferred
    from the sorted distinct labels, which is only safe for nominal use.
    """
    data = [list(r) for r in data]
    n_units = len(data[0])
    if any(len(r) != n_units for r in data):
        raise ValueError("all raters must cover the same units")

    if categories is not None:
        # Use the FULL provided scale so ordinal/interval spacing is preserved
        # even when a category never appears in the data (absent categories just
        # carry zero mass and do not affect the result).
        values: list[Category] = list(categories)
        present = {v for r in data for v in r if v is not None}
        unknown = present - set(values)
        if unknown:
            raise ValueError(f"data contains labels not in categories: {sorted(map(str, unknown))}")
    else:
        values = sorted(
            {v for r in data for v in r if v is not None},
            key=lambda v: (str(type(v)), v),
        )
    index = {v: i for i, v in enumerate(values)}
    k = len(values)
    if k < 2:
        # Only one distinct value across all raters -> no variation to assess.
        raise UndefinedStatistic("alpha undefined: only one category present")

    if level == "nominal":
        delta = _weight_matrix(k, None)
    elif level == "interval":
        idx = np.arange(k, dtype=float)
        delta = (idx[:, None] - idx[None, :]) ** 2
    else:
        raise ValueError(f"unknown level: {level!r}")

    # Coincidence matrix over pairable values within units.
    o = np.zeros((k, k), dtype=float)
    for u in range(n_units):
        unit_vals = [r[u] for r in data if r[u] is not None]
        m_u = len(unit_vals)
        if m_u < 2:
            continue
        for i, vi in enumerate(unit_vals):
            for j, vj in enumerate(unit_vals):
                if i != j:
                    o[index[vi], index[vj]] += 1.0 / (m_u - 1)
    n = o.sum()
    if n <= 1:
        raise ValueError("not enough pairable values")
    marg = o.sum(axis=1)
    d_o = (delta * o).sum()
    d_e = (delta * np.outer(marg, marg)).sum() / (n - 1)
    if d_e == 0:
        # All pairable values fell in one category -> undefined.
        raise UndefinedStatistic("alpha undefined: no observed disagreement is possible")
    return 1.0 - d_o / d_e


# ---------------------------------------------------------------------------
# ICC — intraclass correlation for continuous measures (e.g. latencies, AHI)
# ---------------------------------------------------------------------------

def icc(data: np.ndarray, form: str = "2,1") -> float:
    """Two-way ICC from ANOVA mean squares. `data` is units x raters (complete).

    form '2,1': two-way random, absolute agreement, single rater — the usual
                choice when raters are interchangeable and calibration matters
                (a constant rater bias LOWERS it).
    form '3,1': two-way mixed, consistency — ignores constant rater bias.
    Choosing the wrong form is the classic ICC mistake; both are provided so the
    difference is testable.
    """
    x = np.asarray(data, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("need at least 2 units and 2 raters")
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    if ss_total == 0:
        # Every value identical: no variance to partition -> ICC is 0/0.
        raise UndefinedStatistic("ICC undefined: zero total variance")
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    if form == "2,1":
        denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    elif form == "3,1":
        denom = msr + (k - 1) * mse
    else:
        raise ValueError(f"unknown form: {form!r}")
    if denom == 0:
        # No between-subject variance -> consistency/agreement is undefined,
        # not perfect. (e.g. all subjects identical, raters differ by a constant.)
        raise UndefinedStatistic("ICC undefined: no between-subject variance")
    return float((msr - mse) / denom)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (resample units)
# ---------------------------------------------------------------------------

def bootstrap_ci(stat: Callable[[list[int]], float], n_units: int,
                 n_boot: int = 1000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for any unit-level statistic.

    `stat` receives a list of unit indices (sampled with replacement) and returns
    the statistic computed on that resample. A point estimate without a CI is
    noise wearing a suit — a kappa from 50 epochs can swing wildly.
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
