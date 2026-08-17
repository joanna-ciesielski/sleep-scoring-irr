"""Intra-rater reliability: does one rater agree with itself on repeat measurement?

Inter-rater reliability asks whether two raters agree. Intra-rater (test-retest)
reliability asks whether ONE rater, shown the same items again, reproduces its
own answers. That is exactly the multi-sampled model-judge case — same judge,
same input, N independent runs — as well as the classic human test-retest
design.

All functions take a ``ratings`` matrix shaped (n_items, n_repeats): one row per
item, one column per repeated measurement by the same rater. Categorical
functions accept any label type; variance-based functions require numeric
ratings.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from ._core import UndefinedStatistic
from .bootstrap import bootstrap_ci


def _as_matrix(ratings, numeric: bool) -> np.ndarray:
    x = np.asarray(ratings, dtype=float if numeric else object)
    if x.ndim != 2:
        raise ValueError("ratings must be 2-D: (n_items, n_repeats)")
    n, k = x.shape
    if n < 1:
        raise ValueError("empty input")
    if k < 2:
        raise ValueError("need at least 2 repeats per item")
    return x


def test_retest_agreement(ratings) -> float:
    """Mean pairwise agreement of a rater with itself across repeats.

    For each item, the fraction of unordered pairs of repeats that assigned the
    same label, averaged over items. 1.0 means the rater is perfectly
    self-consistent; with k repeats a single deviant label on an item costs
    (k-1) of the k*(k-1)/2 pairs.

    This is raw (not chance-corrected) agreement: for a stability check on one
    rater that is usually what you want first, alongside ``icc_1_1`` for the
    chance-and-variance-aware view.
    """
    x = _as_matrix(ratings, numeric=False)
    n, k = x.shape
    total_pairs = k * (k - 1) / 2
    per_item = np.empty(n)
    for i in range(n):
        matches = sum(
            1 for a in range(k) for b in range(a + 1, k) if x[i, a] == x[i, b]
        )
        per_item[i] = matches / total_pairs
    return float(per_item.mean())


def icc_1_1(ratings) -> float:
    """ICC(1,1): one-way random-effects intraclass correlation, single measurement.

    The natural intra-rater form: repeats of the same item are interchangeable
    draws from that rater on that item (there is no repeat-specific "column
    effect" worth modeling, unlike the two-way inter-rater forms). From one-way
    ANOVA: ICC(1,1) = (MSB - MSW) / (MSB + (k-1) * MSW), where MSB is the
    between-item and MSW the within-item mean square.

    High ICC(1,1) means item identity, not run-to-run noise, drives the scores.
    Requires numeric ratings and at least 2 items.
    """
    x = _as_matrix(ratings, numeric=True)
    n, k = x.shape
    if n < 2:
        raise ValueError("need at least 2 items")
    grand = x.mean()
    row_means = x.mean(axis=1)
    ssb = k * ((row_means - grand) ** 2).sum()
    ssw = ((x - row_means[:, None]) ** 2).sum()
    if ssb + ssw == 0:
        # Every value identical: no variance to partition -> ICC is 0/0.
        raise UndefinedStatistic("ICC(1,1) undefined: zero total variance")
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    denom = msb + (k - 1) * msw
    if denom == 0:
        raise UndefinedStatistic("ICC(1,1) undefined: no variance in the data")
    return float((msb - msw) / denom)


def within_item_variance(ratings) -> float:
    """Mean within-item sample variance (ddof=1) across repeats.

    The run-to-run noise of the rater in the units of the rating scale, averaged
    over items. 0 means perfectly reproducible. This is the quantity that decides
    how many repeats you need (see ``min_samples_for_ci``).
    """
    x = _as_matrix(ratings, numeric=True)
    return float(x.var(axis=1, ddof=1).mean())


def within_item_variance_ci(ratings, n_boot: int = 1000, seed: int = 7,
                            alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for ``within_item_variance``, resampling items."""
    x = _as_matrix(ratings, numeric=True)

    def stat(idx: list[int]) -> float:
        return float(x[idx].var(axis=1, ddof=1).mean())

    return bootstrap_ci(stat, x.shape[0], n_boot=n_boot, seed=seed, alpha=alpha)


def flip_rate(ratings, threshold: float | None = None) -> float:
    """Proportion of items whose outcome changes across repeats.

    With ``threshold`` None: an item flips if the rater did not give it the same
    label on every repeat (any-disagreement rate on raw labels).

    With a numeric ``threshold``: ratings are binarized as pass (>= threshold) /
    fail (< threshold) and an item flips if BOTH outcomes occur across its
    repeats. This is usually the number that matters in practice: a judge that
    wobbles between 3 and 4 on a 1-5 scale is noisy, but if the pass mark is 3.5
    that wobble flips the decision itself.
    """
    if threshold is None:
        x = _as_matrix(ratings, numeric=False)
        flips = [len(set(row)) > 1 for row in x]
    else:
        x = _as_matrix(ratings, numeric=True)
        passed = x >= threshold
        flips = [bool(row.any() and not row.all()) for row in passed]
    return float(np.mean(flips))


def min_samples_for_ci(target_width: float, pilot_ratings,
                       confidence: float = 0.95) -> int:
    """Minimum repeats per item for a target CI width on the item's mean score.

    Answers the planning question: "how many times must I run this rater on each
    item before its mean score means anything?" Uses the pooled within-item
    standard deviation from ``pilot_ratings`` and the normal approximation: a
    two-sided CI on a mean of n repeats has width 2 * z * sd / sqrt(n), so
    n = ceil((2 * z * sd / target_width)^2), floored at 2.

    This is a planning estimate, not a guarantee: it assumes the pilot variance
    is representative and the sample-mean CI is approximately normal. Widths on
    discrete bounded scales are, if anything, conservative near the scale ends.
    """
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    sd = float(np.sqrt(within_item_variance(pilot_ratings)))
    if sd == 0:
        # Perfectly reproducible pilot: any n achieves any width; 2 is the
        # smallest count from which spread is estimable at all.
        return 2
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    n = int(np.ceil((2 * z * sd / target_width) ** 2))
    return max(n, 2)
