"""Categorical agreement: percent agreement, Cohen's kappa, Fleiss' kappa.

Every statistic is hand-implemented (numpy only) so the math is inspectable and
testable against hand-computed reference values — the posture any audited
measurement pipeline needs: deterministic, versioned, and traceable to a
reference.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._core import Category, UndefinedStatistic, _weight_matrix


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


def cohen_kappa(a: Sequence[Category], b: Sequence[Category],
                categories: Sequence[Category], weights: str | None = None) -> float:
    """Chance-corrected agreement for two raters.

    Unweighted for nominal labels; 'linear'/'quadratic' for ordinal categories
    (severity grades, quality scores), where a near-miss to an adjacent category
    should cost less than a distant one. `categories` order defines ordinal
    distance.
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
