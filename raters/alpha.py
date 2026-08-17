"""Krippendorff's alpha — the most general agreement coefficient."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._core import Category, UndefinedStatistic, _weight_matrix


def krippendorff_alpha(data: Sequence[Sequence[Category | None]],
                       level: str = "nominal",
                       categories: Sequence[Category] | None = None) -> float:
    """The most general agreement coefficient.

    `data` is raters x units; None marks a missing score, and units with fewer
    than two scores are ignored. `level` is 'nominal' (0/1 distance) or
    'interval' (squared distance on category index — also the standard practical
    stand-in for ordinal rating scales).

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
