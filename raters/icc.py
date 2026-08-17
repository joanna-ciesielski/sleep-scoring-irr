"""ICC — intraclass correlation for continuous measures."""
from __future__ import annotations

import numpy as np

from ._core import UndefinedStatistic


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
