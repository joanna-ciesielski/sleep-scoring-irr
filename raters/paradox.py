"""Kappa-paradox diagnostics for 2x2 agreement tables.

High raw agreement can sit next to a very low kappa — the "kappa paradox"
(Feinstein & Cicchetti 1990). It is not a bug in kappa: when almost every item
falls in one category, chance agreement is enormous, so observed agreement has
to clear a much higher bar. But a report that shows only the raw percentage
walks straight into it. These diagnostics quantify WHY agreement and kappa
diverge for a 2x2 table:

- prevalence_index: how skewed the categories are (drives the paradox),
- bias_index: how differently the two raters use the categories,
- pabak: what kappa would be with no prevalence or bias imbalance,
- diagnose: all of the above plus kappa itself and a plain-language flag.

References
----------
Feinstein, A. R. & Cicchetti, D. V. (1990). High agreement but low kappa: I.
The problems of two paradoxes. J Clin Epidemiol 43(6), 543-549.

Byrt, T., Bishop, J. & Carlin, J. B. (1993). Bias, prevalence and kappa.
J Clin Epidemiol 46(5), 423-429. (Defines PI, BI, PABAK, and the identity
kappa = (PABAK - PI^2 + BI^2) / (1 - PI^2 + BI^2).)
"""
from __future__ import annotations

import numpy as np

from ._core import UndefinedStatistic


def _as_2x2(table) -> np.ndarray:
    t = np.asarray(table, dtype=float)
    if t.shape != (2, 2):
        raise ValueError("paradox diagnostics are defined for a 2x2 table "
                         f"(got shape {t.shape})")
    if np.any(t < 0):
        raise ValueError("counts must be non-negative")
    if t.sum() == 0:
        raise ValueError("empty table")
    return t


def prevalence_index(table) -> float:
    """|a - d| / n for a 2x2 table [[a, b], [c, d]] (Byrt, Bishop & Carlin 1993).

    a = both raters positive, d = both negative. Near 1 means one category
    dominates — the condition under which kappa collapses while raw agreement
    stays high. Near 0 means the categories are balanced and kappa is trustworthy.
    The sign is dropped: only the magnitude of the imbalance matters.
    """
    t = _as_2x2(table)
    return float(abs(t[0, 0] - t[1, 1]) / t.sum())


def bias_index(table) -> float:
    """|b - c| / n for a 2x2 table [[a, b], [c, d]] (Byrt, Bishop & Carlin 1993).

    b and c are the two disagreement cells. Near 0 means the raters distribute
    their disagreements symmetrically (no systematic tendency); a large value
    means one rater says "positive" systematically more often than the other.
    Unlike prevalence, bias INFLATES kappa relative to PABAK. Sign is dropped.
    """
    t = _as_2x2(table)
    return float(abs(t[0, 1] - t[1, 0]) / t.sum())


def pabak(table) -> float:
    """Prevalence-adjusted bias-adjusted kappa: 2 * p_o - 1 (Byrt et al. 1993).

    The kappa the table WOULD yield if prevalence were balanced and rater bias
    absent — equivalently a linear rescaling of raw agreement from [0, 1] to
    [-1, 1]. Comparing kappa with PABAK separates "the raters disagree" from
    "the marginals make chance agreement huge": a low kappa with a high PABAK is
    the paradox signature, not evidence of unreliable raters.
    """
    t = _as_2x2(table)
    p_o = (t[0, 0] + t[1, 1]) / t.sum()
    return float(2.0 * p_o - 1.0)


def _kappa_2x2(t: np.ndarray) -> float:
    n = t.sum()
    p = t / n
    p_o = p[0, 0] + p[1, 1]
    row = p.sum(axis=1)
    col = p.sum(axis=0)
    p_e = float(row @ col)
    if p_e == 1.0:
        raise UndefinedStatistic("kappa undefined: only one category present")
    return float((p_o - p_e) / (1.0 - p_e))


def diagnose(table, gap: float = 0.3) -> dict:
    """Full paradox work-up for a 2x2 table: the numbers plus a plain reading.

    Returns a dict with 'percent_agreement', 'kappa' (None when mathematically
    undefined), 'prevalence_index', 'bias_index', 'pabak', a boolean 'paradox'
    flag, and a human-readable 'note'.

    The flag is a reporting heuristic, not a theorem: it fires when raw
    agreement exceeds kappa by at least ``gap`` (default 0.3) — the situation
    where quoting the raw percentage alone would materially overstate
    reliability — or when kappa is undefined because every item landed in one
    category (the paradox's limiting case).
    """
    t = _as_2x2(table)
    p_o = float((t[0, 0] + t[1, 1]) / t.sum())
    pi = prevalence_index(t)
    bi = bias_index(t)
    pb = pabak(t)
    try:
        k = _kappa_2x2(t)
    except UndefinedStatistic:
        k = None

    if k is None:
        paradox = True
        note = (f"kappa is undefined: every rating fell in a single category "
                f"(prevalence index {pi:.2f}). Raw agreement of {p_o:.0%} "
                "carries no information about reliability here.")
    elif p_o - k >= gap:
        paradox = True
        note = (f"raw agreement {p_o:.0%} but kappa only {k:.2f} — "
                f"prevalence index {pi:.2f} means chance agreement is inflated "
                f"by skewed categories. PABAK {pb:.2f} shows what kappa would "
                "be with balanced marginals; report both, not the raw "
                "percentage alone.")
    else:
        paradox = False
        note = (f"raw agreement {p_o:.0%} and kappa {k:.2f} are consistent "
                f"(prevalence index {pi:.2f}, bias index {bi:.2f}).")

    return {
        "percent_agreement": p_o,
        "kappa": k,
        "prevalence_index": pi,
        "bias_index": bi,
        "pabak": pb,
        "paradox": paradox,
        "note": note,
    }
