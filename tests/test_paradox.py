"""Tests reproducing the PUBLISHED kappa-paradox examples.

The two 2x2 tables below are Feinstein & Cicchetti (1990)'s canonical pair:
identical raw agreement (85%) but kappa 0.70 vs 0.32 — the paradox itself.
The PI/BI/PABAK values and the reconstruction identity are from Byrt, Bishop &
Carlin (1993).
"""
import numpy as np
import pytest

from raters import paradox
from raters import cohen_kappa


# Feinstein & Cicchetti 1990 — Table 1: balanced marginals.
# [[both-yes, A-yes/B-no], [A-no/B-yes, both-no]]
FC_TABLE_1 = np.array([[40, 9], [6, 45]], dtype=float)   # n=100, p_o=0.85
# Feinstein & Cicchetti 1990 — Table 2: same p_o, skewed prevalence.
FC_TABLE_2 = np.array([[80, 10], [5, 5]], dtype=float)   # n=100, p_o=0.85


def test_fc1990_paradox_pair_kappas():
    """Both tables agree on 85 of 100 items, yet:
    Table 1: p_e = 0.49*0.46 + 0.51*0.54 = 0.5008 -> kappa = 0.3492/0.4992 = 0.6995
    Table 2: p_e = 0.90*0.85 + 0.10*0.15 = 0.7800 -> kappa = 0.07/0.22   = 0.3182
    Same raw agreement, kappa halved — the published paradox."""
    d1 = paradox.diagnose(FC_TABLE_1)
    d2 = paradox.diagnose(FC_TABLE_2)
    assert d1["percent_agreement"] == pytest.approx(0.85)
    assert d2["percent_agreement"] == pytest.approx(0.85)
    assert d1["kappa"] == pytest.approx(0.6995, abs=1e-4)
    assert d2["kappa"] == pytest.approx(0.3182, abs=1e-4)


def test_fc1990_indices_and_pabak():
    """PI = |a-d|/n, BI = |b-c|/n, PABAK = 2*p_o - 1 (Byrt et al. 1993).
    Table 1: PI = 0.05, BI = 0.03. Table 2: PI = 0.75, BI = 0.05.
    PABAK is 0.70 for BOTH — identical observed agreement — which isolates the
    prevalence imbalance as the entire source of the kappa gap."""
    assert paradox.prevalence_index(FC_TABLE_1) == pytest.approx(0.05)
    assert paradox.bias_index(FC_TABLE_1) == pytest.approx(0.03)
    assert paradox.prevalence_index(FC_TABLE_2) == pytest.approx(0.75)
    assert paradox.bias_index(FC_TABLE_2) == pytest.approx(0.05)
    assert paradox.pabak(FC_TABLE_1) == pytest.approx(0.70)
    assert paradox.pabak(FC_TABLE_2) == pytest.approx(0.70)


def test_bbc1993_reconstruction_identity():
    """Byrt, Bishop & Carlin (1993): kappa = (PABAK - PI^2 + BI^2)/(1 - PI^2 + BI^2).
    Verified on both published tables and a third asymmetric one."""
    for t in (FC_TABLE_1, FC_TABLE_2, np.array([[70, 12], [4, 14]], dtype=float)):
        d = paradox.diagnose(t)
        pi, bi, pb = d["prevalence_index"], d["bias_index"], d["pabak"]
        reconstructed = (pb - pi ** 2 + bi ** 2) / (1 - pi ** 2 + bi ** 2)
        assert d["kappa"] == pytest.approx(reconstructed)


def test_diagnose_flags_the_paradox_case_only():
    d1 = paradox.diagnose(FC_TABLE_1)
    d2 = paradox.diagnose(FC_TABLE_2)
    assert d1["paradox"] is False           # 0.85 vs 0.70: consistent
    assert d2["paradox"] is True            # 0.85 vs 0.32: gap > 0.3
    assert "prevalence" in d2["note"]
    assert "PABAK" in d2["note"]


def test_diagnose_matches_frozen_suite_paradox_example():
    """The existing suite pins percent=0.91 / kappa=0.1346 for the table
    [[90, 5], [4, 1]]; diagnose must reproduce those numbers and flag it,
    and its kappa must equal cohen_kappa on the equivalent label sequences."""
    t = np.array([[90, 5], [4, 1]], dtype=float)
    d = paradox.diagnose(t)
    assert d["percent_agreement"] == pytest.approx(0.91)
    assert d["kappa"] == pytest.approx(0.134615, abs=1e-5)
    assert d["pabak"] == pytest.approx(0.82)
    assert d["prevalence_index"] == pytest.approx(0.89)
    assert d["paradox"] is True

    a = ["0"] * 95 + ["1"] * 5
    b = ["0"] * 90 + ["1"] * 5 + ["0"] * 4 + ["1"] * 1
    assert d["kappa"] == pytest.approx(cohen_kappa(a, b, ["0", "1"]))


def test_diagnose_single_category_kappa_none():
    """All ratings in one cell: raw agreement 100%, kappa undefined — the
    limiting case of the paradox. kappa must be None, not 1.0."""
    d = paradox.diagnose(np.array([[50, 0], [0, 0]], dtype=float))
    assert d["percent_agreement"] == pytest.approx(1.0)
    assert d["kappa"] is None
    assert d["paradox"] is True
    assert "undefined" in d["note"]


def test_perfect_balanced_agreement_not_flagged():
    d = paradox.diagnose(np.array([[25, 0], [0, 25]], dtype=float))
    assert d["kappa"] == pytest.approx(1.0)
    assert d["paradox"] is False


def test_input_validation():
    with pytest.raises(ValueError):
        paradox.prevalence_index(np.zeros((3, 3)))   # not 2x2
    with pytest.raises(ValueError):
        paradox.bias_index([[1, -2], [0, 3]])        # negative count
    with pytest.raises(ValueError):
        paradox.pabak(np.zeros((2, 2)))              # empty
    with pytest.raises(ValueError):
        paradox.diagnose([[1, 2, 3], [4, 5, 6]])     # wrong shape
