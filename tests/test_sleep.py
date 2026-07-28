"""Tests for the sleep-scoring domain layer: deterministic synthetic data,
realistic error structure, and a coherent drill-down report."""
import numpy as np
import pytest

import irr
from irr import sleep


def test_generate_epoch_scores_deterministic():
    t1, r1 = sleep.generate_epoch_scores(n_epochs=150, seed=42)
    t2, r2 = sleep.generate_epoch_scores(n_epochs=150, seed=42)
    assert t1 == t2 and r1 == r2  # same seed -> identical


def test_generated_scores_are_valid_stages():
    _, raters = sleep.generate_epoch_scores(n_epochs=200, n_raters=3, seed=5)
    assert len(raters) == 3
    for r in raters:
        assert len(r) == 200
        assert set(r) <= set(sleep.AASM_STAGES)


def test_weighted_beats_unweighted_on_near_misses():
    """The generator makes errors mostly to ADJACENT stages, so linear-weighted
    kappa should exceed unweighted kappa on this data — the whole reason weighting
    matters for ordinal sleep stages."""
    _, raters = sleep.generate_epoch_scores(n_epochs=400, near_miss=0.15,
                                            far_miss=0.01, seed=7)
    a, b = raters[0], raters[1]
    unweighted = irr.cohen_kappa(a, b, sleep.AASM_STAGES)
    weighted = irr.cohen_kappa(a, b, sleep.AASM_STAGES, weights="linear")
    assert weighted > unweighted


def test_analyze_epoch_scores_report_shape():
    _, raters = sleep.generate_epoch_scores(n_epochs=200, n_raters=3, seed=11)
    rep = sleep.analyze_epoch_scores(raters)
    assert rep.n_epochs == 200 and rep.n_raters == 3
    assert len(rep.per_pair) == 3  # C(3,2) pairs
    for d in rep.per_pair.values():
        lo, hi = d["ci"]
        assert lo <= hi
        assert 0.0 <= d["percent"] <= 1.0
    assert rep.fleiss is not None
    assert rep.alpha_interval is not None
    assert rep.confusion_hotspots  # some confusion pairs exist
    # summary_lines renders without error and mentions the key stats
    text = "\n".join(rep.summary_lines())
    assert "Fleiss" in text and "kappa" in text


def test_continuous_indices_bias_lowers_icc_2_1():
    """A constant rater bias should pull ICC(2,1) (absolute agreement) below
    ICC(3,1) (consistency), while zero bias keeps them close."""
    unbiased = sleep.generate_continuous_indices(n_subjects=60, rater_bias=0.0,
                                                 noise=1.0, seed=3)
    biased = sleep.generate_continuous_indices(n_subjects=60, rater_bias=6.0,
                                               noise=1.0, seed=3)
    icc21_u, icc31_u = irr.icc(unbiased, "2,1"), irr.icc(unbiased, "3,1")
    icc21_b, icc31_b = irr.icc(biased, "2,1"), irr.icc(biased, "3,1")
    # bias barely moves consistency but clearly drops absolute agreement
    assert icc31_b - icc21_b > icc31_u - icc21_u
    assert icc21_b < icc21_u


def test_continuous_indices_nonnegative():
    x = sleep.generate_continuous_indices(n_subjects=50, seed=9)
    assert np.all(x >= 0)


def test_analyze_requires_two_raters_with_clear_error():
    _, raters = sleep.generate_epoch_scores(n_epochs=40, n_raters=1, seed=1)
    with pytest.raises(ValueError, match="at least 2 raters"):
        sleep.analyze_epoch_scores(raters)


def test_analyze_rejects_ragged_raters():
    with pytest.raises(ValueError, match="same number of epochs"):
        sleep.analyze_epoch_scores([["W", "N1", "N2"], ["W", "N1"]])
