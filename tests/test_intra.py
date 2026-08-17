"""Hand-computed reference tests for intra-rater (test-retest) reliability.

Note: ``raters.intra.test_retest_agreement`` is deliberately referenced through
the module (``intra.test_retest_agreement``) — importing a callable named
``test_*`` into a test module's namespace would make pytest try to collect it.
"""
import numpy as np
import pytest

from raters import intra
from raters import UndefinedStatistic


# ---------------------------------------------------------------------------
# test-retest agreement — hand-computed
# ---------------------------------------------------------------------------

def test_retest_agreement_hand_computed():
    """Item 1: labels (A, A, B) -> pairs AA, AB, AB -> 1/3 agree.
    Item 2: labels (B, B, B) -> all 3 pairs agree -> 1.
    Mean over items = (1/3 + 1) / 2 = 2/3."""
    ratings = [["A", "A", "B"],
               ["B", "B", "B"]]
    assert intra.test_retest_agreement(ratings) == pytest.approx(2 / 3)


def test_retest_agreement_perfect_and_worst():
    assert intra.test_retest_agreement([["x", "x"], ["y", "y"]]) == pytest.approx(1.0)
    # Two repeats that never match -> 0.
    assert intra.test_retest_agreement([["x", "y"], ["y", "z"]]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ICC(1,1) — hand-computed one-way ANOVA
# ---------------------------------------------------------------------------

def test_icc_1_1_hand_computed():
    """3 items x 2 repeats: rows (1,2), (3,4), (5,6).

    grand=3.5; row means 1.5, 3.5, 5.5.
    SSB = 2*((-2)^2 + 0 + 2^2) = 16 -> MSB = 16/2 = 8.
    SSW = 6 * 0.25 = 1.5 -> MSW = 1.5/3 = 0.5.
    ICC(1,1) = (8 - 0.5) / (8 + 1*0.5) = 7.5/8.5 = 15/17.
    """
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    assert intra.icc_1_1(x) == pytest.approx(15 / 17)


def test_icc_1_1_perfect_repeats():
    """Zero within-item variance with real between-item variance -> exactly 1."""
    x = np.array([[2, 2, 2], [5, 5, 5], [9, 9, 9]], dtype=float)
    assert intra.icc_1_1(x) == pytest.approx(1.0)


def test_icc_1_1_constant_data_undefined():
    with pytest.raises(UndefinedStatistic):
        intra.icc_1_1(np.full((4, 3), 7.0))


def test_icc_1_1_needs_two_items():
    with pytest.raises(ValueError):
        intra.icc_1_1(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# within-item variance + bootstrap CI
# ---------------------------------------------------------------------------

def test_within_item_variance_hand_computed():
    """Rows (0,2) and (1,3): each has mean-centered deviations +-1, so sample
    variance (ddof=1) is 2 per item; the mean over items is 2."""
    x = np.array([[0, 2], [1, 3]], dtype=float)
    assert intra.within_item_variance(x) == pytest.approx(2.0)


def test_within_item_variance_zero_when_reproducible():
    assert intra.within_item_variance([[4.0, 4.0], [1.0, 1.0]]) == pytest.approx(0.0)


def test_within_item_variance_ci_degenerate_data():
    """Every item has identical spread, so every bootstrap resample computes the
    same value and the interval collapses onto the point estimate."""
    x = np.array([[0, 2]] * 8, dtype=float)
    lo, hi = intra.within_item_variance_ci(x, n_boot=200, seed=3)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def test_within_item_variance_ci_brackets_point():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 5)) + rng.normal(size=(40, 1)) * 3
    point = intra.within_item_variance(x)
    lo, hi = intra.within_item_variance_ci(x, n_boot=500, seed=4)
    assert lo <= point <= hi


# ---------------------------------------------------------------------------
# flip rate — raw labels and pass/fail threshold
# ---------------------------------------------------------------------------

def test_flip_rate_labels_hand_computed():
    """Items: (A,A) stable, (A,B) flips, (C,C) stable -> 1/3."""
    ratings = [["A", "A"], ["A", "B"], ["C", "C"]]
    assert intra.flip_rate(ratings) == pytest.approx(1 / 3)


def test_flip_rate_threshold_vs_labels():
    """Scores (3,4), (4,5), (1,2) with pass mark 3.5.

    On raw labels every item changes across repeats -> label flip rate 1.
    But only (3,4) straddles the threshold: (4,5) both pass, (1,2) both fail.
    Decision flip rate = 1/3 — the wobble that matters is the one that crosses
    the pass mark.
    """
    x = np.array([[3, 4], [4, 5], [1, 2]], dtype=float)
    assert intra.flip_rate(x) == pytest.approx(1.0)
    assert intra.flip_rate(x, threshold=3.5) == pytest.approx(1 / 3)


def test_flip_rate_zero_for_stable_rater():
    assert intra.flip_rate([["p", "p"], ["q", "q"]]) == pytest.approx(0.0)
    assert intra.flip_rate(np.array([[4.0, 5.0]]), threshold=3.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# minimum repeats for a target CI width
# ---------------------------------------------------------------------------

def test_min_samples_hand_computed():
    """Pilot rows (0,1,2) and (5,6,7): each has sample variance 1 -> pooled sd 1.
    For width 0.5 at 95%: n = ceil((2 * 1.959964 * 1 / 0.5)^2) = ceil(61.46) = 62."""
    pilot = np.array([[0, 1, 2], [5, 6, 7]], dtype=float)
    assert intra.min_samples_for_ci(0.5, pilot) == 62


def test_min_samples_tighter_width_needs_more():
    pilot = np.array([[0, 1, 2], [5, 6, 7]], dtype=float)
    assert intra.min_samples_for_ci(0.25, pilot) > intra.min_samples_for_ci(0.5, pilot)


def test_min_samples_zero_variance_floor():
    """A perfectly reproducible pilot needs no repeats for precision; the floor
    of 2 is returned because spread is not even estimable from 1."""
    pilot = np.array([[4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    assert intra.min_samples_for_ci(0.1, pilot) == 2


def test_min_samples_invalid_inputs():
    pilot = np.array([[0, 1], [1, 2]], dtype=float)
    with pytest.raises(ValueError):
        intra.min_samples_for_ci(0.0, pilot)
    with pytest.raises(ValueError):
        intra.min_samples_for_ci(0.5, pilot, confidence=1.0)


# ---------------------------------------------------------------------------
# input validation shared by the module
# ---------------------------------------------------------------------------

def test_intra_rejects_bad_shapes():
    with pytest.raises(ValueError):
        intra.test_retest_agreement(["A", "B", "C"])  # 1-D
    with pytest.raises(ValueError):
        intra.within_item_variance(np.array([[1.0], [2.0]]))  # 1 repeat
    with pytest.raises(ValueError):
        intra.flip_rate(np.empty((0, 2)))  # no items
