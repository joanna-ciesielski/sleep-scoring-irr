"""Reference-value tests for the IRR statistics.

Every non-trivial statistic is checked against a value computed BY HAND (see the
worked arithmetic in each docstring) or against a PUBLISHED canonical example.
This is the point of the package: the math is traceable to a reference, not to
whatever the code happens to emit.
"""
import numpy as np
import pytest

import raters
from raters import sleep


# ---------------------------------------------------------------------------
# Percent agreement + confusion matrix
# ---------------------------------------------------------------------------

def test_percent_agreement():
    a = ["W", "N1", "N2", "N3", "R"]
    b = ["W", "N1", "N3", "N3", "R"]  # one disagreement of five
    assert raters.percent_agreement(a, b) == pytest.approx(0.8)


def test_confusion_matrix_counts():
    a = ["W", "W", "N1", "N1"]
    b = ["W", "N1", "N1", "N1"]
    m = raters.confusion_matrix(a, b, ["W", "N1"])
    # M[W,W]=1, M[W,N1]=1, M[N1,N1]=2
    assert m.tolist() == [[1, 1], [0, 2]]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        raters.percent_agreement(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        raters.confusion_matrix(["a"], ["a", "b"], ["a", "b"])


def test_confusion_matrix_unknown_label_raises():
    with pytest.raises(ValueError):
        raters.confusion_matrix(["W", "X"], ["W", "W"], ["W", "N1"])


# ---------------------------------------------------------------------------
# Cohen's kappa — hand-computed
# ---------------------------------------------------------------------------

def test_cohen_kappa_binary_reference():
    """Classic 2x2 worked example.

    Confusion: both-Yes=20, A-Yes/B-No=5, A-No/B-Yes=10, both-No=15 (n=50).
    p_o = 35/50 = 0.70; p_e = 0.5*0.6 + 0.5*0.4 = 0.50;
    kappa = (0.70-0.50)/(1-0.50) = 0.40.
    """
    a = ["Y"] * 20 + ["Y"] * 5 + ["N"] * 10 + ["N"] * 15
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert raters.cohen_kappa(a, b, ["Y", "N"]) == pytest.approx(0.40)


def test_weighted_kappa_reference_and_ordering():
    """3-category ordinal example where every miss is to an ADJACENT category.

    A: 0 0 1 1 2 2 ; B: 0 1 1 2 2 2.
    Hand arithmetic gives unweighted kappa = 0.500 and linear-weighted = 0.625.
    Because the disagreements are near-misses, weighting them less RAISES kappa.
    """
    a = ["0", "0", "1", "1", "2", "2"]
    b = ["0", "1", "1", "2", "2", "2"]
    cats = ["0", "1", "2"]
    unweighted = raters.cohen_kappa(a, b, cats)
    linear = raters.cohen_kappa(a, b, cats, weights="linear")
    assert unweighted == pytest.approx(0.500)
    assert linear == pytest.approx(0.625)
    assert linear > unweighted  # near-misses cost less under weighting


def test_quadratic_weight_leq_linear_penalty():
    """For adjacent-only misses, quadratic weights penalize even less than linear,
    so quadratic-weighted kappa >= linear-weighted kappa."""
    a = ["0", "0", "1", "1", "2", "2"]
    b = ["0", "1", "1", "2", "2", "2"]
    cats = ["0", "1", "2"]
    linear = raters.cohen_kappa(a, b, cats, weights="linear")
    quad = raters.cohen_kappa(a, b, cats, weights="quadratic")
    assert quad >= linear


def test_perfect_agreement_is_one():
    a = ["W", "N1", "N2", "N3", "R", "N2"]
    assert raters.cohen_kappa(a, a, sleep.AASM_STAGES) == pytest.approx(1.0)
    assert raters.cohen_kappa(a, a, sleep.AASM_STAGES, weights="linear") == pytest.approx(1.0)


def test_kappa_paradox():
    """High raw agreement + skewed prevalence -> low kappa (the kappa paradox).

    n=100: M[0,0]=90, M[0,1]=5, M[1,0]=4, M[1,1]=1.
    percent = 0.91 but kappa = 0.014/0.104 = 0.1346.
    """
    a = ["0"] * 90 + ["0"] * 5 + ["1"] * 4 + ["1"] * 1
    b = ["0"] * 90 + ["1"] * 5 + ["0"] * 4 + ["1"] * 1
    assert raters.percent_agreement(a, b) == pytest.approx(0.91)
    assert raters.cohen_kappa(a, b, ["0", "1"]) == pytest.approx(0.134615, abs=1e-5)


def test_unknown_weights_raises():
    with pytest.raises(ValueError):
        raters.cohen_kappa(["a", "b"], ["a", "b"], ["a", "b"], weights="cubic")


# ---------------------------------------------------------------------------
# Fleiss' kappa — hand-computed
# ---------------------------------------------------------------------------

def test_fleiss_kappa_reference():
    """3 raters, 4 items, 2 categories.

    counts rows [3,0],[2,1],[1,2],[0,3]; p_bar = 2/3, p_e = 0.5,
    kappa = (2/3 - 1/2)/(1 - 1/2) = 1/3.
    """
    counts = np.array([[3, 0], [2, 1], [1, 2], [0, 3]])
    assert raters.fleiss_kappa(counts) == pytest.approx(1 / 3)


def test_fleiss_perfect_agreement():
    counts = np.array([[3, 0], [0, 3], [3, 0]])
    assert raters.fleiss_kappa(counts) == pytest.approx(1.0)


def test_fleiss_unequal_raters_raises():
    with pytest.raises(ValueError):
        raters.fleiss_kappa(np.array([[3, 0], [2, 0]]))  # rows sum to 3 vs 2


# ---------------------------------------------------------------------------
# Krippendorff's alpha — hand-computed + published canonical example
# ---------------------------------------------------------------------------

def test_krippendorff_nominal_reference():
    """2 raters, 4 units; A: a a b b, B: a b b b.

    Coincidences give d_o=2, d_e=30/7, alpha = 1 - 2/(30/7) = 8/15.
    """
    a = ["a", "a", "b", "b"]
    b = ["a", "b", "b", "b"]
    assert raters.krippendorff_alpha([a, b], level="nominal") == pytest.approx(8 / 15)


def test_krippendorff_interval_reference():
    """A: 1 2 3 ; B: 1 2 4 (interval on category index).

    d_o=2, d_e=82/5, alpha = 1 - 2/16.4 = 0.878049.
    """
    a = ["1", "2", "3"]
    b = ["1", "2", "4"]
    val = raters.krippendorff_alpha([a, b], level="interval")
    assert val == pytest.approx(0.8780487804878049)


def test_krippendorff_interval_preserves_scale_when_category_absent():
    """If a middle category is absent from the data but present in `categories`,
    the ordinal spacing must be preserved (not compressed). A distance-2 miss
    (1 vs 3, spanning the absent '2') must cost more than a distance-1 miss."""
    cats = ["0", "1", "2", "3", "4"]
    full = raters.krippendorff_alpha([["0", "1", "1", "4"], ["0", "1", "3", "4"]],
                                  level="interval", categories=cats)
    adjacent = raters.krippendorff_alpha([["0", "1", "1", "4"], ["0", "1", "2", "4"]],
                                      level="interval", categories=cats)
    assert full < adjacent


def test_krippendorff_rejects_labels_outside_categories():
    with pytest.raises(ValueError):
        raters.krippendorff_alpha([["0", "9"], ["0", "0"]],
                               level="interval", categories=["0", "1", "2"])


def test_krippendorff_canonical_missing_data():
    """Krippendorff (2011) canonical reliability data, 4 observers x 12 units,
    with missing scores. Published: alpha_nominal = 0.743, alpha_interval = 0.849.
    This validates the missing-data / coincidence-matrix handling."""
    N = None
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N]
    B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3]
    C = [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N]
    D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]
    data = [A, B, C, D]
    assert raters.krippendorff_alpha(data, level="nominal") == pytest.approx(0.743, abs=1e-3)
    assert raters.krippendorff_alpha(data, level="interval") == pytest.approx(0.849, abs=1e-3)


def test_krippendorff_perfect_agreement():
    a = ["W", "N1", "N2", "N3"]
    assert raters.krippendorff_alpha([a, a, a], level="nominal") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ICC — hand-computed, both forms diverge under constant bias
# ---------------------------------------------------------------------------

def test_icc_forms_reference():
    """4 units x 2 raters; rater2 = rater1 + 1 (a constant +1 bias).

    MSR=10/3, MSC=2, MSE=0. ICC(2,1)=10/13=0.7692 (penalizes the bias);
    ICC(3,1)=1.0 (consistency ignores the constant offset).
    """
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    assert raters.icc(x, form="2,1") == pytest.approx(10 / 13)
    assert raters.icc(x, form="3,1") == pytest.approx(1.0)
    assert raters.icc(x, "3,1") > raters.icc(x, "2,1")  # consistency >= absolute agreement


def test_icc_unknown_form_raises():
    x = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)
    with pytest.raises(ValueError):
        raters.icc(x, form="1,1")


# ---------------------------------------------------------------------------
# Undefined ("no information") cases must raise, not return 1.0
# ---------------------------------------------------------------------------

def test_degenerate_single_category_is_undefined_not_one():
    """A single category carries no information: chance-corrected agreement is
    0/0, which must raise (UndefinedStatistic) rather than silently return 1.0.
    This is what makes the bootstrap skip such resamples instead of inflating."""
    with pytest.raises(raters.UndefinedStatistic):
        raters.cohen_kappa(["W", "W", "W"], ["W", "W", "W"], ["W", "N1"])
    with pytest.raises(raters.UndefinedStatistic):
        raters.fleiss_kappa(np.array([[3, 0], [3, 0], [3, 0]]))
    with pytest.raises(raters.UndefinedStatistic):
        raters.krippendorff_alpha([["W", "W"], ["W", "W"]], level="nominal")


def test_icc_zero_variance_is_undefined():
    """Constant data (no variance) is undefined, not perfect reliability."""
    with pytest.raises(raters.UndefinedStatistic):
        raters.icc(np.full((5, 3), 7.0), form="2,1")
    # subjects identical, raters differ by a constant -> no between-subject
    # variance -> consistency undefined (must not report 1.0).
    with pytest.raises(raters.UndefinedStatistic):
        raters.icc(np.array([[1, 2], [1, 2], [1, 2], [1, 2]], dtype=float), form="3,1")


# ---------------------------------------------------------------------------
# Bootstrap CI — excludes undefined resamples, brackets the point estimate
# ---------------------------------------------------------------------------

def test_input_validation_guards():
    """Defensive guards raise clear errors rather than producing garbage."""
    with pytest.raises(ValueError):
        raters.percent_agreement([], [])
    with pytest.raises(ValueError):
        raters.cohen_kappa([], [], ["a", "b"])          # empty input
    with pytest.raises(ValueError):
        raters.fleiss_kappa(np.zeros((0, 2)))           # empty table
    with pytest.raises(ValueError):
        raters.krippendorff_alpha([["a", "b"], ["a"]])  # ragged raters
    with pytest.raises(ValueError):
        raters.krippendorff_alpha([["a", "b"]], level="ordinal")  # unknown level
    with pytest.raises(ValueError):
        raters.icc(np.array([[1.0]]))                    # too small
    with pytest.raises(ValueError):
        raters.bootstrap_ci(lambda idx: (_ for _ in ()).throw(raters.UndefinedStatistic("x")),
                         5, n_boot=10)                # all resamples degenerate


def test_bootstrap_skips_undefined_resamples():
    """bootstrap_ci must DROP resamples whose statistic is undefined, not count
    them. Here the stat is 'undefined' (raises UndefinedStatistic) unless unit 0
    is present; every defined draw returns 0.5, so the whole interval must be
    exactly 0.5 — proving the undefined draws never entered the distribution.
    (Before the fix, the degenerate branches returned 1.0 and this exclusion
    never happened.)"""
    def stat(idx):
        if 0 not in idx:
            raise raters.UndefinedStatistic("degenerate")
        return 0.5

    lo, hi = raters.bootstrap_ci(stat, 6, n_boot=500, seed=2)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    cats = sleep.AASM_STAGES
    truth = rng.integers(0, 5, size=300)
    a = [cats[i] for i in truth]
    # b agrees ~85% of the time, else a random stage
    b = [cats[i] if rng.random() < 0.85 else cats[rng.integers(0, 5)] for i in truth]
    point = raters.cohen_kappa(a, b, cats)

    def stat(idx):
        aa = [a[t] for t in idx]
        bb = [b[t] for t in idx]
        return raters.cohen_kappa(aa, bb, cats)

    lo, hi = raters.bootstrap_ci(stat, len(a), n_boot=400, seed=3)
    assert lo <= hi
    assert lo <= point <= hi
    assert hi - lo < 0.3  # 300 epochs -> reasonably tight
