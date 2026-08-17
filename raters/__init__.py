"""raters — chance-corrected agreement statistics for any raters, human or model.

When two raters score the same items, how much do they agree — and is that
agreement better than chance? This package answers that question for human
annotators, clinical scorers, and LLM judges alike: percent agreement,
chance-corrected kappa (unweighted and weighted for ordinal scales), Fleiss'
kappa, Krippendorff's alpha (with missing-data handling), and ICC for
continuous measures — each with bootstrap confidence intervals. All statistics
are hand-implemented (numpy only) and tested against hand-computed or published
reference values, so every reported number is traceable to a reference.
"""
from ._core import UndefinedStatistic
from .agreement import (
    confusion_matrix,
    percent_agreement,
    cohen_kappa,
    fleiss_kappa,
)
from .alpha import krippendorff_alpha
from .icc import icc
from .bootstrap import bootstrap_ci

__all__ = [
    "percent_agreement", "cohen_kappa", "fleiss_kappa", "krippendorff_alpha",
    "icc", "bootstrap_ci", "confusion_matrix", "UndefinedStatistic",
]
__version__ = "0.1.0"
