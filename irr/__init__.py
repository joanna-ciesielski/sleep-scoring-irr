"""sleep-scoring-irr — inter-rater reliability for sleep study scoring.

A compact, tested reference implementation of the agreement statistics a clinical
inter-rater-reliability module needs: chance-corrected kappa (unweighted and
weighted for ordinal sleep stages), Fleiss' kappa, Krippendorff's alpha, and ICC
for continuous indices — each with bootstrap confidence intervals and per-rater
drill-down. All statistics are hand-implemented (numpy only) and tested against
hand-computed reference values.
"""
from .stats import (
    percent_agreement,
    cohen_kappa,
    fleiss_kappa,
    krippendorff_alpha,
    icc,
    bootstrap_ci,
    confusion_matrix,
    UndefinedStatistic,
)
from .sleep import (
    AASM_STAGES,
    generate_epoch_scores,
    generate_continuous_indices,
    analyze_epoch_scores,
    EpochAgreementReport,
)

__all__ = [
    "percent_agreement", "cohen_kappa", "fleiss_kappa", "krippendorff_alpha",
    "icc", "bootstrap_ci", "confusion_matrix", "UndefinedStatistic",
    "AASM_STAGES", "generate_epoch_scores", "generate_continuous_indices",
    "analyze_epoch_scores", "EpochAgreementReport",
]
__version__ = "0.1.0"
