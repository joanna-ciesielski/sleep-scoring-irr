"""Compatibility layer — the statistics now live in the domain-neutral
``raters`` package.

This module re-exports every public name (and the private helpers historical
callers may have imported) so that existing code and tests keep working
unchanged. New code should import from ``raters`` directly.
"""
from raters._core import (  # noqa: F401
    Category,
    UndefinedStatistic,
    _weight_matrix,
)
from raters.agreement import (  # noqa: F401
    confusion_matrix,
    percent_agreement,
    cohen_kappa,
    fleiss_kappa,
)
from raters.alpha import krippendorff_alpha  # noqa: F401
from raters.icc import icc  # noqa: F401
from raters.bootstrap import bootstrap_ci  # noqa: F401

__all__ = [
    "percent_agreement", "cohen_kappa", "fleiss_kappa", "krippendorff_alpha",
    "icc", "bootstrap_ci", "confusion_matrix", "UndefinedStatistic", "Category",
]
