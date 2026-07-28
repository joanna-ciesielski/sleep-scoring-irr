"""Sleep-scoring domain layer on top of the generic IRR statistics.

Models the real sleep-medicine use case: two or more scorers labeling 30-second
epochs with AASM sleep stages, and continuous summary indices (e.g. AHI). The
synthetic data is deliberately structured so raters mostly agree but make
realistic *near-miss* errors (N2 vs N3, not N2 vs REM) — which is exactly why
weighted kappa and interval-level alpha matter here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats

# AASM stages in physiological/ordinal order: Wake -> N1 -> N2 -> N3 -> REM.
# Ordering matters: weighted kappa penalizes far misses more than near ones.
AASM_STAGES = ["W", "N1", "N2", "N3", "R"]


def generate_epoch_scores(n_epochs: int = 200, n_raters: int = 2,
                          near_miss: float = 0.12, far_miss: float = 0.02,
                          seed: int = 1) -> tuple[list[str], list[list[str]]]:
    """Synthesize a truth hypnogram and `n_raters` noisy scorings of it.

    Returns (truth, rater_scores). Errors are mostly to an *adjacent* stage
    (near_miss) with occasional non-adjacent errors (far_miss), mirroring how
    human scorers actually disagree. 100% synthetic — no patient data.
    """
    rng = np.random.default_rng(seed)
    k = len(AASM_STAGES)

    # A simple stage-transition (Markov) chain for a plausible night.
    trans = np.array([
        [0.60, 0.25, 0.10, 0.00, 0.05],  # W
        [0.15, 0.35, 0.45, 0.00, 0.05],  # N1
        [0.05, 0.15, 0.55, 0.15, 0.10],  # N2
        [0.02, 0.05, 0.33, 0.55, 0.05],  # N3
        [0.10, 0.10, 0.25, 0.00, 0.55],  # R
    ])
    truth_idx = [0]
    for _ in range(n_epochs - 1):
        truth_idx.append(int(rng.choice(k, p=trans[truth_idx[-1]])))
    truth = [AASM_STAGES[i] for i in truth_idx]

    def score_one(ti: int) -> str:
        u = rng.random()
        if u < far_miss:  # non-adjacent error
            choices = [j for j in range(k) if abs(j - ti) > 1]
            return AASM_STAGES[int(rng.choice(choices))] if choices else AASM_STAGES[ti]
        if u < far_miss + near_miss:  # adjacent error
            choices = [j for j in range(k) if abs(j - ti) == 1]
            return AASM_STAGES[int(rng.choice(choices))]
        return AASM_STAGES[ti]

    raters = [[score_one(ti) for ti in truth_idx] for _ in range(n_raters)]
    return truth, raters


def generate_continuous_indices(n_subjects: int = 40, n_raters: int = 2,
                                rater_bias: float = 0.0, noise: float = 1.5,
                                seed: int = 2) -> np.ndarray:
    """Synthesize per-subject AHI-like values scored by several raters.

    Returns a (subjects x raters) matrix. `rater_bias` adds a constant offset to
    later raters — the thing ICC(2,1) absolute-agreement penalizes and ICC(3,1)
    consistency ignores, so the two forms diverge in a testable way.
    """
    rng = np.random.default_rng(seed)
    true_ahi = rng.gamma(shape=2.0, scale=8.0, size=n_subjects)  # skewed, like real AHI
    out = np.zeros((n_subjects, n_raters))
    for r in range(n_raters):
        bias = rater_bias * r
        out[:, r] = true_ahi + bias + rng.normal(0, noise, size=n_subjects)
    return np.clip(out, 0, None)


@dataclass
class EpochAgreementReport:
    n_epochs: int
    n_raters: int
    per_pair: dict[tuple[int, int], dict] = field(default_factory=dict)
    fleiss: float | None = None
    alpha_interval: float | None = None
    per_stage_agreement: dict[str, float] = field(default_factory=dict)
    confusion_hotspots: list[tuple[str, str, int]] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = [f"Epoch staging agreement — {self.n_epochs} epochs, {self.n_raters} raters",
                 "-" * 60]
        for (i, j), d in self.per_pair.items():
            lines.append(
                f"raters {i}-{j}: percent={d['percent']:.3f}  "
                f"kappa={d['kappa']:.3f}  weighted-kappa={d['weighted']:.3f}  "
                f"CI[{d['ci'][0]:.2f},{d['ci'][1]:.2f}]"
            )
        if self.fleiss is not None:
            lines.append(f"Fleiss' kappa (all raters): {self.fleiss:.3f}")
        if self.alpha_interval is not None:
            lines.append(f"Krippendorff alpha (interval): {self.alpha_interval:.3f}")
        lines.append("Per-stage agreement (where raters diverge most):")
        for stage, ag in sorted(self.per_stage_agreement.items(), key=lambda kv: kv[1]):
            lines.append(f"  {stage:>3}: {ag:.3f}")
        if self.confusion_hotspots:
            lines.append("Top confusion pairs (rater 0 vs 1):")
            for s1, s2, c in self.confusion_hotspots:
                lines.append(f"  {s1} <-> {s2}: {c} epochs")
        return lines


def analyze_epoch_scores(raters: list[list[str]],
                         stages: list[str] = AASM_STAGES) -> EpochAgreementReport:
    """Full drill-down: per-pair kappas with CIs, Fleiss, interval alpha, per-stage
    agreement, and the biggest confusion pairs — so a lead sees WHERE scorers
    diverge, not just one global number."""
    n_raters = len(raters)
    if n_raters < 2:
        raise ValueError("inter-rater reliability needs at least 2 raters")
    n_epochs = len(raters[0])
    if any(len(r) != n_epochs for r in raters):
        raise ValueError("all raters must score the same number of epochs")
    rep = EpochAgreementReport(n_epochs=n_epochs, n_raters=n_raters)

    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            a, b = raters[i], raters[j]

            def stat(idx: list[int], a=a, b=b) -> float:
                aa = [a[t] for t in idx]
                bb = [b[t] for t in idx]
                return stats.cohen_kappa(aa, bb, stages, weights="linear")

            rep.per_pair[(i, j)] = {
                "percent": stats.percent_agreement(a, b),
                "kappa": stats.cohen_kappa(a, b, stages),
                "weighted": stats.cohen_kappa(a, b, stages, weights="linear"),
                "ci": stats.bootstrap_ci(stat, n_epochs, n_boot=500),
            }

    # Fleiss + alpha across all raters.
    index = {s: c for c, s in enumerate(stages)}
    counts = np.zeros((n_epochs, len(stages)))
    for t in range(n_epochs):
        for r in raters:
            counts[t, index[r[t]]] += 1
    rep.fleiss = stats.fleiss_kappa(counts)
    rep.alpha_interval = stats.krippendorff_alpha(raters, level="interval", categories=stages)

    # Per-stage agreement (fraction of epochs where all raters agree, by rater-0 stage).
    for stage in stages:
        epochs = [t for t in range(n_epochs) if raters[0][t] == stage]
        if epochs:
            agree = sum(all(r[t] == raters[0][t] for r in raters) for t in epochs)
            rep.per_stage_agreement[stage] = agree / len(epochs)

    # Confusion hotspots between rater 0 and 1.
    if n_raters >= 2:
        m = stats.confusion_matrix(raters[0], raters[1], stages)
        pairs = []
        for a in range(len(stages)):
            for b in range(len(stages)):
                if a != b and m[a, b] > 0:
                    pairs.append((stages[a], stages[b], int(m[a, b])))
        rep.confusion_hotspots = sorted(pairs, key=lambda x: -x[2])[:4]

    return rep
