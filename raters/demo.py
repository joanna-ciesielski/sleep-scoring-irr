"""Command-line demo: run the full inter-rater-reliability drill-down on
synthetic sleep-scoring data.

    python -m raters.demo                 # epoch staging + continuous-index report
    python -m raters.demo --epochs 300 --raters 3
    python -m raters.demo --seed 12

100% synthetic data — no patient records are read or written.
"""
from __future__ import annotations

import argparse

import numpy as np

import raters
from raters import sleep


def _epoch_section(n_epochs: int, n_raters: int, seed: int) -> list[str]:
    _, raters = sleep.generate_epoch_scores(n_epochs=n_epochs, n_raters=n_raters,
                                            seed=seed)
    rep = sleep.analyze_epoch_scores(raters)
    return rep.summary_lines()


def _continuous_section(n_subjects: int, seed: int) -> list[str]:
    # Two scorers of an AHI-like index, the second carrying a small constant bias.
    data = sleep.generate_continuous_indices(n_subjects=n_subjects, n_raters=2,
                                             rater_bias=1.5, noise=1.5, seed=seed)
    icc_abs = raters.icc(data, form="2,1")
    icc_con = raters.icc(data, form="3,1")

    def stat_abs(idx):
        return raters.icc(data[idx], form="2,1")

    lo, hi = raters.bootstrap_ci(stat_abs, data.shape[0], n_boot=500, seed=seed)
    lines = [
        "",
        f"Continuous index agreement (AHI-like) — {n_subjects} subjects, 2 raters",
        "-" * 60,
        f"ICC(2,1) absolute agreement : {icc_abs:.3f}  CI[{lo:.2f},{hi:.2f}]",
        f"ICC(3,1) consistency        : {icc_con:.3f}",
        f"gap (calibration cost)      : {icc_con - icc_abs:.3f}  "
        "<- a constant rater bias lives here",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=200, help="number of 30s epochs")
    p.add_argument("--raters", type=int, default=3, help="number of scorers")
    p.add_argument("--subjects", type=int, default=40,
                   help="subjects for the continuous-index section")
    p.add_argument("--seed", type=int, default=1, help="RNG seed (reproducible)")
    args = p.parse_args(argv)

    lines = _epoch_section(args.epochs, args.raters, args.seed)
    lines += _continuous_section(args.subjects, args.seed + 1)
    lines += [
        "",
        "Reading it: percent agreement flatters skewed data; kappa corrects for",
        "chance; weighted kappa forgives N2<->N3 near-misses; interval alpha and",
        "Fleiss extend to >2 raters; ICC(2,1) vs (3,1) exposes rater calibration.",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
