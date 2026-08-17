"""Worked example — judge stability: one LLM judge, the same items, N repeats.

Multi-sampling a judge (same model, same prompt, temperature > 0) is a repeated
measurement of a single rater — the intra-rater reliability design. This
example runs one synthetic judge over the same item set 20 times and reports
the numbers that decide whether a judge's score means anything:

- flip rate on raw labels: how often the score itself changes across runs;
- flip rate at the pass mark: how often the DECISION changes — the number a
  leaderboard actually rides on;
- ICC(1,1): how much of the score variance is item identity vs run-to-run noise;
- within-item variance with a bootstrap CI: the judge's noise floor;
- min_samples_for_ci: how many runs you need before a mean score has the
  precision you are claiming for it.

Everything is synthetic and seeded: this example runs OFFLINE, with no API
keys and no model calls — deliberately, so the statistics are reproducible
from a clean clone by anyone.

Run:

    python examples/judge_stability.py
    python examples/judge_stability.py --items 100 --repeats 30 --target-width 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raters import intra


def generate_repeated_scores(n_items: int, n_repeats: int, seed: int) -> np.ndarray:
    """Synthesize one judge scoring the same items `n_repeats` times (1-5 scale).

    Each item has a latent quality on a continuous 1-5 scale and an
    item-specific ambiguity: most items are easy (low run-to-run noise), a
    minority are genuinely ambiguous (high noise) — mirroring how a sampled
    judge is rock-solid on clear cases and wobbles on borderline ones.
    100% synthetic — no model is called.
    """
    rng = np.random.default_rng(seed)
    quality = rng.uniform(1.0, 5.0, size=n_items)
    ambiguous = rng.random(n_items) < 0.25
    sigma = np.where(ambiguous, 0.85, 0.25)
    raw = quality[:, None] + rng.normal(0.0, 1.0, size=(n_items, n_repeats)) * sigma[:, None]
    return np.clip(np.round(raw), 1, 5)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=int, default=60, help="number of items")
    p.add_argument("--repeats", type=int, default=20, help="runs of the judge per item")
    p.add_argument("--seed", type=int, default=11, help="RNG seed (reproducible)")
    p.add_argument("--pass-mark", type=float, default=3.5,
                   help="pass/fail threshold on the score")
    p.add_argument("--target-width", type=float, default=0.5,
                   help="target 95%% CI width for a per-item mean score")
    args = p.parse_args(argv)

    x = generate_repeated_scores(args.items, args.repeats, args.seed)

    print(f"One judge, {args.items} items x {args.repeats} repeats (synthetic, offline)")
    print("-" * 68)
    print(f"test-retest agreement           : {intra.test_retest_agreement(x):.3f}")
    print(f"flip rate (any label change)    : {intra.flip_rate(x):.3f}")
    print(f"flip rate at pass mark {args.pass_mark:.1f}     : "
          f"{intra.flip_rate(x, threshold=args.pass_mark):.3f}")
    print(f"ICC(1,1)                        : {intra.icc_1_1(x):.3f}")
    wiv = intra.within_item_variance(x)
    lo, hi = intra.within_item_variance_ci(x, n_boot=1000, seed=args.seed)
    print(f"within-item variance            : {wiv:.3f}  CI[{lo:.3f},{hi:.3f}]")
    n_needed = intra.min_samples_for_ci(args.target_width, x)
    print(f"runs needed for CI width {args.target_width:.2f}   : {n_needed}")
    print()
    print("Reading it: the label flip rate says the judge often wobbles between")
    print("adjacent scores; the pass-mark flip rate says how often that wobble")
    print("flips the actual decision. ICC(1,1) near 1 means item identity, not")
    print("sampling noise, drives the score. The last line is the budget answer:")
    print("that many runs per item before a mean score supports the claimed")
    print("precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
