"""Worked example — model raters: two LLM judges scoring the same items.

Two judges score the same N items on an ordinal 1-5 quality scale. The
synthetic generator mimics how real LLM judges disagree: scores cluster at the
top of the scale (most benchmark answers are decent), disagreements are mostly
near-misses (4 vs 5, not 2 vs 5), and one judge is slightly more lenient than
the other.

The example shows the two traps a raw agreement percentage sets:

1. On the 1-5 scale, unweighted kappa treats a 4-vs-5 near-miss exactly like a
   1-vs-5 blunder. Weighted kappa prices the ordinal structure in — the gap
   between weighted and unweighted kappa is the near-miss signal.

2. Binarized at a pass mark (as leaderboard-style evals do), the skewed pass
   rate inflates chance agreement: raw pass/fail agreement looks excellent
   while kappa collapses — the kappa paradox (Feinstein & Cicchetti 1990).
   PABAK and the prevalence index (Byrt, Bishop & Carlin 1993) show why.

Everything is synthetic and seeded: this example runs OFFLINE, with no API
keys and no model calls — deliberately, so the statistics are reproducible
from a clean clone by anyone.

Run:

    python examples/llm_judges.py
    python examples/llm_judges.py --items 500 --seed 3 --pass-mark 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raters import cohen_kappa, confusion_matrix, krippendorff_alpha, percent_agreement
from raters import paradox

SCALE = [1, 2, 3, 4, 5]


def generate_judge_scores(n_items: int, seed: int,
                          leniency_b: float = 0.25) -> tuple[list[int], list[int]]:
    """Synthesize two judges' 1-5 scores for the same items.

    Latent item quality is drawn skewed-high (a mostly-good answer set). Each
    judge reads the latent quality with small, mostly-adjacent noise; judge B
    carries a slight lenient bias. 100% synthetic — no model is called.
    """
    rng = np.random.default_rng(seed)
    quality = rng.choice(SCALE, size=n_items, p=[0.01, 0.01, 0.03, 0.15, 0.80])

    def judge(bias: float) -> list[int]:
        noise = rng.normal(bias, 0.65, size=n_items)
        return [int(np.clip(round(q + e), 1, 5)) for q, e in zip(quality, noise)]

    return judge(0.0), judge(leniency_b)


def pass_fail_table(a: list[int], b: list[int], pass_mark: int) -> np.ndarray:
    """2x2 table [[both-pass, A-pass/B-fail], [A-fail/B-pass, both-fail]]."""
    a_pass = [x >= pass_mark for x in a]
    b_pass = [x >= pass_mark for x in b]
    t = np.zeros((2, 2))
    for pa, pb in zip(a_pass, b_pass):
        t[1 - int(pa), 1 - int(pb)] += 1
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=int, default=300, help="number of scored items")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (reproducible)")
    p.add_argument("--pass-mark", type=int, default=4,
                   help="minimum score that counts as a pass")
    args = p.parse_args(argv)

    a, b = generate_judge_scores(args.items, args.seed)
    cats = SCALE

    print(f"Two LLM judges, {args.items} items, ordinal 1-5 scale (synthetic, offline)")
    print("-" * 68)
    print(f"percent agreement (exact score) : {percent_agreement(a, b):.3f}")
    print(f"Cohen kappa, unweighted         : {cohen_kappa(a, b, cats):.3f}")
    print(f"Cohen kappa, linear weights     : {cohen_kappa(a, b, cats, weights='linear'):.3f}")
    print(f"Cohen kappa, quadratic weights  : {cohen_kappa(a, b, cats, weights='quadratic'):.3f}")
    print(f"Krippendorff alpha (interval)   : {krippendorff_alpha([a, b], level='interval', categories=cats):.3f}")
    m = confusion_matrix(a, b, cats)
    off = [(cats[i], cats[j], int(m[i, j])) for i in range(5) for j in range(5)
           if i != j and m[i, j] > 0]
    top = sorted(off, key=lambda x: -x[2])[:3]
    pairs = ", ".join(f"{x}v{y}:{c}" for x, y, c in top)
    print(f"top disagreement pairs          : {pairs}")
    print()
    print("The weighted-vs-unweighted kappa gap is the near-miss signal: most")
    print("disagreements are adjacent scores, which weighting forgives.")
    print()

    t = pass_fail_table(a, b, args.pass_mark)
    d = paradox.diagnose(t)
    print(f"Binarized at pass mark >= {args.pass_mark} (as leaderboard evals do)")
    print("-" * 68)
    print(f"table [[both-pass, A-only], [B-only, both-fail]]: "
          f"{[[int(v) for v in row] for row in t.tolist()]}")
    print(f"raw pass/fail agreement         : {d['percent_agreement']:.3f}")
    kappa_txt = "undefined" if d["kappa"] is None else f"{d['kappa']:.3f}"
    print(f"kappa on the pass/fail table    : {kappa_txt}")
    print(f"prevalence index                : {d['prevalence_index']:.3f}")
    print(f"bias index                      : {d['bias_index']:.3f}")
    print(f"PABAK                           : {d['pabak']:.3f}")
    print(f"paradox flag                    : {d['paradox']}")
    print(f"reading: {d['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
