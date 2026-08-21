"""Worked example — human raters: sleep-study scoring (100% synthetic data).

Two or more scorers label the same synthetic recordings with AASM sleep stages
(an ordinal scale: W, N1, N2, N3, R), plus a continuous per-subject index scored
by two raters. The example shows the full drill-down: per-pair kappa with
bootstrap CIs, weighted kappa (near-misses to an adjacent stage cost less),
Fleiss' kappa and interval-level Krippendorff alpha across all raters, and
ICC(2,1) vs ICC(3,1) to expose rater calibration bias.

No patient data is read or written anywhere — the generators are seeded and
fully synthetic. The domain layer (stage constants, generators, report) lives
in ``raters.sleep``; the statistics come from the domain-neutral ``raters``
package.

Run from a clean clone:

    python examples/sleep_scoring.py
    python examples/sleep_scoring.py --epochs 300 --raters 3 --seed 12
"""
import sys

from raters import demo

if __name__ == "__main__":
    raise SystemExit(demo.main(sys.argv[1:]))
