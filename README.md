# sleep-scoring-irr

> **Provenance.** Authored on synthetic data, independently of and prior to any
> client engagement. No client data, code, or domain material appears in this
> repository.

Inter-rater reliability (IRR) statistics for sleep-study scoring — the agreement
math a clinical scoring pipeline needs, implemented from first principles and
tested against hand-computed and published reference values.

When two sleep technologists score the same night, *how much do they agree, and
where do they diverge?* "They agreed 91% of the time" is often a lie told by
skewed data. This package computes the chance-corrected statistics that tell the
truth, plus the per-stage drill-down that says **where** the disagreement lives.

```
$ python -m irr.demo

Epoch staging agreement — 200 epochs, 3 raters
------------------------------------------------------------
raters 0-1: percent=0.730  kappa=0.647  weighted-kappa=0.765  CI[0.70,0.82]
raters 0-2: percent=0.725  kappa=0.641  weighted-kappa=0.756  CI[0.70,0.81]
raters 1-2: percent=0.790  kappa=0.723  weighted-kappa=0.803  CI[0.74,0.85]
Fleiss' kappa (all raters): 0.670
Krippendorff alpha (interval): 0.861
Per-stage agreement (where raters diverge most):
    W: 0.500   <- lowest-agreement stage surfaces first
   ...
Top confusion pairs (rater 0 vs 1):
  N1 <-> N2: 9 epochs   <- near-miss; weighted kappa forgives it
  N3 <-> N2: 9 epochs
```

## Why it exists

A regulated clinical system can't ship an agreement number it can't trace. So
every statistic here is:

- **hand-implemented** (numpy only) — no black-box dependency between the raw
  scores and the reported coefficient;
- **tested against a reference** — a published canonical example or arithmetic
  worked out by hand in the test docstring, so the number is provably correct,
  not merely self-consistent;
- **deterministic** — seeded synthetic data, reproducible runs.

The synthetic data is **100% generated** — no patient records are read or
written. Errors are injected the way real scorers disagree: mostly to an
*adjacent* stage (N2↔N3), rarely a far jump (N2↔REM) — which is exactly why
weighted kappa and interval-level alpha belong in the report.

## What's included

| Statistic | Function | Use |
|---|---|---|
| Percent agreement | `percent_agreement` | Report it, never rely on it — blind to chance. |
| Cohen's kappa | `cohen_kappa` | 2 raters, chance-corrected; `weights="linear"`/`"quadratic"` for ordinal stages. |
| Fleiss' kappa | `fleiss_kappa` | 3+ raters over a counts table. |
| Krippendorff's alpha | `krippendorff_alpha` | Any number of raters, **handles missing scores**; nominal or interval. |
| ICC | `icc` | Continuous indices (AHI, latencies); `form="2,1"` absolute agreement vs `"3,1"` consistency. |
| Bootstrap CI | `bootstrap_ci` | Percentile CI for any unit-level statistic — a point estimate without a CI is noise. |

Domain layer (`irr.sleep`): AASM stage constants, synthetic epoch/index
generators, and `analyze_epoch_scores` → a per-pair, per-stage drill-down report.

## The judgment calls it encodes

- **Weighted vs unweighted kappa.** Sleep stages are ordinal, so scoring N2 as
  N3 (adjacent) is a smaller error than N2 as REM. Linear/quadratic weights
  price that in; on near-miss-dominated data weighted kappa is *higher* than
  unweighted, and that gap is the ordinal signal.
- **ICC(2,1) vs ICC(3,1).** A rater who reads every index 2 points high is
  perfectly *consistent* but badly *calibrated*. ICC(3,1) forgives the constant
  bias; ICC(2,1) penalizes it. Choosing the wrong form is the classic ICC
  mistake — both are provided so the difference is visible.
- **The kappa paradox.** With skewed prevalence, 91% raw agreement can sit next
  to a kappa of 0.13. The report always shows both so the number can't mislead.
- **Ordinal spacing for interval alpha.** Pass `categories` in scale order for
  interval/ordinal alpha: the distance between two labels is the gap between
  their positions on the *full* scale, so a category absent from the data still
  holds its place instead of collapsing the spacing.

## Install & run

```bash
pip install -e ".[dev]"     # numpy + pytest
python -m irr.demo          # run the drill-down on synthetic data
pytest -q                   # 36 tests, all against reference values
```

```python
import irr
a = ["W", "N1", "N2", "N2", "N3"]
b = ["W", "N1", "N3", "N2", "N3"]
irr.cohen_kappa(a, b, irr.AASM_STAGES)                    # unweighted
irr.cohen_kappa(a, b, irr.AASM_STAGES, weights="linear")  # ordinal-aware
```

## Validation

`tests/` checks each statistic against a value computed independently of the
code: the standard 2×2 Cohen's-kappa worked example (κ=0.40), a hand-derived
weighted-kappa case (0.625), Fleiss (1/3), Krippendorff's **published canonical
example** with missing data (α_nominal=0.743, α_interval=0.849), and a
constructed ICC case where the two forms diverge (0.769 vs 1.0). CI, invariants
(perfect agreement → 1.0), the kappa paradox, ordinal-scale spacing, and input
validation are covered too.

A note on the "no information" case: a single category, or zero-variance
continuous data, makes every chance-corrected coefficient 0/0 — *undefined*, not
perfect agreement. Those inputs raise `UndefinedStatistic` rather than silently
returning 1.0, which also lets `bootstrap_ci` correctly exclude collapsed
resamples instead of pinning the upper bound to 1.0.

## License

MIT.
