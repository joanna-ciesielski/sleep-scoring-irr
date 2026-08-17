# rater-agreement

> **Provenance.** Authored on synthetic data, independently of and prior to any
> client engagement. No client data, code, or domain material appears in this
> repository.

Chance-corrected agreement statistics for any raters — human or model.

When two raters score the same items, how much do they agree, and is that
agreement better than chance? The question is the same whether the raters are
two clinicians scoring the same recordings, two annotators labeling a corpus,
or two LLM judges grading the same benchmark answers — and so is the trap:
**a raw agreement percentage is often a lie told by skewed data.** This package
computes the statistics that tell the truth, implemented from first principles
in pure numpy and tested against hand-computed and published reference values.

It also covers the case classical IRR toolkits mostly skip and LLM evaluation
newly needs: **intra-rater reliability** — does one rater agree with *itself*
on repeat measurement? Multi-sampling a judge (same model, same input, N runs)
is exactly a repeated-measurement design, and it gets the matching statistics:
test-retest agreement, ICC(1,1), flip rates, and a sample-size planner.

## Why chance correction matters

Suppose two raters label 100 items pass/fail: 90 both-pass, 1 both-fail, 9
split. Raw agreement is 91% — but because nearly everything passes, two raters
answering "pass" at random would agree almost as often, and Cohen's kappa is
0.13. This is the **kappa paradox** (Feinstein & Cicchetti 1990): high raw
agreement, low true concordance — and it is precisely the failure mode of
LLM-judge evaluations that report "our judges agree 91% of the time" on
skewed pass rates. `raters.paradox.diagnose` computes the work-up:

```python
>>> from raters import paradox
>>> paradox.diagnose([[90, 5], [4, 1]])   # -> kappa 0.13, PABAK 0.82, paradox: True
```

## What it computes

| Statistic | Function | Use |
|---|---|---|
| Percent agreement | `percent_agreement` | Report it, never rely on it — blind to chance. |
| Cohen's kappa | `cohen_kappa` | 2 raters, chance-corrected; `weights="linear"`/`"quadratic"` for ordinal scales. |
| Fleiss' kappa | `fleiss_kappa` | 3+ raters over a counts table. |
| Krippendorff's alpha | `krippendorff_alpha` | Any number of raters, **handles missing scores**; nominal or interval. |
| ICC(2,1), ICC(3,1) | `icc` | Continuous scores; absolute agreement vs consistency. |
| Bootstrap CI | `bootstrap_ci` | Percentile CI for any unit-level statistic. |
| Test-retest agreement | `intra.test_retest_agreement` | One rater vs itself across repeats. |
| ICC(1,1) | `intra.icc_1_1` | Item identity vs run-to-run noise, one-way ANOVA. |
| Within-item variance | `intra.within_item_variance` (+`_ci`) | The rater's noise floor, with bootstrap CI. |
| Flip rate | `intra.flip_rate` | Items whose label — or pass/fail **decision** — changes across repeats. |
| Sample-size planner | `intra.min_samples_for_ci` | Repeats needed for a target CI width on an item mean. |
| Prevalence & bias index | `paradox.prevalence_index`, `paradox.bias_index` | Why kappa and raw agreement diverge (2×2). |
| PABAK | `paradox.pabak` | Kappa under balanced marginals (Byrt, Bishop & Carlin 1993). |
| Paradox work-up | `paradox.diagnose` | All of the above plus a plain-language flag. |

```python
import raters

a = [4, 5, 5, 3, 5]                # judge A, ordinal 1-5
b = [5, 5, 4, 3, 5]                # judge B
raters.cohen_kappa(a, b, [1, 2, 3, 4, 5])                        # unweighted
raters.cohen_kappa(a, b, [1, 2, 3, 4, 5], weights="quadratic")   # ordinal-aware

runs = [[4, 5, 4], [3, 3, 3], [5, 5, 4]]   # one judge, 3 items x 3 repeats
raters.intra.flip_rate(runs)                       # 2/3 of items wobble
raters.intra.flip_rate(runs, threshold=3.5)        # ...but do decisions flip?
```

## Two worked examples

Both run **offline on seeded synthetic data — no API keys, no model calls, no
patient data**. That constraint is deliberate: every number in the output
reproduces from a clean clone.

**Model raters** — `python examples/llm_judges.py`: two LLM judges score 300
items on an ordinal 1–5 scale. Shows the near-miss signal (unweighted kappa
0.20 vs quadratic-weighted 0.57 — disagreements are mostly 4-vs-5), then
binarizes at a pass mark the way leaderboard evals do: raw pass/fail agreement
93% but kappa 0.51 with prevalence index 0.85 — the paradox, flagged and
explained by `diagnose`.

**Judge stability** — `python examples/judge_stability.py`: one judge, 60
items × 20 repeats. Label flip rate 0.97 but decision flip rate 0.38 — the
wobble that matters is the one that crosses the pass mark. ICC(1,1) 0.80,
within-item variance with CI, and the budget answer: 19 runs per item for a
0.5-wide 95% CI on a mean score.

**Human raters** — `python examples/sleep_scoring.py`: the original domain
example, two-plus scorers staging synthetic sleep-study epochs on the ordinal
AASM scale, with per-pair kappas, per-stage drill-down, and ICC calibration
gap. 100% generated data; the domain layer lives in `irr.sleep`, not in the
core package.

## Design decisions

- **Pure numpy, hand-implemented.** No black-box dependency sits between the
  raw scores and the reported coefficient. Every statistic is short enough to
  audit and is pinned by a test against a reference value computed *outside*
  the code — a published canonical example or arithmetic worked by hand in the
  test docstring.
- **Weighted vs unweighted kappa.** Ordinal scales make adjacent disagreements
  cheaper than distant ones; the weighted/unweighted gap is itself diagnostic
  (large gap = near-miss-dominated disagreement).
- **ICC forms are explicit.** ICC(2,1) penalizes a rater's constant bias,
  ICC(3,1) forgives it, ICC(1,1) is the intra-rater form; choosing the wrong
  one is the classic ICC mistake, so all three are separate and tested. The
  inter-rater `icc()` deliberately rejects `form="1,1"` — repeats of one rater
  are not interchangeable raters.
- **Undefined is not 1.0.** A single category or zero variance makes every
  chance-corrected coefficient 0/0. Those inputs raise `UndefinedStatistic`
  rather than returning perfect agreement, which also lets `bootstrap_ci`
  exclude collapsed resamples instead of inflating the interval.
- **Backward compatibility.** The original `irr` package still imports and
  works; `irr.stats` re-exports from `raters`.

## Limitations

Things this package deliberately does not do:

- **No Bayesian agreement models** — point estimates and bootstrap CIs only;
  no posterior over kappa, no hierarchical rater models.
- **No missing-data imputation** beyond what Krippendorff's alpha handles
  natively (pairable-value coincidences). Fleiss' kappa and ICC require
  complete tables.
- **No multi-label support** — each rater gives one label per item.
- **Paradox diagnostics are 2×2 only.** PI, BI, and PABAK are implemented as
  published; the multi-category generalization is not included.
- **`min_samples_for_ci` is a normal-approximation planning estimate**, not a
  guarantee — it assumes the pilot variance is representative.
- **Not a general statistics library.** It is a small, auditable,
  dependency-light implementation of chance-corrected agreement for settings
  where you must be able to show your work; for everything else use pingouin
  or statsmodels.

## Install & validate

```bash
pip install -e ".[dev]"     # numpy + pytest
pytest -q                   # 62 tests, all against reference values
python examples/llm_judges.py
python examples/judge_stability.py
python examples/sleep_scoring.py
```

## Validation

Every non-trivial statistic is checked against a value computed independently
of the code: the standard 2×2 Cohen's-kappa worked example (κ=0.40), a
hand-derived weighted-kappa case (0.625), Fleiss (1/3), Krippendorff's
published canonical example with missing data (α_nominal=0.743,
α_interval=0.849), a constructed ICC case where the forms diverge (0.769 vs
1.0), a hand-worked one-way ANOVA for ICC(1,1) (15/17), the Feinstein &
Cicchetti (1990) paradox pair (85% agreement, κ 0.70 vs 0.32), and the Byrt,
Bishop & Carlin (1993) identity κ = (PABAK − PI² + BI²)/(1 − PI² + BI²).
Invariants, undefined-case behavior, and input validation are covered too.

## References

- Byrt, T., Bishop, J. & Carlin, J. B. (1993). Bias, prevalence and kappa.
  *J Clin Epidemiol* 46(5), 423–429.
- Feinstein, A. R. & Cicchetti, D. V. (1990). High agreement but low kappa: I.
  The problems of two paradoxes. *J Clin Epidemiol* 43(6), 543–549.
- Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability.

## License

MIT.
