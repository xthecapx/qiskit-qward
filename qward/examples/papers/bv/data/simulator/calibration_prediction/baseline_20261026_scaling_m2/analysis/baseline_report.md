# BV noisy-simulator baseline

Completed 864 executions (1,769,472 shots): 96 configurations × 3 profiles × 3 seeds, with 2,048 shots per execution.

## Predicted error versus final histogram error

For a primary BV circuit, the ideal distribution is a point mass at the known answer t. Consequently, the empirical histogram error is exactly `E_observed = TVD(counts/N, ideal) = 1 - counts[t]/N`. No ideal state simulation is needed. This is the observed error estimate; the stored interval represents its finite-shot uncertainty.

Let Q be the calibration product saved before execution. We compare two explicit hypotheses against that same observed error:

- **M1-success:** `E_predicted = 1 - Q`. This treats Q as a prediction of target success.
- **M1-mixture:** `E_predicted = (1 - Q)(1 - 1/D)`, with D=2^m. This treats Q as visibility in a global uniform-output mixture. At two measured bits, this is `0.75*(1-Q)`.
- **Observed − predicted:** the signed difference between histogram error and predicted error. The table expresses this in **percentage points (pp)**: positive means more error than predicted, negative means less. This difference is not a sigma score; uncertainty-standardized scores are reported separately below.

These quantities are available for every primary BV run in [error_comparison.csv](error_comparison.csv), together with counts, error intervals, interval inclusion, and identifiers linking to the frozen prediction. Injected-fault controls are excluded from this scalar comparison because their hardware reference differs from the nominal task reference.

- **M2-depolarizing:** each native gate with average infidelity r and dimension d scrambles the register with probability `p = r·d/(d−1)`; polarization `λ = Π(1−p)`; readout applies independent bit flips A. Then `P(t) = λ·Π(1−r_ro) + (1−λ)/D` and `E_predicted = 1 − P(t)`. Like M1 it aggregates per-instruction calibration only, but it can never predict more error than a scrambled output (`1 − 1/D`). Older records predate M2; their M2 value is recomputed from the archived pre-run snapshot (`m2_source` column). M2 has no fitted parameters.

| Noise | Bits | L | Observed error | M1 product | Obs − M1 (pp) | M1 mixture | Obs − mix (pp) | M2 depolarizing | Obs − M2 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IDEAL | 2 | 0 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 2 | 2 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 2 | 8 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 2 | 32 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 4 | 0 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 4 | 2 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 4 | 8 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 4 | 32 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 6 | 0 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 6 | 2 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 6 | 8 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 6 | 32 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 8 | 0 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 8 | 2 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 8 | 8 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| IDEAL | 8 | 32 | 0.00% | 0.00% | +0.00 | 0.00% | +0.00 | 0.00% | +0.00 |
| NOMINAL | 2 | 0 | 10.80% | 11.24% | -0.44 | 8.43% | +2.37 | 11.44% | -0.64 |
| NOMINAL | 2 | 2 | 14.63% | 15.81% | -1.18 | 11.86% | +2.77 | 16.08% | -1.45 |
| NOMINAL | 2 | 8 | 22.99% | 24.28% | -1.29 | 18.21% | +4.78 | 24.37% | -1.38 |
| NOMINAL | 2 | 32 | 44.31% | 48.81% | -4.50 | 36.61% | +7.70 | 45.57% | -1.26 |
| NOMINAL | 4 | 0 | 26.24% | 26.56% | -0.32 | 24.90% | +1.34 | 29.95% | -3.71 |
| NOMINAL | 4 | 2 | 38.49% | 39.79% | -1.30 | 37.30% | +1.19 | 44.71% | -6.23 |
| NOMINAL | 4 | 8 | 53.58% | 54.64% | -1.06 | 51.22% | +2.36 | 60.55% | -6.97 |
| NOMINAL | 4 | 32 | 82.44% | 84.30% | -1.87 | 79.04% | +3.40 | 85.84% | -3.40 |
| NOMINAL | 6 | 0 | 39.61% | 40.33% | -0.72 | 39.70% | -0.09 | 45.82% | -6.22 |
| NOMINAL | 6 | 2 | 55.49% | 55.96% | -0.47 | 55.08% | +0.40 | 62.88% | -7.39 |
| NOMINAL | 6 | 8 | 73.79% | 72.66% | +1.13 | 71.52% | +2.26 | 79.61% | -5.82 |
| NOMINAL | 6 | 32 | 94.96% | 94.54% | +0.42 | 93.07% | +1.90 | 96.27% | -1.31 |
| NOMINAL | 8 | 0 | 49.93% | 50.39% | -0.46 | 50.20% | -0.26 | 56.38% | -6.44 |
| NOMINAL | 8 | 2 | 63.99% | 63.95% | +0.04 | 63.70% | +0.29 | 71.06% | -7.07 |
| NOMINAL | 8 | 8 | 84.01% | 82.05% | +1.96 | 81.73% | +2.28 | 88.16% | -4.15 |
| NOMINAL | 8 | 32 | 98.49% | 97.89% | +0.61 | 97.50% | +0.99 | 98.98% | -0.48 |
| STRESS | 2 | 0 | 37.11% | 41.49% | -4.38 | 31.12% | +5.99 | 39.62% | -2.51 |
| STRESS | 2 | 2 | 46.90% | 54.12% | -7.22 | 40.59% | +6.31 | 49.76% | -2.86 |
| STRESS | 2 | 8 | 60.64% | 71.95% | -11.32 | 53.96% | +6.67 | 62.20% | -1.56 |
| STRESS | 2 | 32 | 73.90% | 94.59% | -20.70 | 70.94% | +2.95 | 73.57% | +0.32 |
| STRESS | 4 | 0 | 66.21% | 70.79% | -4.58 | 66.37% | -0.16 | 72.45% | -6.24 |
| STRESS | 4 | 2 | 80.04% | 85.21% | -5.17 | 79.89% | +0.15 | 84.64% | -4.60 |
| STRESS | 4 | 8 | 90.02% | 95.60% | -5.58 | 89.62% | +0.40 | 91.99% | -1.97 |
| STRESS | 4 | 32 | 93.51% | 99.94% | -6.43 | 93.70% | -0.18 | 93.75% | -0.23 |
| STRESS | 6 | 0 | 79.75% | 82.84% | -3.09 | 81.54% | -1.79 | 84.99% | -5.24 |
| STRESS | 6 | 2 | 90.80% | 92.93% | -2.13 | 91.48% | -0.67 | 94.10% | -3.30 |
| STRESS | 6 | 8 | 97.31% | 98.91% | -1.60 | 97.37% | -0.06 | 98.08% | -0.77 |
| STRESS | 6 | 32 | 98.44% | 100.00% | -1.56 | 98.44% | +0.00 | 98.44% | +0.00 |
| STRESS | 8 | 0 | 86.79% | 88.18% | -1.39 | 87.83% | -1.04 | 90.03% | -3.24 |
| STRESS | 8 | 2 | 95.08% | 95.82% | -0.74 | 95.45% | -0.36 | 97.21% | -2.13 |
| STRESS | 8 | 8 | 99.27% | 99.66% | -0.39 | 99.28% | -0.00 | 99.53% | -0.25 |
| STRESS | 8 | 32 | 99.62% | 100.00% | -0.38 | 99.61% | +0.01 | 99.61% | +0.01 |

**Reading STRESS, L=8:** the mixture model predicted 53.96% error; the observed error was 60.64%. Their difference is **+6.67 percentage points**. The mixture prediction underestimated the wrong-answer fraction by that amount. This is not a sigma score or a relative percent change. The product prediction for the same row was 71.95%, giving -11.32 percentage points; that model overestimated error.

Each row averages three secrets × three seeds × the layouts in scope. `L` is the number of forward padding layers before their inverse; larger L increases noise exposure while preserving the ideal BV answer. `STRESS` uses the strongest simulated noise profile. See [error_comparison_by_depth.csv](error_comparison_by_depth.csv). A row average does not have a single per-run sigma value.

### One actual saved run

Configuration `BVCP2-low_predicted_error-ALT-L032`, noise `NOMINAL`, repetition 0: the expected answer `01` occurred **1346/2048** times.

- Before execution: Q=0.631128, so the product hypothesis predicted **36.89% error**.
- After execution: `1 - 1346/2048` gives **34.28% histogram error**.
- Difference: **-2.61 percentage points**. The mixture hypothesis predicted **27.67%**, with residual **+6.61 pp**.
- The stored 99.9965% sampling interval is **[30.01%, 38.73%]**. Product estimate inside: `True`; mixture estimate inside: `False`. These are interval-inclusion diagnostics, not new jointly calibrated hypothesis tests, and they do not include calibration or model uncertainty.

![Predicted versus observed histogram error](expected_vs_observed_error.png)

Points above the diagonal have more observed error than predicted; points below have less. Vertical bars are the stored per-run sampling intervals. Equal scalar error does not imply equal histogram shape: `mixture_prediction_histogram_tvd` separately measures TVD(observed histogram, predicted histogram). The existing `model_status` tests that full mixture distribution, rather than only whether the two error scalars agree.


## Error budget and uncertainty screening

The new screening compares the observed **number of wrong BV answers** with an exact binomial count range calculated from each frozen prediction and the shot count. This is a predictive range for counts, whereas the earlier Clopper–Pearson interval estimates an unknown probability from the observed counts. Neither includes model uncertainty.

For each primary run, error_comparison.csv now includes `*_budget_residual_shot_z`, central predicted count bounds, a separate one-sided `*_budget_upper_count_budget`, exact tail probabilities, and range/budget flags. The signed z-score is `(observed_error - predicted_error)/sqrt(predicted_error*(1-predicted_error)/shots)`. It uses the predicted shot variance, not the plug-in observed variance used in the earlier illustrative calculation. Decisions use exact bounds; a zero-variance prediction has a null z-score and is still tested exactly.

These retrospective diagnostics use alpha=8.2671958e-06 per claim: 0.05 divided by seven claims per primary run and 864 primary runs. Seven claims means a central range and an upper budget for each of the three models, plus one prespecified-target test against uniform background. The adjustment applies to these new checks, not a combined guarantee with legacy tests or future repeated looks. The analysis policy was selected after inspecting this baseline, so these results are exploratory rather than a confirmatory validation.

For the example above, **702 wrong answers** were observed. The product prediction's central range is **659–854**, with upper budget **850**; its signed residual is **-2.45 predicted shot standard deviations**. The mixture range is **478–658**, upper budget **655**, and residual **+6.69**. Which prediction is used changes the conclusion.

| Noise | Runs | Above product upper budget | Below product central range | Above mixture upper budget | BV target resolved vs uniform |
|---|---:|---:|---:|---:|---:|
| IDEAL | 288 | 0 | 0 | 0 | 288 |
| NOMINAL | 288 | 9 | 21 | 65 | 268 |
| STRESS | 288 | 0 | 215 | 53 | 147 |

![Predicted count ranges and observed errors](error_budget_screening.png)

Passing an upper budget does not establish useful signal: an overly pessimistic prediction can pass every run. The separate target test compares the known BV answer with a uniform-background probability of 1/D using an exact upper binomial tail. Uniform background is an explicit hypothesis, not a calibrated per-bin hardware-noise model. An unresolved target remains inconclusive at these shots, not proven absent. Diagnostic controls are excluded from this BV-specific screen.

`uncertainty_review_action` is an exploratory review suggestion based on the product prediction and target test. Unexpectedly low or high error requests prediction review; independently inspect the upper-budget flag for unexpectedly poor runs. Every raw histogram is retained. Legacy 80% success / 0.10 TVD decisions have been removed from current code and exports. Historical raw records retain their original fields for provenance; the current report does not use them.

Before this becomes a hardware acceptance rule, estimate prediction discrepancy from separate calibration windows, freeze the resulting predictive bound, and check coverage and false dismissals on new windows. Three simulator seeds under one stationary noise model per profile do not calibrate drift or hardware prediction uncertainty. A per-bin noise-only screen additionally requires a validated background probability for that bin.

## Outcome classes: combining histogram signal with prediction consistency

The histogram decides whether this run's answer is usable (target resolved above the 1/D background). The M2 prediction decides whether that outcome was expected (observed wrong-answer count inside M2's central count range). `outcome_class` in error_comparison.csv combines the two:

| | Target resolved | Target unresolved |
|---|---|---|
| Within M2 range | `success` | `expected_failure` |
| Above M2 range (more error) | `degraded_success` | `anomalous_failure` |
| Below M2 range (less error) | `success_better_than_predicted` | `unresolved_better_than_predicted` |

| Noise | expected_failure | success | success_better_than_predicted |
|---|---:|---:|---:|
| IDEAL | 0 | 288 | 0 |
| NOMINAL | 20 | 123 | 145 |
| STRESS | 141 | 60 | 87 |

Range status here includes shot noise only. In this simulator the noise is generated from the same calibration the predictor reads, so the classes test the aggregation model, not hardware predictivity. On hardware, widen the range by a model-discrepancy term estimated on separate calibration windows before using these classes as decisions.

## Design and provenance

Scope `scaling`: 96 configurations; measured widths [2, 4, 6, 8]; layouts ['contrasting_predicted_error', 'low_predicted_error']; 864 primary BV executions and 0 diagnostic-control executions. Primary BV patterns are ONES/ALT/SINGLE × L=0/2/8/32. The `contrasting_predicted_error` layout has twice the error rates. Both circuit constructions and result tracking are shared with the IBM runner.

| Profile | Average 1Q infidelity | Average 2Q infidelity | Symmetric readout flip |
|---|---:|---:|---:|
| IDEAL | 0.000% | 0.000% | 0.000% |
| NOMINAL | 0.100% | 1.000% | 2.000% |
| STRESS | 0.500% | 5.000% | 8.000% |

The noise model is built from the same synthetic Target used by the predictor. Aer converts average gate infidelity to local depolarization strength; virtual RZ has no noise, readout is symmetric and independent, and thermal/idle noise is disabled. Routing and native decomposition are included. GenericBackendV2 is a local synthetic device, not a live IBM calibration snapshot.

Methods: {'stabilizer': 864}. Stabilizer is used for BV and Clifford controls; matrix-product-state handles the π/6 rotation diagnostic. No dense statevector or density-matrix method is used. RNG seed base: 20261026; exact per-execution seeds are in raw records. Repetitions share stationary calibration and must not be described as hardware drift windows.

Every prediction and calibration snapshot was archived before its local simulation. All 288 prepared-record/QPY/noise-model hashes have been verified. See the top-level manifest, raw JSON records, QPY files, and noise-model JSONs. The analysis reads those saved counts without rerunning shots or fitting a new prediction.

## Prediction accuracy on primary BV only

| Profile | BV runs | M1 product MAE | M1 mixture MAE | M2 depolarizing MAE | M1 mixture histogram mismatches | M2 histogram mismatches |
|---|---:|---:|---:|---:|---:|---:|
| IDEAL | 288 | 0.00% | 0.00% | 0.00% | 0/288 | 0/288 |
| NOMINAL | 288 | 1.47% | 2.30% | 4.01% | 245/288 | 162/288 |
| STRESS | 288 | 4.80% | 1.89% | 2.27% | 138/288 | 104/288 |

| Profile | Bits | Runs | M1 product MAE | M1 mixture MAE | M2 MAE | Runs within M1 / M2 count range |
|---|---:|---:|---:|---:|---:|---:|
| IDEAL | 2 | 72 | 0.00% | 0.00% | 0.00% | 72 / 72 |
| IDEAL | 4 | 72 | 0.00% | 0.00% | 0.00% | 72 / 72 |
| IDEAL | 6 | 72 | 0.00% | 0.00% | 0.00% | 72 / 72 |
| IDEAL | 8 | 72 | 0.00% | 0.00% | 0.00% | 72 / 72 |
| NOMINAL | 2 | 72 | 1.89% | 4.40% | 1.24% | 63 / 72 |
| NOMINAL | 4 | 72 | 1.56% | 2.08% | 5.08% | 65 / 27 |
| NOMINAL | 6 | 72 | 1.25% | 1.46% | 5.20% | 67 / 20 |
| NOMINAL | 8 | 72 | 1.16% | 1.24% | 4.54% | 65 / 24 |
| STRESS | 2 | 72 | 10.90% | 5.49% | 1.88% | 20 / 70 |
| STRESS | 4 | 72 | 5.44% | 0.81% | 3.33% | 18 / 40 |
| STRESS | 6 | 72 | 2.09% | 0.77% | 2.40% | 15 / 44 |
| STRESS | 8 | 72 | 0.75% | 0.47% | 1.45% | 20 / 47 |

MAE is the mean absolute observed-minus-predicted error, expressed in probability units (1% here means one percentage point). Full histogram mismatches also examine how probability is distributed among wrong answers; that is a different check from comparing total error.

## Diagnostic checks

- Deliberately wrong BV peaks: observed target success rates []; rank statuses {}. A well-resolved winner can be the wrong answer.
- Uniform-output controls: observed TVD range None. These circuits are intended to produce uniform outputs; a BV target-signal rule does not apply to them.
- GHZ Z/X controls and SPAM/fault treatments remain in jobs.csv and raw records because they test readout, reference alignment, and failure modes of histogram screening.

## Scope and remaining validation

The experiment uses the existing BV builder, controlled padding, the M1 product and uniform-mixture predictions, the parameter-free M2 depolarizing aggregation, exact count budgets, and archived simulator/IBM results. No learned predictor or additional idle/mirror campaign is included. Raw counts and frozen predictions are preserved.

The count budgets are conditional on each fixed prediction and account for shot variation only. Before automatic hardware triage, prediction uncertainty needs calibration and validation across independent time windows. Per-bin noise screening also needs a validated background model. This baseline does not justify automatic histogram deletion.

Artifacts: [execution manifest](../manifest.json), [job table](jobs.csv), [error comparison](error_comparison.csv), [metrics](baseline_metrics.json), [error-budget SVG](error_budget_screening.svg).
