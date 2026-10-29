# BV calibration-prediction experiment

This is a separate small campaign using the existing QWARD BV implementation,
`BVExperimentConfig` patterns, and a `BVIBMExperiment` subclass. Its results use
the existing `batch_summary` / `individual_results` JSON structure. The scientific
design is in [the plan](../../../../docs/calibration_histogram_experiment_plan.md).

## Start here: what exists and how to use it

These are experimental APIs in this working tree, not a validated hardware filter.

| Component | Input | What it tells us |
|---|---|---|
| `capture_calibration` | Exact transpiled circuit and backend | Errors/durations for its physical instructions, readout, coherence data, coverage and timestamps |
| `CircuitErrorPrediction` | That circuit, frozen calibration and analytic reference | Product-as-success and uniform-mixture predictions, before execution |
| `HistogramAssessment` | The same circuit/prediction, known reference and counts | Observed error, shot interval, conditional error budget, mixture mismatch and winner-rank evidence |
| `binomial_prediction_check` | Event count, shots and a supplied fixed probability | Exact count bounds/tails and a signed deviation in predicted shot standard deviations |
| `minimum_binomial_shots` | Null/alternative probabilities, alpha and desired power | Shot planning under those explicit assumptions |

The experiment runners archive the original circuits, calibration, predictions,
job IDs and raw results. The offline analysis compares those frozen predictions
with counts. No automatic pre-submission rejection or per-bin noise removal is
implemented. The gate/readout product does not yet model idle noise, coherent
errors or crosstalk; timing/coherence features are recorded separately.

### Run a read-only example using the saved baseline

From the repository root, using the project's installed environment:

```bash
.venv/bin/python -m qward.examples.papers.bv.bv_calibration_demo
.venv/bin/python -m qward.examples.papers.bv.bv_calibration_demo --profile STRESS
```

The [demo source](bv_calibration_demo.py) replays the prediction from the archived
pre-run calibration, checks that it equals the saved prediction, and analyzes one
saved histogram. It does not connect to IBM, generate shots or modify records.
Choose `--layers 0`, `2`, `8` or `32`, and `--run-index 0`, `1` or `2` to inspect
other saved ALT configurations. The default is L=32, repetition 0.

| Same BV answer `01`, 2,048 shots | NOMINAL | STRESS |
|---|---:|---:|
| Correct counts | 1,347 | 562 |
| Predicted error, product | 36.89% | 90.01% |
| Observed error | 34.23% | 72.56% |
| Observed − predicted | −2.66 percentage points | −17.46 percentage points |
| Difference in predicted shot standard deviations | −2.49 | −26.35 |
| Product predicted central range of wrong counts | 671–841 | 1,789–1,895 |
| Observed wrong counts | 701 | 1,486 |
| BV target above uniform background at the archive's adjusted threshold | Resolved | Unresolved |

The nominal histogram retains a clear BV target signal and its total error is
compatible with the product prediction. The stress histogram is close to uniform
and its target is unresolved at this adjusted threshold; the product also predicts
much more error than observed. In both examples the full uniform-mixture histogram
hypothesis fails its check. These are separate findings about output signal and
prediction accuracy. The demo reports both prediction hypotheses and explicitly
labels its retrospective tests and assumed background.

### Estimate from an IBM backend without submitting a job

With a configured IBM account, replace `YOUR_BACKEND` with an accessible backend:

```python
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qward.algorithms import BernsteinVazirani
from qward.metrics import CircuitErrorPrediction
from qward.metrics.circuit_error_prediction import capture_calibration

backend = QiskitRuntimeService().backend("YOUR_BACKEND")
bv = BernsteinVazirani("10")
reference = {bv.expected_outcomes[0]: 1.0}  # displayed answer: "01"
isa = generate_preset_pass_manager(
    backend=backend, optimization_level=3, scheduling_method="alap"
).run(bv.circuit)

snapshot = capture_calibration(isa, backend)
prediction = CircuitErrorPrediction(
    isa, snapshot=snapshot, reference=reference
).get_metrics()
if prediction.m1_success is None:
    raise ValueError("Calibration is incomplete; a full prediction is unavailable")
print("Predicted error, product:", 1 - prediction.m1_success)
print("Predicted error, mixture:", prediction.mixture_tvd)
```

This performs classical transpilation and metadata retrieval only. It does not
submit a QPU job. The unpadded BV example uses optimization level 3; the controlled
padding experiment defaults to 0 to preserve its intended noise exposure.
For tracked preparation, `BVCalibrationIBMExperiment.prepare(...)` already saves
the ISA QPY, snapshot and prediction without submission. The CLI's
`--preflight-only` performs structural checks; it does not fetch IBM calibration.

### Analyze counts from that same circuit afterward

Keep the original `isa`, `reference`, and `prediction`; do not replace the
prediction with a newer backend calibration after seeing the result.

```python
from qward.metrics import HistogramAssessment
from qward.metrics.histogram_assessment import binomial_prediction_check

# counts must be raw counts from executing this exact ISA circuit.
assessment = HistogramAssessment(
    isa, counts=counts, reference=reference, prediction=prediction,
    alpha=0.01,  # illustrative per-check allocation; budget across your claims
).get_metrics()
print("Observed error:", assessment.tvd)
print("Error interval:", assessment.tvd_interval)
print("Count budget:", assessment.error_budget)

target = bv.expected_outcomes[0]
target_count = sum(v for k, v in counts.items() if k.replace(" ", "") == target)
signal = binomial_prediction_check(
    target_count, sum(counts.values()), 2.0 ** -isa.num_clbits, alpha=0.01
)
print("Target above uniform background:", signal.exceeds_upper_budget)
```

Use the count budget to investigate prediction disagreement and the target test
to assess whether the known answer rises above the assumed background. Neither
labels individual shots as noise. A calibrated prediction uncertainty and a
validated background model remain necessary before automatic triage.

## Error models compared

All three aggregate per-instruction calibration of the exact ISA circuit; none
simulates the circuit or fits parameters. For BV with known answer t and D=2^m:

| Model | Predicted error | Assumption |
|---|---|---|
| M1-success | `1 - Q`, `Q = Π(1-r_gate)·Π(1-r_ro)` | every error gives a wrong answer |
| M1-mixture | `(1 - Q)(1 - 1/D)` | Q is the visibility of a uniform mixture |
| **M2-depolarizing** | `1 - [λ·Π(1-r_ro) + (1-λ)/D]`, `λ = Π(1 - r·d/(d-1))` | gate errors scramble the register; readout flips bits independently |

`r` is an average gate infidelity, so the depolarizing probability of a
d-dimensional gate is `r·d/(d-1)`. A scrambled register still yields t with
probability 1/D, so M2 never exceeds `1 - 1/D`; M1 can (90% predicted at 2 bits
where 75% is the maximum). `CircuitErrorPrediction` freezes M2 as
`m2_success`, `m2_tvd` and `m2_probabilities`; `HistogramAssessment` reports
`m2_error_budget` and `m2_model_status`. Records frozen before M2 are given M2
by the offline analysis from their archived snapshot (`m2_source`).

### Deciding success or failure

The histogram decides whether the answer is usable; the prediction decides
whether that outcome was expected. `outcome_class` in `error_comparison.csv`:

| | target resolved above 1/D | target unresolved |
|---|---|---|
| within M2 count range | `success` | `expected_failure` |
| above M2 range | `degraded_success` | `anomalous_failure` |
| below M2 range | `success_better_than_predicted` | `unresolved_better_than_predicted` |

Ranges include shot noise only. On hardware, add a model-discrepancy allowance
estimated on separate calibration windows before using classes as decisions.

## Hardware findings so far (ibm_fez, 2026-10-26/27)

Campaigns `bv-ibm-fez-1` (2 jobs) and `bv-ibm-fez-2` (8 jobs, one Batch), 4-bit
ALT BV, 2,048 shots, layouts chosen by `--rank-layouts`: low `108-112`,
contrasting `126 127 137 147 146`. Re-prediction from `job.properties()` equaled
the frozen prediction for every job (calibration drift on these qubits = 0), so
all gaps below are model error.

| Setting | Low: observed (M1 / M2 predicted) | Contrasting: observed (M1 / M2) |
|---|---|---|
| L=0 | 18.4% (6.7 / 8.5) | 20.1% (16.6 / 17.3) |
| L=8, fez-1 | 59.6% (16.4 / 21.3) | 44.4% (33.3 / 40.3) |
| L=8, fez-2 ×2 | 39.7%, 40.9% | 44.7%, 42.6% |
| L=8, gate twirling | 48.2% | 49.8% |

- The BV target was resolved above 1/D in all 10 jobs.
- Every model underpredicts error, by 3-43 pp; M2 is closest on the stable
  contrasting layout (within range on 2 of 3 L=8 runs and on L=0).
- The low layout changed between 00:21 and 00:54 UTC (59.6% -> 40%, p ~ 1e-37)
  with no change in reported calibration; the contrasting layout repeated
  (p = 0.85). The apparent layout-ranking inversion came from that unstable run.
- The gap is already present at L=0 on the low line, so part of it is not
  padding-related.
- Gate twirling increased error by 5-8 pp; coherent error is not the main cause.
- Wrong answers are mostly 1-2 bit flips, not global scrambling.

Interpretation: calibration-product predictions are useful as a hardware
health check (observed far above predicted = device worse than reported), not
yet as an absolute error estimate or layout selector. The next campaign,
`bv-ibm-fez-3`, repeats L=0 and L=8 on both layouts over three windows to
measure the run-to-run spread that a pass/fail tolerance requires.

## Campaign design

The primary sweep has 48 configurations: 2 measured widths (2, 4), 3 secrets
(ONES, ALT, SINGLE), 4 padding lengths (0, 2, 8, 32), and 2 layout labels. Another
52 configurations are SPAM, GHZ, uniform-output, and deliberately perturbed BV
controls. Padding seed 11 is shared across patterns; its inverse preserves the
ideal answer. The original library BV circuit is the L=0 baseline. Its expected
displayed answer is `secret_string[::-1]`; the ancilla is not measured.

## Inspect and schedule without execution

From the repository root:

```bash
uv run -m qward.examples.papers.bv.bv_calibration_ibm --list

uv run -m qward.examples.papers.bv.bv_calibration_ibm \
  --preflight-only --config BVCP2-low_predicted_error-ALT-L000

uv run -m qward.examples.papers.bv.bv_calibration_ibm \
  --plan --smoke-test --plan-file /tmp/bv-smoke-schedule.json
```

These operations do not authenticate to IBM, submit jobs, execute a simulator,
or generate measurement counts. `--plan` produces randomized configuration order
for each window; `--smoke-test` selects 25 configurations and one window. Without
that filter the schedule has 100 configurations across 3 windows. At 2,048 shots,
the respective budgets are 51,200 and 614,400 shots.

Before hardware use, choose the backend and two contrasting physical allocations
for each circuit width using calibration only. A layout must include **all**
logical qubits: 3 or 5 entries for BV, 2 or 4 for the other controls. A layout label
identifies an allocation class; it does not automatically pick physical qubits.
Keep compatible physical data qubits across controls where feasible, and save
each selected allocation. Layouts can change between calibration windows.

## Run the noisy-simulator baseline first

```bash
uv run -m qward.examples.papers.bv.bv_calibration_simulator \
  --scope smoke --profiles IDEAL NOMINAL STRESS \
  --shots 2048 --runs 3 --seed 20261022
```

This executes 25 configurations × 3 profiles × 3 seeds = **225 executions and
460,800 shots**, locally. `--scope scaling` runs only primary BV at 2, 4, 6 and 8
measured bits on both layouts (96 configurations, 864 executions); the 6- and
8-bit configurations are simulator-only and do not change the IBM matrix. It saves results under
`data/simulator/calibration_prediction/baseline_TIMESTAMP_UNIQUEID/`, keeping
simulator records separate from QPU records. `--scope core` selects all 100
configurations; `--output-dir` selects a fresh explicit destination.

The noise profiles use synthetic average 1Q/2Q infidelities and symmetric readout
errors: IDEAL=(0,0,0), NOMINAL=(0.001,0.01,0.02), STRESS=(0.005,0.05,0.08). Native
RZ is noiseless; thermal/idle errors are disabled in this first controlled model.
The second allocation, when included, has twice those error rates. Noise channels
are generated from the same synthetic Target used by the predictor, using Aer's
average-infidelity-to-depolarization conversion. The exact noise-model JSON is
archived and hash-checked before execution.

The runner reuses circuit preparation, calibration snapshots, predictions, QPY
artifacts, assessment, and rich result construction from the IBM campaign.
Records explicitly say `execution_type=NOISY_SIMULATOR` and
`backend_type=simulator`; Aer job IDs are local. Stabilizer trajectories handle
BV and Clifford controls; matrix-product-state trajectories handle the π/6
diagnostic. Dense statevector and density-matrix methods are not used.

The run produces raw JSON records, a manifest, a CSV table, a Markdown report,
and PNG/SVG comparisons. Analysis can be regenerated without rerunning shots:

```bash
uv run -m qward.examples.papers.bv.bv_calibration_baseline_report BASELINE_DIRECTORY
```

The first completed baseline is
[`baseline_20261022_smoke_v2`](data/simulator/calibration_prediction/baseline_20261022_smoke_v2/analysis/baseline_report.md).
It contains 108 primary BV executions and 117 diagnostic executions. A same-seed
rerun reproduced all 225 histograms; duplicate rerun counts are excluded from the
published baseline. Every frozen circuit/prediction and noise-model hash was
verified. These are stationary synthetic-noise repetitions, not hardware drift
windows. No IBM job has been submitted.

## IBM workflow: predict from reported calibration, then compare

Credentials come from `IBM_QUANTUM_*` in `.env`. Steps 1–2 never submit a job.

```bash
# 1. Rank physical lines by predicted error from the backend's current calibration.
uv run -m qward.examples.papers.bv.bv_calibration_ibm --rank-layouts \
  --config BVCP4-low_predicted_error-ALT-L008 --backend ibm_fez > layouts.json

# 2. Freeze circuit + calibration + M1/M2 prediction (prints predicted errors).
uv run -m qward.examples.papers.bv.bv_calibration_ibm --prepare-only \
  --config BVCP4-low_predicted_error-ALT-L008 --backend ibm_fez \
  --layout Q0 Q1 Q2 Q3 Q4 --campaign-id bv-ibm-1 --planned-executions 25

# 3. Submit that exact frozen record; exits once the job ID is saved.
#    A recalibration since step 2 is warned about and flagged, not refused.
uv run -m qward.examples.papers.bv.bv_calibration_ibm --submit-prepared RECORD_JSON --runs 1

# 4. When IBM shows the job finished, retrieve it (never resubmits; safe to repeat).
uv run -m qward.examples.papers.bv.bv_calibration_ibm --recover RECORD_JSON

# 5. Compare frozen predictions with observed counts.
uv run -m qward.examples.papers.bv.bv_calibration_analysis \
  --data-dir qward/examples/papers/bv/data/qpu/calibration_prediction/raw --output-dir OUT
```

CLI output is a compact summary; add `--full-json` to print the whole record
(the file always keeps everything). `--submit-prepared --wait` blocks up to
`--timeout` seconds instead of exiting.

### Several jobs in one Batch

Write a plan (example: `data/qpu/calibration_prediction/plans/bv-ibm-fez-2.json`),
then:

```bash
uv run -m qward.examples.papers.bv.bv_calibration_ibm --prepare-batch PLAN.json   # no jobs
uv run -m qward.examples.papers.bv.bv_calibration_ibm --submit-batch PLAN_manifest.json
uv run -m qward.examples.papers.bv.bv_calibration_ibm --recover-batch PLAN_manifest.json
```

Plans may set `"window"`; prepare each window just before submitting it so
its calibration is current. All records are prepared from one calibration read and validated before any
job is sent; jobs share one Runtime Batch whose ID is stored in the manifest
and every record. Items may set `"twirl_gates": true` (separate `arm`).

Single-record `--submit-prepared` uses Runtime job mode: each repetition is one job with one circuit,
no Batch or Session. Repetitions (`--runs N`) are N independent jobs.

What is captured from IBM before submission: per-instruction `gate_error` and
duration for the exact physical gates, `readout_error` plus directional
`prob_meas1_prep0` / `prob_meas0_prep1` (used by M2), T1/T2, ALAP idle times, and
the full `properties()` record with its `last_update_date`. Preparation refuses
circuits on qubits/edges IBM reports as faulty.

After execution, `collect` also reads `job.properties()` (calibration in effect
when the job ran) and stores a diagnostic re-prediction per job. The export
reports `calibration_drift_m1/m2` = run-time minus frozen predicted success, so
a residual can be split into drift (calibration changed) and model error (the
model is wrong even with current calibration). The frozen prediction stays the
primary, pre-registered one.

Layout ranking: `low_predicted_error` takes the best lines; the contrasting
label takes lines near twice the best predicted error and excludes lines with
any instruction error ≥ 0.5. Save `layouts.json` with the campaign.

## Hardware execution, when requested

Submission requires `--submit`. Example syntax (replace placeholders with the
chosen backend and physical indices):

```bash
uv run -m qward.examples.papers.bv.bv_calibration_ibm \
  --submit --config BVCP2-low_predicted_error-ALT-L000 \
  --backend BACKEND_NAME --layout DATA_0 DATA_1 ANCILLA \
  --window 1 --campaign-id bv-calibration-smoke \
  --planned-executions 25 --opt-levels 0 --runs 1 --shots 2048
```

For the full study use a distinct campaign ID, `--planned-executions 100`, and
windows 1, 2, and 3. Follow the generated randomized order; this runner executes
one configuration per invocation, like the neighboring BV campaigns. Counts of
repeated jobs contribute to the declared window budget. Increase that budget
before preparation if adding repetitions; it determines the confidence allocation.
Use a single sequential campaign process to preserve ordering and budget accounting.

The existing `IBM_QUANTUM_*` credential resolution is reused; credentials are not
written to result files. Optimization defaults to 0 to preserve the exposure
treatment. The runner rejects padded circuits whose compiled two-qubit count
does not exceed the unpadded baseline. ALAP scheduling provides idle features.
Gate/measurement twirling and dynamical decoupling are explicitly disabled for
the baseline; initialization is enabled. Other Runtime defaults are recorded as
unknown server defaults when unset. These settings can be tested separately in a
later campaign; they should not change silently within this study.

Each invocation first writes logical and ISA QPY files, a complete calibration
snapshot, analytic nominal/injected references, package versions, compiler and
Runtime settings, a prediction, and integrity hashes. It then submits that same
archived ISA circuit. Predictions are based on scalar average calibration errors;
they are hypotheses to validate, not state fidelities or guaranteed error bounds.

## Results and recovery

Default location:

```text
qward/examples/papers/bv/data/qpu/calibration_prediction/raw/
  CONFIG_IBM_BACKEND_TIMESTAMP_UNIQUEID.json
  artifacts/CONFIG_IBM_BACKEND_TIMESTAMP_UNIQUEID/
    logical.qpy
    isa.qpy
    prepared.json
```

The top-level JSON is an atomic submission/retrieval journal and rich experiment
result. It includes `config`, `campaign_id`, `window`, `run_id`, `batch_id`,
`submission_journal`, `batch_summary`, and `individual_results`. Individual results
include `experiment_id`, `job_id`, counts, QWARD metrics, prediction, and
`histogram_assessment`. The assessment reports observed error, its interval, model mismatch, rank
evidence, and conditional error budgets separately. No quantum-advantage verdict is inferred from above-chance
BV success. Legacy batch statistics are descriptive; the new assessment contains
the task-confidence intervals.

### Error budgets and uncertainty units

For a deterministic reference matching the prediction's reference,
`HistogramAssessment.get_metrics().error_budget` now compares the number of wrong
answers with a binomial prediction. It reports a central count range, a separate
one-sided upper count budget, exact tail probabilities, and a signed residual in
predicted shot standard deviations. `Scanner` exports these with the
`error_budget_` prefix. The schema is version 1.2; existing raw records remain in
their original schema.

```python
from qward.metrics.histogram_assessment import binomial_prediction_check

# Conditional on a noise-only probability fixed independently of these counts:
check = binomial_prediction_check(2, 1000, predicted_probability=0.002, alpha=0.05)
print(check.predicted_count_lower, check.predicted_count_upper)  # 0, 5
print(check.upper_tail_pvalue)  # approximately 0.5943
```

Two occurrences are compatible with this background; frequency alone cannot
classify them as noise. A supplied probability can instead describe total BV
error, in which case `count` is the number of wrong answers. The helper assumes
independent shots and a stationary, fixed predicted probability. Its range covers
sampling variation, **not** calibration/model uncertainty or drift. An estimated
probability fitted to the same counts is not an independent prediction.

The offline exporter derives the new checks from frozen predictions and saved
counts, including old records. `error_comparison.csv` adds both M1 hypotheses'
`*_budget_*` fields, `target_signal_*`, and `uncertainty_review_action`. A separate
Bonferroni allocation of `0.05/(5 * completed_primary_runs)` covers the five new
claims per run: two central ranges, two upper budgets, and a BV-target test against
uniform background. Analyze a fixed archive; this is not a sequential stopping
rule. Retrospective baseline results are exploratory.

The generic metric receives its alpha from the caller, who must allocate it for
the chosen claims; its central range and upper budget are separate claims. The
new derived exporter policy is separate from the original campaign's legacy
interval/rank tests. Historical raw records retain `task_status` and `recommended_action` for
provenance. Current code and exports omit these earlier arbitrary-tolerance
decisions; review suggestions use uncertainty checks.

Use the upper-budget flag to find unexpectedly poor runs, and the central range
to investigate surprises in either direction. An unresolved BV-target test means
inconclusive at the current shot count. No data are deleted. Before automatic
hardware triage, calibrate prediction discrepancy on separate time windows and
validate predictive coverage and false dismissals on untouched windows. Per-bin
noise screening requires a justified background probability for each tested bin;
the present uniform-target check is only an explicit diagnostic hypothesis.

Recovery requires the saved top-level result JSON:

```bash
uv run -m qward.examples.papers.bv.bv_calibration_ibm \
  --recover PATH_TO_RESULT_JSON --timeout 600
```

It retrieves saved job IDs and verifies the original artifacts. It never
resubmits or rebuilds a prediction using newer calibration. Retrieval failures
leave the record incomplete and retain known job IDs/counts. If submission was
interrupted before an ID was saved, the journal retains its batch ID, run UUID
job tag and repetition tag for reconciliation in IBM; an unknown submission is
not retried automatically. Keep the raw JSON and its artifact directory together.
The prepared hash detects accidental modifications; it is not an authenticity
signature against someone who can replace both files and hashes.

## Offline analysis after results exist

```bash
uv run -m qward.examples.papers.bv.bv_calibration_analysis \
  --data-dir qward/examples/papers/bv/data/qpu/calibration_prediction/raw \
  --output-dir /tmp/bv-calibration-analysis
```

This writes `jobs.csv` and `summary.json`. It excludes unfinished jobs, rejects
duplicate jobs, and keeps controls separate. Use a separate raw output directory (`--output-dir` on the runner)
for each campaign/backend if comparing multiple campaigns.

The export also writes `error_comparison.csv`, restricted to primary BV runs so
the prediction and measured error use the same ideal target. It contains the
pre-run predicted histogram errors (`1-Q` under M1-success and
`(1-Q)*(1-1/2^m)` under M1-mixture), the post-run histogram TVD
(`1-counts[target]/shots`), the signed difference `observed-predicted` in probability units, absolute
differences, and the stored shot-uncertainty interval. Positive differences mean
the model underestimated error. `mixture_prediction_histogram_tvd` separately
compares the complete observed and predicted distributions. Interval inclusion
is descriptive and excludes calibration/model uncertainty.

The simulator baseline report leads with this comparison, includes a worked
example from an actual run, and labels table differences in percentage points
(`100 * (observed - predicted)`). Those differences are not sigma scores. It writes `error_comparison_by_depth.csv` and
`expected_vs_observed_error.png` / `.svg`. These derived artifacts can be
regenerated from existing raw counts without executing any additional circuits.

M1-success and M1-mixture predictions are frozen before execution. The analysis
compares these predictions with saved counts; it does not train additional models.
Prediction uncertainty remains to be calibrated on independent windows before
automatic triage.

New preparations allocate alpha=0.05/(5 × planned executions per window) to the
error interval, mixture check, rank test, central count range, and upper count
budget. Archived preparations retain their original allocation. Inside
general-distribution/model comparisons, Bonferroni
binomial intervals cover all output bins including unobserved ones. This is a
conservative simultaneous allocation. Failed rank tests do
not establish uniformity. Task intervals use an exact deterministic-target
binomial interval, or conservative simultaneous-bin TVD bounds for the small
distribution controls. Large deterministic BV references remain sparse; general
distribution intervals are intentionally limited to 12 output bits.

## Reusable QWARD strategies

The optional `CircuitErrorPrediction` and `HistogramAssessment` strategies are
exported from `qward.metrics`, with Pydantic outputs and flat Scanner exports.
They are experimental and are not included in default scans.

```python
from qward import Scanner
from qward.metrics import CircuitErrorPrediction, HistogramAssessment
from qward.metrics.circuit_error_prediction import capture_calibration

# isa is the exact already-transpiled circuit; backend is its target.
# reference is analytically known; counts come from a completed measurement.
snapshot = capture_calibration(isa, backend)
predictor = CircuitErrorPrediction(isa, snapshot=snapshot, reference=reference)
prediction = predictor.get_metrics()
assessment = HistogramAssessment(
    isa, counts=counts, reference=reference, prediction=prediction,
    alpha=0.05,
)
tables = Scanner(isa, strategies=[predictor, assessment]).calculate_metrics()
```

Save the snapshot and prediction before submission. Use the archived prediction
for post-run analysis. For deliberately injected faults, prediction uses the
injected reference while task assessment uses the nominal BV reference.

Optional idle-only/mirror campaigns, asymmetric assignment-matrix modeling,
Escofet's analytic comparator, and calibrated model-discrepancy intervals remain
follow-up work. No QPU result or predictive hardware validity is claimed by this
implementation. The campaign has been checked with construction/tableau tests,
mocked Runtime jobs, and actual local noisy-simulator shots. Hardware execution
and predictive hardware validation remain pending.
