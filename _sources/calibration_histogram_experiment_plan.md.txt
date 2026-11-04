# BV calibration prediction and histogram uncertainty: active experiment plan

Status: experimental implementation with a completed local simulator baseline
and two small ibm_fez pilots (10 jobs; findings in CALIBRATION_EXPERIMENT.md).
A three-window drift campaign (bv-ibm-fez-3) is planned.

This plan contains the current experiment and the
validation required to use its results; speculative model-fitting and additional
circuit campaigns have been removed.

## 1. Objective and scope

Use QWARD's existing Bernstein–Vazirani implementation to test whether frozen
backend calibration and the actual transpiled circuit predict histogram error
well enough to guide review. The ideal BV answer is known analytically, so no
statevector or density-matrix simulation is needed for the reference.

Keep two explicit prediction hypotheses: the calibration product as success
(M1-success), and that same product as uniform-mixture visibility (M1-mixture).
They are hypotheses to evaluate, not proven fidelity estimates or hardware error
bounds. A third, parameter-free hypothesis (M2-depolarizing) converts average
infidelity to depolarizing probability and keeps the 1/D floor. No fitted
predictors, idle/mirror campaign, or probability reconstruction is part of this
experiment.

The existing implementation uses the library's BV builder and experiment result
conventions. See [usage and recovery](../qward/examples/papers/bv/CALIBRATION_EXPERIMENT.md)
and the [baseline report](../qward/examples/papers/bv/data/simulator/calibration_prediction/baseline_20261022_smoke_v2/analysis/baseline_report.md).

## 2. The quantities being compared

Let m be the number of measured output bits, D=2^m, N the actual number of shots,
and t the known BV answer in Qiskit's displayed classical-bit order.

- Observed histogram error: `E_observed = 1 - counts[t]/N`. For deterministic BV
  this equals total variation distance from the ideal histogram exactly.
- Calibration product: `Q = product(1 - gate_error) * product(1 - readout_error)`,
  using each executed native instruction and measurement once. Compute in log
  space, record calibration coverage, and return an unavailable complete
  prediction if required errors are missing.
- Product prediction: `E_product = 1 - Q`.
- Depolarizing prediction (M2): `λ = product(1 - r*d/(d-1))` over native gates,
  `E_M2 = 1 - [λ * product(1 - readout_error) + (1 - λ)/D]`. Parameter-free,
  bounded by `1 - 1/D`.
- Uniform-mixture prediction: `E_mixture = (1 - Q) * (1 - 1/D)`.
- Difference: `E_observed - E_predicted`. Tables multiply this by 100 and label it
  **Observed − predicted (percentage points)**. Positive means underestimated
  error; negative means overestimated error. This is not a sigma score.

For STRESS, L=8 in the current smoke baseline, observed error is 52.56%, product
prediction is 59.71%, and mixture prediction is 44.78%. Thus the product difference
is −7.15 percentage points and the mixture difference is +7.78 percentage points.
These are averages over three secrets and three seeds, not one run's uncertainty.

Gate calibration errors are average instruction errors, not automatically
independent failure probabilities. A scalar product does not specify which wrong
bitstrings receive probability. The full uniform-mixture histogram comparison is
retained to test that additional assumption.

## 3. Uncertainty and error budgets

For each deterministic-target result, calculate an exact Clopper–Pearson interval
for the true wrong-answer probability. Also report the estimated shot standard
error `sqrt(E_observed * (1-E_observed) / N)`. At empirical 0 or 1, this plug-in
standard error is zero but the exact interval still has width.

Conditional on a fixed, independently supplied predicted error probability e,
the wrong-answer count has the model `K ~ Binomial(N, e)`. Compute:

1. An exact central count range, allocating alpha/2 to each tail.
2. A one-sided upper count budget, allocating alpha to the upper tail.
3. Exact lower and upper tail probabilities for the observed count.
4. The descriptive signed score
   `Z = (E_observed - e) / sqrt(e * (1-e) / N)`.

This Z uses the shot variance under the prediction. It is undefined for e=0 or 1;
return null and use the exact test. These conditional budgets include shot
variation only. They do not include calibration uncertainty, model discrepancy,
or drift, and assume independent stationary shots.

`HistogramAssessment.error_budget` provides this check when the task and
prediction have the same deterministic reference. The reusable
`binomial_prediction_check` also accepts a justified noise-only event probability:
for example, 2 counts in 1,000 shots is compatible with probability 0.002, whose
95% central count range is 0–5. Two occurrences alone cannot establish noise.

For primary BV only, the exporter separately tests whether the known target
exceeds a uniform-background probability of 1/D. This is an explicit diagnostic
hypothesis. It is not a calibrated per-bin hardware noise model, and does not
apply to intentionally uniform controls. Selected-winner evidence uses QWARD's
existing exact rank test; a statistically resolved wrong answer is still wrong.

For a fixed archive of R primary runs, the offline analysis uses alpha=0.05/(5R)
for two central ranges, two upper budgets, and one target-background claim per
run. These retrospective baseline checks are exploratory and separate from the
historical raw-record tests; they are not a sequential stopping rule. The generic
metric's caller is responsible for allocating alpha across its intended claims.

## 4. Circuit catalogue and execution budget

Use existing ONES, ALT and SINGLE secret patterns. Read the expected output from
`bv.expected_outcomes[0]`; secret strings and displayed counts use opposite index
orders. BV uses m data qubits and one unmeasured ancilla.

| Family | Configurations per hardware window | Purpose |
|---|---:|---|
| Primary BV | 48: widths 2/4 × 3 secrets × L=0/2/8/32 × 2 layouts | Preserve ideal answer while increasing native noise exposure |
| SPAM | 16 | Check preparation/readout behavior across masks and layouts |
| GHZ Z/X | 16 | Check multi-answer references and phase faults invisible in one basis |
| Uniform | 4 | Ensure intended uniform output is not treated as algorithm failure |
| BV RY faults | 16 | Check known target deviations, including a sharply peaked wrong answer |

L is the number of forward padding layers U_L, followed by its inverse. Insert
barrier-protected padding after the BV oracle and before the final data Hadamards.
Use seeded H/S gates and alternating nearest-neighbor CX layers, seed 11, with
shared prefixes across depths. L=0 is the unchanged library circuit. Verify that
the padding preserves the ideal Clifford circuit and survives transpilation.

Core hardware design: 100 configurations × 2,048 shots × 3 calibration windows =
300 executions and 614,400 shots. The width-2, one-layout smoke subset has 25
configurations and 51,200 shots per window. No hardware jobs have been submitted.
The catalogue and budgets are recorded in [pilot specification](calibration_histogram_pilot.json).

## 5. Existing simulator baseline

Completed: 25 smoke configurations × IDEAL/NOMINAL/STRESS × 3 independent shot
seeds × 2,048 shots = 225 executions and 460,800 shots. Preserve all raw counts,
frozen predictions, calibration snapshots, QPY artifacts and noise-model hashes.

The synthetic backend has local depolarizing gate noise and symmetric readout
noise. Aer converts average gate infidelity into the corresponding local channel.
Thermal/idle channels are disabled. Clifford circuits use stabilizer simulation;
only the pi/6 fault control uses matrix product states. Repeated seeds share a
stationary calibration and are not hardware drift windows. Reproducibility reruns
are not counted as independent experiments.

Regenerate analysis from these saved counts. Keep one error comparison table,
one worked run, error-versus-prediction and count-budget figures, control checks,
and provenance. Do not duplicate error results as another success plot/table.

## 6. Tracking and review workflow

1. Build the existing BV circuit and selected treatment; freeze the physical layout,
   transpiler seed, scheduling and runtime options.
2. Capture calibration for the exact ISA circuit; save native counts, durations,
   calibration coverage and timestamps. These features explain noise exposure;
   no learned predictor consumes them in this pilot.
3. Save the prediction, logical/ISA QPY files, hashes and analytic references before
   execution. Submit only that ISA circuit. Persist job IDs before collecting.
4. Store results in the existing rich experiment schema. Recover jobs without
   recomputing predictions or resubmitting an unknown job.
5. Report observed error and its interval, conditional error budgets, target signal,
   and rank/model evidence separately. Keep controls separate from primary BV rows.

Unexpected error in either direction requests prediction review. Unresolved target
signal is inconclusive at the current shot budget. Passing an upper budget alone
can reflect an overly pessimistic model and is not proof of useful information.
All histograms remain in the research dataset. The earlier arbitrary 80% success
and 0.10 TVD acceptance rules have been removed from current metrics and exports;
historical raw records retain their original fields only for provenance.

## 7. Validation still required

Before automatic hardware triage, quantify prediction discrepancy on independent
calibration windows, freeze a predictive bound, and test its coverage and false
dismissals on untouched windows. Keep one hardware window reserved for validation;
three windows are a pilot, not a long-term drift characterization. Do not treat
an interval fitted after seeing a result as a prospective prediction.

Combined uncertainty units require a justified prediction variance. For independent
errors, use `sqrt(sigma_shot^2 + sigma_prediction^2)`; dependent estimates need
covariance treatment. The current implementation intentionally reports shot-only
scores until prediction uncertainty has been calibrated. A per-bin noise decision
also requires a validated background probability for that bin.

Required checks cover reference/bit-order correctness, identity padding, native
calibration accounting, exact count-tail coverage and boundaries, same-reference
comparison, immutable artifacts, job recovery, and Scanner/schema integration.

## 8. Literature used for context

| Reference | Relevance and experimental role |
|---|---|
| [Escofet et al., *An Accurate Efficient Analytic Model of Fidelity under Depolarizing Noise oriented to Large Scale Quantum System Design* (2025), arXiv:2503.06693](https://arxiv.org/html/2503.06693v1) | Closest match: calibration-based analytic prediction with per-qubit updates. Algorithm 1 introduces an entanglement parameter; this pilot cites the work but does not implement that comparator. Its reported comparisons include ESP, QVA, and hardware experiments. Treat its assumptions and reported fidelity ranges as model-specific, not universal hardware guarantees. |
| [Murali et al., *Noise-Adaptive Compiler Mappings for Noisy Intermediate-Scale Quantum Computers* (ASPLOS 2019)](https://arxiv.org/abs/1901.11054) | Establishes calibration-aware mapping as prior art. Motivates comparing physical layouts and preserving the calibration snapshot used to choose them. |
| [Proctor et al., *Measuring the Capabilities of Quantum Computers* (Nature Physics 2022)](https://arxiv.org/abs/2008.11294) | Scalable, verifiable circuit benchmarks and evidence that structured circuits can behave differently from random benchmarks. Background for the limitations of structured benchmarks; no mirror campaign is included here. |
| [McKay et al., *Benchmarking Quantum Processor Performance at Scale* (2023 preprint)](https://arxiv.org/abs/2311.05933) | Layer fidelity uses simultaneous benchmarking and is sensitive to crosstalk. Explains why isolated gate calibration may omit relevant error sources. |
| [Magesan et al., *Efficient measurement of quantum gate error by interleaved randomized benchmarking* (PRL 2012)](https://arxiv.org/abs/1203.4550) | Background for interpreting average gate-error estimates and their assumptions. An average error is not a literal independent Bernoulli failure probability for each instruction. |
| [Hung and Fithian, *Rank Verification for Exponential Families* (Annals of Statistics 2019)](https://arxiv.org/abs/1610.03944) | Selected-winner inference. QWARD already implements the conservative exact top-two multinomial rank test described by this work. Use it instead of interpreting a selected peak's Wald score as a correctness certificate. |

These papers motivate the experiment and its limitations; they do not establish
that the pilot’s calibration product is a valid hardware error bound.
