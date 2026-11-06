# Shor growth campaign evaluation — November 5, 2026

Status: offline evaluation completed. The proposed 39-job workload is now configured
on the Shor growth branch; no IBM jobs submitted. See [local run instructions](shor_growth_local_run.md).
The legacy full profile is preserved; the expanded profile is selected explicitly.

## Objective and measured preflight

Measure changes in output quality against total logical qubits, executed ISA depth,
and two-qubit gate count. Compare slopes within explicitly identified families
(N, base a, construction strategy). An M sweep changes counting precision while
N remains 15; it is not a sweep of increasingly large integers being factored.

Preflight used the current Shor builder, Qiskit 2.6.2, FakeFez, optimization level 3,
transpiler seed 11, ALAP scheduling, and exact noiseless logical references.
Full measurements, including excluded M2 candidates, are in
[shor_growth_preflight.csv](shor_growth_preflight.csv). These depths are offline
estimates, not current live-Fez calibration results or observed hardware performance.

| Family | Counting qubits M | Total qubits | ISA depths across M | 2Q gates across M |
| --- | --- | --- | --- | --- |
| N15, a=4, order 2 | 2,3,4,5,6 | 6,7,8,9,10 | 197,205,238,290,323 | 60,59,74,87,115 |
| N15, a=2, order 4 | 3,4,5,6 | 7,8,9,10 | 498,548,580,641 | 153,164,188,225 |
| N15, a=7, order 4 (existing family) | 3,4,5,6 | 7,8,9,10 | 3955,4039,4042,4065 | 1096,1121,1140,1167 |

The existing a=7 construction synthesizes M7 as a generic permutation unitary.
Its expensive fixed block dominates: ISA depth increases only about 2.8% from
M3 to M6. Higher controlled powers become identity because the order is 4.
The shallower a=2 and a=4 constructions provide more useful relative depth spans.
Gate counts need not increase monotonically after optimization and routing.

M2 has six total qubits for all three bases, but for a=2 and a=7 its ideal
measurement distribution is uniform and correct-order success is 0.5, exactly
its uniform-noise baseline. Counts alone cannot distinguish those ideal results
from uniform noise. Exclude those two M2 cases from the decay regression. For
a=4, M2 has nonuniform ideal output and a 0.25 correct-order noise baseline;
it is a meaningful six-qubit starting point. Smaller N values are handled by the
implementation's classical prechecks; reducing the work register would introduce
a different, compiled/iterative construction and must be identified separately.

## Proposed workload

Use all 13 configurations above, 3 independent jobs each, 1,024 shots per job:

- Tonight, November 5: a distinct Shor-only supplemental identity, 39 jobs.
- Tomorrow, November 6 / third collection: the same 39 Shor jobs replace the
  existing two M3 jobs. The other 342 jobs remain, for 381 total jobs.
- Retain the existing a=7 M3 identity as the link to prior periods. New bases need
  unambiguous configuration IDs; M5 and a=4 M2 also need registry entries.
- Preserve existing Shor gate-twirling and dynamical-decoupling settings for
  comparability, and document them. Do not describe this as an unmitigated run.
- Keep at most 10 jobs per batch and all repetitions of a configuration together.
  With three repetitions per configuration, this is five batches for Shor.
- Give tonight a separate campaign/plan/claim identity so it cannot collide with
  the completed November 5 full collection. Preserve earlier data and budgets.

The smaller alternative is two jobs per configuration: 26 tonight and 368 total
tomorrow, with less information about between-job variability. Three repetitions
are still exploratory rather than a precise estimate of hardware variability.
Actual billed usage and execution time require IBM measurements.

## Metrics and regression interpretation

Primary task success is recovery of the true order through the existing continued-
fraction postprocessing. Its ideal probability is 0.5 for these configurations.
Compute each configuration's uniform-output success baseline with that same
postprocessing; do not assume it is always a power of two (a=4 M6 is 3/64).

Use a chance- and ideal-adjusted signal for cross-width analysis:

    signal = (observed_success - uniform_baseline) / (ideal_success - uniform_baseline)

This diagnostic has 0 at uniform noise and 1 at ideal success. Keep finite-shot
values outside [0,1] for fitting rather than clipping them. Preserve raw success
and full-reference TVD/Hellinger fidelity alongside it. Uniform-output baselines
for distribution metrics also change with width and must be considered.

A falling raw success curve can be produced solely by the changing random
baseline, even without a change in circuit quality. The existing M3 hardware runs
are already close to the 25% baseline, so more a=7 points may describe the noise
floor rather than a resolvable decay slope. Do not force an exponential curve or
log-transform zero/noise-level values. Use binomial shot counts for success,
retain each job, account for collection period, and report slope uncertainty.
Two periods support exploratory comparisons; more periods are needed to assess
stable monthly behavior. Width and depth are correlated within a family, so two
plots do not identify their independent causal effects.

The current Michelson DSR uses only true-order-success labels. Perfect Shor output
also has legitimate peaks that fail that postprocessing criterion; these compete
at equal height and can give zero task-labelled DSR even ideally. Therefore it
cannot be the sole Shor decay measure. A separately labelled full-ideal-support
DSR could measure phase-peak visibility, but answers a different question from
correct-order recovery; preserve both meanings explicitly rather than silently
changing existing DSR columns.

## Implementation checklist (now configured)

1. Add configuration IDs and validate noiseless references/success labels.
2. Add a distinct Shor-only profile/identity for tonight and a versioned expanded
   full profile for tomorrow. Do not modify already-used window-2 template plans.
3. Update bounded-job checks, workflow summaries, and tests for the chosen totals.
4. Use fresh live-backend preparation; reuse only compatible logical metrics.
5. Group dashboard Shor sweeps by N/base/strategy, expose total and counting qubits,
   and describe the selected order-finding task, ideal ceiling, and noise baseline.
6. Export using the new multi-file dataset format after retrieval.

Background: [IBM Shor/order-finding tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm).
