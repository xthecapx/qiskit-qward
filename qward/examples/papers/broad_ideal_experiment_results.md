# Broad-Ideal, Known-`E` Experiment: Results

This experiment compares a compact known-answer computation with one dense
implementation of a full ideal distribution comparison. It supports a claim
about representation availability, not a universal simulation bound.

Script: `broad_ideal_experiment.py`. Data: `broad_ideal_experiment_results.json`.
Figure: `plots/3_broad_ideal_scaling.png`.

## Setup

Multi-marked Grover search (`K = 3` marked states, analytically known),
increasing qubit count `n`, `shots = 4096`, optimal iteration count per
Grover's formula.

**Stage 1 (n = 6, 8, 10, 12, 14, real circuits).** Both the
full-distribution path (build the exact ideal statevector over all `2**n`
outcomes via `Statevector.from_instruction`, then compute Hellinger distance
and fidelity against the observed counts) and the compact `DSRProfiler` path
are run on the **same** locally simulated Grover
circuit and counts. This is a correctness cross-check as well as a timing
comparison — the coarse profile should track the full-distribution result
sensibly at every `n`.

**Stage 2 (n = 26, 28, 30, 32, 36, 40, synthetic counts).** No circuit is
simulated. Counts are generated directly as a
multinomial sample over the `K` marked states plus a small set of random
"other" bitstrings. They test only whether `DSRProfiler` accepts compact
records at these widths; they do not represent hardware accuracy.
`DSRProfiler` is run on these counts using only the known
marked-state set `E` — no `2**n`-sized object is ever constructed.

## Results

### Timing (Stage 1, measured on both paths)

| n | full-distribution H/HF | DSR profile | speedup |
|---|---|---|---|
| 6 | 108 ms | 4.2 ms | 26x |
| 8 | 556 ms | 0.10 ms | 5,673x |
| 10 | 2,379 ms | 0.25 ms | 9,636x |
| 12 | 9,636 ms | 2.2 ms | 4,306x |
| 14 | 62,915 ms (~63 s) | 0.28 ms | 222,054x |

The measured dense path grows rapidly with `n`, as expected from the
statevector construction and distribution comparison used here. The compact
path does not exhibit comparable growth in these runs. The Stage 2
`theoretical_full_distribution_bytes` column gives the storage for one dense
float64 probability array: 34 GB at `n=32` and 8.8 TB at `n=40`. These values
do not include the statevector or prove that every possible R2 algorithm has
the same cost; structured ideals may admit sparse or analytic computation.

### Correctness cross-check (Stage 1)

At every `n`, `coarse_hellinger_fidelity` tracks `full_hellinger_fidelity`
closely (e.g. `n=14`: coarse `0.999957` vs. full `0.999801`), and both
increase with `n` as the optimal Grover iteration count more precisely
amplifies the marked-state amplitude. `coarse_tvd_similarity` is
systematically a little lower than the full-distribution Hellinger fidelity
at fixed `n` (e.g. `n=6`: `0.9882` vs. full HF `0.9975`) — consistent with
TVD being a stricter (L1-type) measure than Hellinger fidelity, not a
methodological problem. Nothing here contradicts the `K=1` scope note in
`narrative_assessment.md`; that note is specifically about the *coarse
metric vs. raw success rate* collapse at `K=1`, not about coarse vs. full
agreement at `K>1`, which is the case exercised here.

### Beyond the wall (Stage 2, profile only)

| n | success_rate | chance_corrected_success | coarse_tvd_similarity | coarse_hellinger_fidelity | profile time | theoretical full-distribution bytes |
|---|---|---|---|---|---|---|
| 26 | 0.377 | 0.377 | 0.377 | 0.377 | 0.6 ms | 537 MB |
| 28 | 0.337 | 0.337 | 0.337 | 0.337 | 0.04 ms | 2.1 GB |
| 30 | 0.300 | 0.300 | 0.300 | 0.300 | 0.03 ms | 8.6 GB |
| 32 | 0.264 | 0.264 | 0.264 | 0.264 | 0.03 ms | 34 GB |
| 36 | 0.176 | 0.176 | 0.176 | 0.176 | 0.03 ms | 550 GB |
| 40 | 0.094 | 0.094 | 0.094 | 0.094 | 0.02 ms | 8.8 TB |

Profile computation time stays below one millisecond in these measurements.
The displayed values of `success_rate`, `coarse_tvd_similarity`, and
`coarse_hellinger_fidelity` are equal only to the printed precision. With
`K = 3`, equality of coarse Hellinger fidelity with success rate requires the
observed marked mass to follow the uniform task weights. The generator makes
that approximately true through a multinomial allocation; it is not a
general identity for multiple accepted answers.

## Manuscript framing

Use this experiment only for the narrower statement supported by its design:
the known-answer path consumes compact inputs at widths where this dense R2
implementation was not run. Cite Stage 1 as timing for one implementation and
Stage 2 as an input availability demonstration. Do not present the synthetic
records as executed circuits or claim that every full distribution method is
infeasible at those widths.
