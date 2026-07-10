# Broad-Ideal, Known-`E` Experiment: Results

Addresses Reviewer 1's request for a case where the full ideal histogram is
genuinely unavailable, so the paper's "histogram-free" framing is tested
against a scenario where it is actually needed rather than merely convenient.

Script: `broad_ideal_experiment.py`. Data: `broad_ideal_experiment_results.json`.
Figure: `plots/3_broad_ideal_scaling.png`.

## Setup

Multi-marked Grover search (`K = 3` marked states, analytically known),
increasing qubit count `n`, `shots = 4096`, optimal iteration count per
Grover's formula.

**Stage 1 (n = 6, 8, 10, 12, 14, real circuits).** Both the legacy
full-distribution path (build the exact ideal statevector over all `2**n`
outcomes via `Statevector.from_instruction`, then compute Hellinger
fidelity/TVD against the observed counts) and the new histogram-free
`DSRProfiler` path are run on the **same** real, locally-simulated Grover
circuit and counts. This is a correctness cross-check as well as a timing
comparison — the coarse profile should track the full-distribution result
sensibly at every `n`.

**Stage 2 (n = 26, 28, 30, 32, 36, 40, synthetic counts).** No circuit is
simulated at all — a real Grover circuit at these sizes is outside the reach
of any classical machine (a bare `2**32` complex128 statevector is ~64 GB;
`2**40` is ~17 TB), which is exactly the "no ideal histogram available"
regime this experiment targets. Counts are generated directly as a
multinomial sample over the `K` marked states plus a small set of random
"other" bitstrings, standing in for what real QPU output would look like at
that scale. `DSRProfiler` is run on these counts using only the known
marked-state set `E` — no `2**n`-sized object is ever constructed.

## Results

### Timing (Stage 1, measured on both paths)

| n | full-distribution HF/TVD | DSR profile | speedup |
|---|---|---|---|
| 6 | 108 ms | 4.2 ms | 26x |
| 8 | 556 ms | 0.10 ms | 5,673x |
| 10 | 2,379 ms | 0.25 ms | 9,636x |
| 12 | 9,636 ms | 2.2 ms | 4,306x |
| 14 | 62,915 ms (~63 s) | 0.28 ms | 222,054x |

Full-distribution cost grows exponentially with `n` (~4-7x per +2 qubits, as
expected for `O(2**n)` statevector construction and dict comparison); the
profile path's cost is dominated by Python/dict overhead on `O(shots + K)`
inputs and does not exhibit any comparable growth. Extrapolating the
observed trend, `n = 20` would already take on the order of hours locally,
and `n = 32` is not just slow but architecturally infeasible (see Stage 2
`theoretical_full_distribution_bytes` column: 34 GB at `n=32`, 8.8 TB at
`n=40`, for the probability array alone, before any statevector simulation
cost).

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

Profile computation time stays flat (sub-millisecond) across a 15-qubit
range where the corresponding full-distribution computation is not merely
slower but requires more memory than exists on essentially any single
machine. `success_rate == coarse_tvd_similarity == coarse_hellinger_fidelity`
here is the expected, exact `K=1`-style behavior generalized to this
synthetic-counts setup (uniform default weights over `E`, no `other`-side
structure to distinguish TVD from Hellinger at this level of precision) and
is a sanity check on the synthetic generator, not a new empirical claim.

## Manuscript framing

Use this experiment as the concrete answer to "when would you actually need
a histogram-free method?" — for algorithms whose target space grows past
`~20`–`25` qubits (well within reach of near-term devices), computing or
even representing the full ideal distribution is not a matter of "faster if
you skip it," it is off the table entirely, while the profile is computed
from the same three inputs (counts, `E`, `m`) it always uses, with per-job
cost independent of `n`. Cite the measured Stage 1 exponential trend as
direct evidence, and the Stage 2 theoretical-byte-count column as the
argument for why Stage 1's trend cannot be pushed further locally.
