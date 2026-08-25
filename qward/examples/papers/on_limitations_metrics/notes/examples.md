# Example selection

Include a job or figure only if it demonstrates Test A, Test B,
Test C, a hand-worked formula, or a stated DSR failure mode.
Do not dump the full Grover / QFT / BV / teleportation corpus.

Primary table: `qward/examples/papers/DSR_result.csv`.
Narrative numbers: `qward/examples/papers/narrative_assessment.md`.
Broad-ideal: `qward/examples/papers/broad_ideal_experiment_results.md`.
BV wall: `qward/examples/papers/bv/bv_scaling_utils.py`.

## Worked calculation (must include)

Use the library's own sanity-check histogram so a reader can
reproduce it from `DSRProfiler`:

```
C = {"01": 40, "00": 20, "10": 20, "11": 20}
E = {"01"}
S = 100,  m = 2,  K = 1,  b = 1/4
```

| Quantity | Value | Role |
|---|---|---|
| \(p_E\) | \(40/100=0.40\) | Score |
| CCS | \(\mathrm{clip}((0.40-0.25)/0.75)=0.20\) | Clipped score |
| Coarse observed / ideal | \((0.40,0.60)\) vs \((1,0)\) | Push-forward |
| Coarse TVD | \(0.60\) | Metric on \(\Delta_2\) |
| Coarse TVD similarity | \(0.40=p_E\) | \(K=1\) collapse |
| \(\mathrm{BC}\) | \(\sqrt{0.40}\) | |
| Coarse HF | \(0.40=p_E\) | \(K=1\) collapse |
| Coarse HD | \(\sqrt{1-\sqrt{0.40}}\) | Metric on \(\Delta_2\) |
| \(\bar p_E\), \(p_{\mathrm{comp}}\) | \(0.40\), \(0.20\) | |
| Michelson DSR | \((0.20)/(0.60)=1/3\) | Contrast score |

Threshold conversion (Class TV, same histogram against the coarse
ideal): "pass if coarse TVD \(\le 0.30\)" is the same decision as
"pass if coarse TVD similarity \(\ge 0.70\)". This is the negative
result from Section 2, shown by hand.

If you later recompute this block, compare against `DSRProfiler` in
`differential_success_rate.py` (the `__main__` example).

## Test A — invariance (single-job monotone families)

Need rows where full-histogram `hellinger_distance`,
`hellinger_fidelity`, `tvd`, and `tvd_fidelity` are all present.

`DSR_result.csv` already stores all four. Any Grover or QFT job
below the simulation wall works.

Show one IBM Grover row and one QFT round-trip row:

- rank by \(H\) equals reverse rank by \(F_H\);
- rank by TVD equals reverse rank by \(1-\mathrm{TVD}\);
- Spearman \(\rho=-1\) inside each pair if there are no ties.

Do **not** expect coarse HF to match full HF on finite-iteration
Grover (`narrative_assessment.md` §3, residual amplitude). That
disagreement is a different attribute, not a Test A failure.

Candidate figure already on disk:
`qward/examples/papers/plots/` `2_full_vs_coarse_comparison.png`
(mentioned in the narrative). Use it only to explain the Grover
ideal-is-not-a-delta point.

## Test B — aggregation sensitivity

This is the only empirical claim that can change the paper's size.
Protocol: [consequence-gate.md](consequence-gate.md).

Until that analysis is run, **do not** write that IBM versus
Rigetti rankings flip under mean(\(F_H\)) versus mean(\(H\)).

What we already know, and must not confuse with Test B:

- Grover / QFT: IBM median success > Rigetti at every matched
  qubit count (`narrative_assessment.md` §1). That is a median of
  *scores*, not a mean-of-distance versus mean-of-similarity test.
- Teleportation payloads 3–4: IBM median success < Rigetti, while
  Michelson is ~0 on both sides (`narrative_assessment.md` §2).
  That is a **score-versus-score** disagreement (CCS vs Michelson),
  excellent for the DSR case study, not for Class H aggregation.

If Test B finds no inversion, keep teleportation as the case-study
example of "which *score*" changes the conclusion, and treat
Class H aggregation as a theoretical warning with a negative
empirical result.

## Test C — information feasibility

Use these, not more:

- Broad-ideal Stage 1: \(n=6\ldots 14\), both full HF/TVD and the
  DSR profile, timing table in
  `broad_ideal_experiment_results.md` (14 qubits: ~63 s vs 0.28 ms).
- Broad-ideal Stage 2: \(n=26\ldots 40\), profile only; full
  distribution memory from 537 MB to 8.8 TB.
- BV scaling note in `bv/bv_scaling_utils.py`: HF needs the
  \(2^n\) ideal; DSR needs `secret[::-1]`.

One timing table plus one sentence that \(H\) and TVD are the
correct distances *when they exist* is enough. Do not re-argue
"histogram-free is always better."

## DSR failure-mode example (Section 9)

Teleportation payloads 3–4, IBM versus Rigetti
(`narrative_assessment.md` §2):

| Payload | IBM median \(p_E\) | Rigetti median \(p_E\) | Michelson |
|---|---|---|---|
| 3 | 0.132 | 0.400 | ~0 both |
| 4 | 0.066 | 0.200 | ~0 both |

This shows clipping and T1 bias, not that Michelson is "more
sensitive" in a good way.

## Figures to reuse if the corresponding test is kept

- Combined provider comparison:
  `qward/examples/papers/plots/1_combined_dsr_comparison.png`
  and `1_combined_dsr_comparison_aws.png` — only if they illustrate
  Test A or the teleportation failure mode.
- Broad-ideal:
  `qward/examples/papers/plots/3_broad_ideal_scaling.png` — Test C.
- Do not include optimization-level or depth boxplots unless they
  demonstrate one of the three tests.

## Estimator illustration (Section 5 / 8)

Not a replacement for Tests A–C. One small numeric example is
enough to show the other interpretation of the same shots:

- Hand-worked Pauli: ideal \(1\), observed \(0.70\), std \(0.04\)
  ([notes/output-types.md](output-types.md)).
- Optional hardware pointer, if kept: GHZ-4 with six observables in
  `qward/examples/estimator_ibm_experiment.py`. Use it only to show
  that aggregating six observable fidelities is a different decision
  from aggregating six absolute deviations.

Do not compute DSR or Hellinger fidelity on Estimator `evs`.

## Explicitly out of scope for this paper

- New QPU jobs.
- Variational / estimator-path fidelity.
- Re-litigating the QCE26 "DSR is sharper than HF" narrative.
- Averaging all profile components into one number.
