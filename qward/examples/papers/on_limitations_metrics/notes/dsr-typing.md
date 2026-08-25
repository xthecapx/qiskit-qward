# DSR profile typing

Formal typing of every component produced by
`qward.metrics.differential_success_rate` and
`DSRProfileSchema`. Lead with weaknesses. Implementation references
are to `qward/metrics/differential_success_rate.py`.

The profile is **not a metric**. It is a known-answer evaluation
record that mixes unary scores with two genuine distances on a
coarse simplex.

## Inputs

- Observed counts \(\mathcal{C}: \{0,1\}^m \to \mathbb{Z}_{\ge 0}\),
  \(S=\sum_x \mathcal{C}(x)>0\).
- Expected set \(E\), \(K=|E|\ge 1\).
- Optional task-reference weights \(w\) on \(E\) (default uniform
  \(1/K\)).
- Optional Michelson layer (`include_michelson=True` by default).

No object of size \(2^m\) is constructed. That is the information
advantage and the applicability restriction: \(E\) must be known.

## Component typing

### 1. `success_rate` — score

\[
p_E=\frac{1}{S}\sum_{x\in E}\mathcal{C}(x)
\]

- Kind: unary score, not a metric.
- Attribute: mass on \(E\).
- Representation: preserves "more shots in \(E\)" among jobs that
  share the same \(E\).
- Scale: derived proportion of absolute counts.
- Weakness: silent about competing peaks and about chance \(K/2^m\).

### 2. `chance_baseline` — derived constant

\[
b=K/2^m
\]

Not a success quantity. It is the random-guessing reference used by
chance-corrected success.

### 3. `chance_corrected_success` — clipped score

\[
\mathrm{CCS}=\mathrm{clip}\Bigl(\frac{p_E-b}{1-b},0,1\Bigr)
\]

(`compute_chance_corrected_success`, clip at `_clip_zero_one`)

- Kind: unary score, not a metric.
- Attribute intended: above-chance success.
- Representation defect: every job with \(p_E\le b\) maps to 0, so
  the ordering "worse than chance" is destroyed. This is a Fenton
  representation failure on the below-chance region, not just a
  numerical convenience.
- Degeneracy: if \(K\ge 2^m\), the function returns 1.
- At \(K=1\), CCS is a clipped affine transform of \(p_E\). It does
  not add an independent degree of freedom.

### 4. `coarse_tvd` — metric on \(\Delta_{K+1}\)

Push-forward: one bin per element of \(E\), plus `__other__`.
Ideal: \(w\) on \(E\) and 0 on other. Observed: empirical masses.

\[
\mathrm{coarse\_tvd}=\tfrac12\sum_k |p_k-q_k|
\]

- Kind: **distance metric** on the coarse simplex (TVD restricted
  to the push-forward).
- Inherits M1–M4 from TVD on that simplex.
- At \(K=1\), \(\mathrm{coarse\_tvd}=1-p_E\) exactly.

### 5. `coarse_tvd_similarity` — score

\[
1-\mathrm{coarse\_tvd}
\]

- Kind: similarity / score, not a metric.
- Per-job threshold-equivalent to coarse TVD.
- At \(K=1\), equals \(p_E\).

### 6. `coarse_hellinger_distance` — metric on \(\Delta_{K+1}\)

Code: \(\sqrt{\max(0,1-\mathrm{BC})}\).

- Kind: **distance metric** on the coarse simplex.
- Note the convention: Qiskit's `hellinger_distance` uses the same
  family; Qiskit `hellinger_fidelity` is \((1-H^2)^2=\mathrm{BC}^2\),
  which matches `coarse_hellinger_fidelity` below, not \(1-H\).

### 7. `coarse_hellinger_fidelity` — score

Code: \(\min(1,\mathrm{BC}^2)\).

- Kind: similarity / score, not a metric.
- At \(K=1\), \(\mathrm{BC}=\sqrt{p_E}\), so \(\mathrm{BC}^2=p_E\).
- Therefore at \(K=1\) this is not independent evidence from
  `success_rate` or `coarse_tvd_similarity`.

### 8. `dsr_michelson` — clipped score (optional fifth layer)

\[
\bar p_E=p_E/K,\qquad
p_{\mathrm{comp}}=\max_{x\notin E}p_x,\qquad
\mathrm{DSR}=\mathrm{clip}\Bigl(\frac{\bar p_E-p_{\mathrm{comp}}}{\bar p_E+p_{\mathrm{comp}}},0,1\Bigr)
\]

- Kind: unary contrast score, not a metric.
- Attribute: dominance of the *mean* expected peak over the
  *strongest* competitor.
- Representation defects:
  - clipping at 0 destroys below-contrast order;
  - dividing by \(K\) makes the score decrease with \(K\) even when
    \(p_E\) is unchanged;
  - comparing a mean to a max is an asymmetric empirical relation
    that must be stated, not hidden.
- Hardware defect: T1 / amplitude damping concentrates mass on
  all-zero. If the competitor is that attractor and the target is
  zero-heavy, \(p_{\mathrm{comp}}\) can match \(\bar p_E\) and the
  score collapses while \(p_E\) still separates providers
  (teleportation payloads 2–4 in `narrative_assessment.md`).
- `peak_mismatch` is a flag, not a metric: true when no element of
  \(E\) is among the modal outcomes.

### 9. Unused contrast variants (not in the profile schema)

`compute_dsr_ratio`, `compute_dsr_log_ratio`,
`compute_dsr_normalized_margin` are alternative scores on the same
\((\bar p_E,p_{\mathrm{comp}})\) pair. They are not additional
metrics. Do not present them as independent confirmation.

## \(K=1\) degeneracy

When \(K=1\), the non-Michelson profile has **one** underlying
degree of freedom, \(p_E\):

| Component | Value at \(K=1\) |
|---|---|
| `success_rate` | \(p_E\) |
| `coarse_tvd` | \(1-p_E\) |
| `coarse_tvd_similarity` | \(p_E\) |
| `coarse_hellinger_fidelity` | \(p_E\) |
| `coarse_hellinger_distance` | \(\sqrt{1-\sqrt{p_E}}\) |
| `chance_corrected_success` | \(\mathrm{clip}((p_E-b)/(1-b),0,1)\) |
| `dsr_michelson` | contrast of \(p_E\) against \(\max_{x\neq x^\star}p_x\) |

This collapse is exact for the coarse construction. It is **not**
identity with full-histogram HF unless the true ideal is a delta on
\(E\). QFT round-trip and teleportation are deltas; finite-iteration
Grover is not (`narrative_assessment.md` §3).

## What the profile is, formally

A **typed record**

\[
\bigl(p_E,\;\mathrm{CCS},\;
d_{\mathrm{TV}}^{\mathrm{coarse}},\;
d_{H}^{\mathrm{coarse}},\;
\text{optional Michelson}\bigr)
\]

together with the derived similarities \(1-d_{\mathrm{TV}}^{\mathrm{coarse}}\)
and \(\mathrm{BC}^2\).

- Two entries are distance metrics, but only on \(\Delta_{K+1}\).
- The headline numbers used in prose (success, CCS, coarse
  similarities, Michelson) are scores.
- Averaging the four headline components into one "DSR" would
  invent a fifth derived measure with no attribute (Fenton 2.4.4).
  The schema already forbids that.

## Defensible scope

Use the profile when:

- the task has an analytically known answer set \(E\);
- the decision is task-level ("did we get the answer?") rather
  than state-level ("did we prepare \(P^\star\)?") or
  observable-level ("did \(\langle O\rangle\) match?");
- the primitive is Sampler (or counts can be reconstructed);
- \(P^\star\) may be incomputable.

Do not use it when:

- the job is Estimator-only (`evs` / `stds` without a histogram);
- the ideal is a broad variational landscape with no discrete \(E\);
- phase errors matter;
- the target is a zero-heavy singleton under strong T1 and the
  chosen layer is Michelson;
- \(K=1\) and the reader is invited to treat four similar numbers
  as independent evidence.

## How the case-study section should be written

1. Weaknesses first (degeneracy, clipping, mean-vs-max, \(K\),
   known \(E\), phase, T1, VQA).
2. Then the two coarse metrics and the information requirement.
3. Then the GQM placement: DSR answers known-answer questions; it
   does not replace TVD / \(H\) on \(\Delta_{2^m}\) or DM fidelity.
4. Do not argue that DSR "outperforms" HF. Argue that it is a
   different attribute with stated failure modes.
