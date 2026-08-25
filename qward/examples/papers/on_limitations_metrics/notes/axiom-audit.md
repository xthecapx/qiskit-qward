# Axiom and measurement-theory audit

First-pass table for paper Section 6. Status: cite or prove before
drafting; do not invent missing proofs.

Notation:

- \(X\) is a named space. A **distance metric** is
  \(d: X\times X\to\mathbb{R}\) satisfying M1–M4.
- \(\Delta_{2^m}\) is the full outcome simplex.
- \(\Delta_{K+1}\) is the coarse simplex \(\{E\}\cup\{\mathrm{other}\}\).
- A **similarity** is typically 1 at equality and decreasing in
  distance. It is not a candidate for M1–M4.
- A **score** is a unary function of one histogram plus a goal or
  model. M1–M4 do not apply.

## Metric-space axioms

On a set \(X\), \(d\) is a metric iff:

- **M1** \(d(x,y)\ge 0\)
- **M2** \(d(x,y)=0\iff x=y\)
- **M3** \(d(x,y)=d(y,x)\)
- **M4** \(d(x,z)\le d(x,y)+d(y,z)\)

Nearby objects (paper Section 3):

- Pseudometric: M2 relaxed to \(d(x,x)=0\).
- Quasimetric: M3 dropped.
- Divergence (KL): not symmetric; M4 fails.
- Similarity / fidelity: higher is better; equality maps to 1.
- Score: unary; not a function on \(X\times X\).

Correct language for fidelities: they are **not metrics** because they
are not distance functions (equality maps to 1, not 0). Do not write
"Hellinger fidelity fails the triangle inequality." If we audit
infidelity \(1-F_H\), give a counter-example against that
dissimilarity.

## Monotone-equivalence classes

For one job and one fixed reference \(Q\):

- Class H: \(H(P,Q)\), \(F_H(P,Q)=\mathrm{BC}(P,Q)^2=(1-H^2)^2\),
  and \(\theta=\arccos\mathrm{BC}\) are strictly monotone transforms
  of \(\mathrm{BC}(P,Q)\). Any threshold on one converts to a unique
  threshold on the others. Single-job pass/fail is invariant.
- Class TV: \(\mathrm{TVD}(P,Q)\) and \(1-\mathrm{TVD}(P,Q)\) are
  exactly complementary. Single-job pass/fail is invariant.
- Class E, \(K=1\): \(p_E\), coarse TVD similarity, and coarse
  Hellinger fidelity are identical. Chance-corrected success is a
  clipped affine transform of \(p_E\).

Aggregation by arithmetic mean is **not** invariant inside Class H.
Median aggregation should preserve group order inside a strictly
monotone class.

## Compact audit

| Quantity | Claimed attribute | Domain / arity | Reference | M1–M4 | Representation | Scale (tentative) | Meaningful operations | Monotone class | Principal limitation |
|---|---|---|---|---|---|---|---|---|---|
| Success rate \(p_E\) | Mass on the goal set | Unary: job \(+\,E\) | Known \(E\) | N/A (not a distance) | Preserves "more mass on \(E\)" | Derived proportion | Thresholds after stating chance baseline; means OK as frequencies | Class E at \(K=1\) | Ignores competing peaks; not injective on histograms |
| Chance-corrected success | Above-chance mass on \(E\) | Unary: job \(+\,E+m\) | Known \(E\), \(m\) | N/A | Broken below chance by clipping to 0 | Clipped derived | Order above chance; no below-chance ranking | Affine of Class E, then clipped | Censors \(p_E<b\); \(K=2^m\) forces 1 |
| Hellinger distance \(H\) | Distance of histograms | Binary on \(\Delta_{2^m}\) | Full \(P^\star\) or second histogram | **Metric** (\(L^2\)) | Preserves "closer on the simplex" | Distance (ratio-like) | Means, triangle bounds | Class H | Needs \(P^\star\); \(O(2^m)\) |
| Hellinger fidelity \(F_H\) | Similarity of histograms | Binary on \(\Delta_{2^m}\) | Full \(P^\star\) | Not a metric (similarity) | Preserves "more overlap" if used as similarity | At least ordinal; ratio claims unsafe | Convert to \(H\) or \(\theta\) before composing | Class H | Same information as \(H\) per job; mean(\(F_H\)) ≠ transform of mean(\(H\)) |
| Hellinger infidelity \(1-F_H\) | Induced dissimilarity | Binary on \(\Delta_{2^m}\) | Full \(P^\star\) | **Not** generally a metric | Do not treat as a distance | — | Use \(H=\sqrt{1-\sqrt{F_H}}\) or \(\arccos\mathrm{BC}\) | Class H | Counter-example needed if we claim M4 failure |
| TVD | \(L^1\) distance of histograms | Binary on \(\Delta_{2^m}\) | Full \(P^\star\) | **Metric** (\(L^1\)) | Preserves "closer in \(L^1\)" | Distance (ratio-like) | Means, triangle bounds | Class TV | Needs \(P^\star\); phase-blind |
| TVD fidelity \(1-\mathrm{TVD}\) | Similarity | Binary on \(\Delta_{2^m}\) | Full \(P^\star\) | Not a metric | Same attribute as TVD | Complement of a distance | Per-job threshold equivalent to TVD | Class TV | Composition only after mapping back to TVD: \(s(P,R)\ge s(P,Q)+s(Q,R)-1\) |
| Uhlmann / DM fidelity | State overlap | Binary on density operators | Ideal state / process | Not a metric | Correct for states, not for bitstring jobs | Similarity | Induce Bures / sine distances | State-fidelity class | Tomography; exponential |
| Bures / sine distances | State distance | Binary on density operators | Ideal state | **Metrics** (cite Gilchrist et al. 2005; verify Ma et al.) | Preserves state closeness | Distance | Triangle bounds on states | State-fidelity class | Not a practical job-level tool |
| ESP | Predicted hardware success | Unary: circuit + calibration | Gate / readout errors | N/A | Prediction system, not a job measure | Derived product of probabilities | Compare predicted reliability, not observed histograms | None | Independent-error assumption; weak for deep circuits |
| PST | Return-to-zero rate of \(U^\dagger U\) | Unary: inverse-circuit job | All-zero target | N/A as a distance on outputs of \(U\) | Measures a different program | Derived proportion | Valid for the inverse-circuit attribute only | None with HF of \(U\) | Doubles depth; T1 bias on \(\lvert 0\rangle^{\otimes n}\) |
| Coarse TVD | Distance on \(\Delta_{K+1}\) | Binary on coarse histograms | Known \(E\), optional weights | **Metric** (TVD on the push-forward) | Preserves coarse closeness | Distance | Triangle bounds on coarse bins | Coarse-TV; equals \(1-p_E\) at \(K=1\) | Forgets structure inside "other" |
| Coarse TVD similarity | Coarse similarity | Binary on \(\Delta_{K+1}\) | Known \(E\) | Not a metric | Complement of coarse TVD | Complement | Per-job equivalent to coarse TVD | Coarse-TV | Same \(K=1\) collapse |
| Coarse Hellinger distance | Distance on \(\Delta_{K+1}\) | Binary on coarse histograms | Known \(E\) | **Metric** (Hellinger on the push-forward) | Preserves coarse closeness | Distance | Triangle bounds on coarse bins | Coarse-H | Convention: codebase uses \(\sqrt{1-\mathrm{BC}}\), Qiskit HF uses \((1-H^2)^2=\mathrm{BC}^2\) |
| Coarse Hellinger fidelity | Coarse similarity | Binary on \(\Delta_{K+1}\) | Known \(E\) | Not a metric | Complement-like similarity | At least ordinal | Convert to coarse \(H\) to compose | Coarse-H; equals \(p_E\) at \(K=1\) | Same information as \(p_E\) when \(K=1\) |
| Michelson DSR | Expected-peak dominance | Unary: job \(+\,E\) | Known \(E\) | N/A | Broken below contrast 0 by clipping; \(K\) changes the attribute | Clipped derived | Order among jobs with the same \(K\) only | None | Mean expected peak vs max competitor; T1 bias |
| Pauli expectation \(\hat ev\) | Value of \(\langle O\rangle\) | Unary on \([-1,1]\) (typical Pauli) | Observable \(O\) | N/A as a distance of histograms | Preserves "larger expectation" on that \(O\) | Interval (0 is not "no success") | Differences; not a success rate | Estimator-value | Same shots as Sampler; different attribute |
| \(\lvert\hat ev-\mathrm{ideal}\rvert\) | Deviation of an expectation | Binary on \([-1,1]\) | Ideal \(\langle O\rangle\) | **Metric** on the value line | Preserves "closer expectation" | Distance | Triangle bounds per observable | Class EV | One observable; ignores histogram shape |
| Observable fidelity \(1-\lvert\Delta ev\rvert/2\) | Similarity of expectations | Binary on \([-1,1]\) | Ideal \(\langle O\rangle\) | Not a metric | Complement of \(\lvert\Delta ev\rvert/2\) | Similarity | Per-observable threshold-equivalent to \(\lvert\Delta ev\rvert\) | Class EV | Code clips to \([0,1]\); mean over observables is a new derived measure |
| Estimator success probability \((1+\hat ev)/2\) | +1-eigenvalue mass of a Pauli | Unary | Pauli \(O\) | N/A | Not \(p_E\); do not treat as shot success on \(E\) | Derived proportion | Meaningful for that Pauli only | Affine of \(\hat ev\) | Easy to confuse with Sampler \(p_E\) |
| Relative error | Relative deviation | Unary vs ideal | Ideal \(\langle O\rangle\) | N/A | Breaks when ideal \(\approx 0\) | Ratio-like only if zero is well away | Avoid near-zero ideals | None | Unstable for vanishing correlators |
| SNR \(\lvert\hat ev\rvert/\mathrm{std}\) | Estimate stability | Unary | Shots / std | N/A | "More stable" ≠ "more correct" | Derived | Compare precision, not success | None | Can be large on the wrong value |
| Depolarization factor | Global shrink toward 0 | Derived from several \(O_i\) | Ideals \(\neq 0\) | N/A | Model parameter, not a job success score | Clipped derived | Interpret as a noise model fit | None | Assumes a single shrink factor |

## Statements to prove or cite

### Threshold invariance (Class H and Class TV)

Let \(\phi\) be strictly monotone. For fixed \(Q\) and threshold
\(\tau\) on \(d(\cdot,Q)\), define
\(\tau'=\phi(\tau)\). Then
\(d(P,Q)\le\tau\) iff \(\phi(d(P,Q))\) stands in the corresponding
relation to \(\tau'\). Hence single-job decisions are invariant
inside a monotone class after threshold conversion.

### Aggregation is not invariant

There exist finite samples \(A,B\) of distances such that
\(\mathrm{mean}(H_A)<\mathrm{mean}(H_B)\) but
\(\mathrm{mean}(F_H(A))<\mathrm{mean}(F_H(B))\) fails to preserve the
intended ranking, because \(F_H=(1-H^2)^2\) is nonlinear. Construct
or find this on `DSR_result.csv` (Test B). If the real data do not
invert, report the negative result.

### Infidelity counter-example sketch

Do not test M4 on \(F_H\). For \(d=1-F_H=1-\mathrm{BC}^2\), look for
distributions \(P,Q,R\) on a small simplex such that
\(d(P,R)>d(P,Q)+d(Q,R)\). Standard references already prefer
\(H=\sqrt{1-\mathrm{BC}}\) or \(\arccos\mathrm{BC}\) as the metric
completions. If a clean counter-example is not written, cite
Gilchrist et al. / QuantumBenchmarkZoo and drop the claim.

### Coarse push-forward

The map that keeps each \(x\in E\) and lumps \(\{0,1\}^m\setminus E\)
into `other` is a measurable push-forward. TVD and Hellinger
distance restricted to the image \(\Delta_{K+1}\) remain metrics
because they are the same functions on a smaller simplex.

### \(K=1\) collapse

Let \(E=\{x^\star\}\), \(p=p_{x^\star}\). Coarse observed
\((p,1-p)\), coarse ideal \((1,0)\).

\[
\mathrm{TVD}=\tfrac12\bigl(|p-1|+|(1-p)-0|\bigr)=1-p,
\]

\[
\mathrm{BC}=\sqrt{p\cdot 1}+\sqrt{(1-p)\cdot 0}=\sqrt{p},
\quad
\mathrm{BC}^2=p.
\]

So coarse TVD similarity and coarse Hellinger fidelity both equal
\(p_E\). Chance-corrected success is
\(\mathrm{clip}((p-b)/(1-b),0,1)\), one degree of freedom plus a
clip. This identity is about the coarse profile, not about
full-histogram HF. Finite-iteration Grover has residual amplitude
outside \(E\), so full HF need not equal \(p_E\).

## GQM triples (for the table's "why this number exists")

| Quantity | Goal | Question | Then this number |
|---|---|---|---|
| TVD or \(H\) | Judge closeness to a reference | How far is the histogram from \(P^\star\)? | Distance on \(\Delta_{2^m}\) |
| \(p_E\) | Judge known-answer success | What fraction of shots landed in \(E\)? | Unary score |
| Chance-corrected | Judge success above guessing | How much of the possible above-chance mass did we get? | Clipped score |
| Michelson DSR | Judge distinguishability | Do expected peaks beat the strongest competitor? | Clipped contrast |
| Coarse TVD / \(H\) | Judge closeness when \(P^\star\) is unavailable | How far is the coarse histogram from the task reference on \(E\)? | Distance on \(\Delta_{K+1}\) |
| ESP | Predict hardware success | What success does calibration imply before the job? | Prediction |
| PST | Validate \(U^\dagger U\) | How often do we return to zero? | Different program |
| \(\lvert\hat ev-\mathrm{ideal}\rvert\) | Judge observable accuracy | How far is the estimate from the ideal value? | Metric on \([-1,1]\) |
| Observable fidelity | Same, as a similarity | How close is the estimate to the ideal value? | Similarity (Class EV) |
| SNR | Judge whether I can decide | Is the estimate stable enough? | Precision, not success |

## Open items

- Defend the scale type of \(p_E\) in the article text.
- Confirm sine-distance attribution before citing arXiv:0808.0984.
- Write the \(1-F_H\) counter-example or drop that row.
- Fill Test B before claiming that aggregation *does* change
  conclusions on our jobs.
