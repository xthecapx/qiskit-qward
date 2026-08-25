# Two ways to understand the same bitstrings

Hardware returns shots. Qiskit's two primitives turn those shots into
different *attributes* of the same entity. That is a Fenton point
(measure attributes of entities, not "the job") and it must appear
before any audit table.

QWARD already splits the two paths in
`qward/metrics/fidelity_metrics.py` (Sampler vs Estimator) and
`qward/metrics/estimator_metrics.py`.

## Shared raw object

A shot is a bitstring \(x\in\{0,1\}^m\). A job is a finite multiset
of shots. Nothing in the hardware distinguishes Sampler from
Estimator at this layer.

The primitive chooses the **empirical relation** we will preserve:

| Primitive | What we keep from the shots | What we throw away | Decision the numbers can support |
|---|---|---|---|
| **Sampler** | The empirical distribution \(\hat P\), or the mass on a goal set \(E\) | Observable algebra; most phase information | "Did the histogram match the expected answers or \(P^\star\)?" |
| **Estimator** | Estimates \(\widehat{\langle O_i\rangle}\) (and stds) for chosen Pauli observables | The histogram as a histogram; which bitstrings produced the mean | "Did the energy / correlators match the intended values?" |

Applying a Sampler quantity to an Estimator job, or the reverse, is
a representation-condition failure: the mapping no longer preserves
the relation the developer actually cares about.

## Sampler interpretation (distribution / known-answer)

Derived objects:

- histogram \(\mathcal{C}\) and \(\hat P=\mathcal{C}/S\);
- goal set \(E\) and \(p_E\);
- optional full reference \(P^\star\);
- DSR profile and full HF / TVD.

Typical algorithms in our corpus: Grover, QFT, BV, teleportation.

Success questions:

1. How far is \(\hat P\) from \(P^\star\)? → TVD or Hellinger
   distance (metrics on \(\Delta_{2^m}\)).
2. Did enough shots land in \(E\)? → \(p_E\), chance-corrected
   success (scores).
3. Do expected peaks dominate? → Michelson DSR (score, with T1
   bias).

## Estimator interpretation (observable / expectation)

The same shots are contracted against an observable. For a Pauli
\(O\) with eigenvalues in \(\{\pm 1\}\),

\[
\widehat{\langle O\rangle}
=\frac{1}{S}\sum_{s=1}^{S} \lambda(x_s),
\qquad
p_{\mathrm{succ}}=\frac{1+\widehat{\langle O\rangle}}{2}.
\]

`EstimatorMetrics` then derives:

| Quantity | Formula in code | Kind | Attribute |
|---|---|---|---|
| Expectation value \(\hat ev\) | job `evs` | derived estimate | value of \(\langle O\rangle\) |
| Success probability | \((1+\hat ev)/2\), clipped to \([0,1]\) | score | +1-eigenvalue mass if \(O\) is Pauli |
| Observable fidelity | \(1-\lvert\hat ev-\mathrm{ideal}\rvert/2\), clipped | similarity | closeness of one scalar to its ideal |
| Relative error | \(\lvert\hat ev-\mathrm{ideal}\rvert/\max(\lvert\mathrm{ideal}\rvert,\varepsilon)\) | score | relative deviation |
| SNR | \(\lvert\hat ev\rvert/\mathrm{std}\) | derived | estimate stability, not task success |
| Depolarization factor | mean of \(\hat ev_i/\mathrm{ideal}_i\) over nonzero ideals, clipped to \([0,1]\) | model parameter | global shrink toward 0 |

None of these is a metric on the histogram simplex. Observable
fidelity is a similarity on \(\mathbb{R}\) (actually on \([-1,1]\)
for Pauli expectations). The induced absolute deviation
\(\lvert\hat ev-\mathrm{ideal}\rvert\) *is* a metric on that
interval. The same monotone-class warning applies: averaging
fidelities across observables is not the same as transforming the
mean deviation.

## What must not be mixed

- Do not report Hellinger fidelity of an Estimator job. There is no
  \(\hat P\) in the result object, only `evs` / `stds`.
- Do not report DSR of an Estimator job unless you first reconstruct
  the raw counts and you have an \(E\). That changes the attribute
  back to Sampler.
- Do not treat `mean_success_probability` as \(p_E\). It is
  \((1+\langle O\rangle)/2\), not the mass on a bitstring set.
- Do not treat SNR as a success score. High SNR can sit on the
  wrong expectation.

## GQM fork (paper Section 8, first question)

**Goal.** Mark this quantum execution successful or not.

**Question 0.** Which attribute of the shots am I reading?

- Histogram / known-answer → Sampler branch (existing guide).
- Observable / energy / correlator → Estimator branch:
  - How far is \(\hat ev\) from the ideal? →
    \(\lvert\hat ev-\mathrm{ideal}\rvert\) (metric on \([-1,1]\))
    or observable fidelity (similarity).
  - Is the estimate stable enough to decide? → SNR and shot budget
    (not a success score).
  - Is a single Pauli enough, or do I need a Hamiltonian sum? →
    say whether you aggregate over observables and on which scale.

Variational algorithms live on the Estimator branch. That is why
DSR's known-\(E\) restriction is a scope limit, not a defect of
the Estimator quantities.

## Worked Estimator sketch (for the paper)

Ideal \(\langle ZZZZ\rangle=1\), observed \(0.70\), std \(0.04\):

- success probability \((1+0.70)/2=0.85\);
- observable fidelity \(1-\lvert 0.70-1\rvert/2=0.85\);
- relative error \(0.30/1=0.30\);
- SNR \(0.70/0.04=17.5\).

Threshold conversion: "pass if \(|\hat ev-\mathrm{ideal}|\le 0.20\)"
is the same single-observable decision as "pass if observable
fidelity \(\ge 0.90\)". Aggregation across six GHZ observables
(`qward/examples/estimator_ibm_experiment.py`) can break that
equivalence for the same reason as mean \(H\) vs mean \(F_H\).

## Reference classes (C7 — write this; do not cite it as prior art)

Elicit column C7 asked whether each *paper* required \(E\),
\(P^\star\), tomography, or an inverse circuit. The reviews do not
answer that per quantity. The article must introduce the classes:

| Class | What the developer must supply | Quantities that need it | Fails when |
|---|---|---|---|
| **R1** Known \(E\) | Analytically known answer bitstrings | \(p_E\), CCS, coarse TVD/\(H\), Michelson DSR | Variational landscape, no discrete answers |
| **R2** Full \(P^\star\) | Ideal distribution on \(\{0,1\}^m\) | Hellinger distance, \(F_H\), TVD, \(1-\mathrm{TVD}\) | \(m\) past the simulation wall |
| **R3** Ideal expectations | \(\langle O_i\rangle\) or a Hamiltonian | Observable fidelity, \(\lvert\Delta ev\rvert\), relative error | Unknown observable, or only counts kept |
| **R4** Tomography / uncompute | Ideal state or \(U^\dagger U\) | DM / Uhlmann fidelity, PST | Cost of tomography or doubled depth |

Sampler vs Estimator (Question 0) chooses the *attribute*. R1–R4
choose whether that attribute is even computable. Both belong in
Section 5 as our framing, not as a finding extracted from Mohapatra
or Gilchrist.

## Implications for the thesis

The limitation of QC "metrics" is not only score vs distance. It is
also **same shots, different attribute**. A number that is valid on
the Sampler branch can be meaningless on the Estimator branch even
when both numbers lie in \([0,1]\) and both are called fidelity.
