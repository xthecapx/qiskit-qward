# On the Limitation of QC Metrics

Working outline for an independent journal paper of about **10 pages**,
so the text can be cut or expanded for different venues. DSR is a
critical case study, not the headline. Do not treat this as a revision
of the QCE26 DSR submission.

The LaTeX source is [draft.tex](draft.tex) with [refs.bib](refs.bib).
A markdown sibling is [draft.md](draft.md). Both are a base for
rewriting, not a submission-ready manuscript.

Companion notes:

- [notes/fenton-bieman.md](notes/fenton-bieman.md)
- [notes/axiom-audit.md](notes/axiom-audit.md)
- [notes/dsr-typing.md](notes/dsr-typing.md)
- [notes/literature.md](notes/literature.md)
- [notes/output-types.md](notes/output-types.md)
- [notes/examples.md](notes/examples.md)
- [notes/consequence-gate.md](notes/consequence-gate.md)
- [elicit-prompt.md](elicit-prompt.md)
- [bibliography.md](bibliography.md)
- [research-checklist.md](research-checklist.md)

## Thesis

QC execution quantities are often reported under the common label
"metric," but they encode different attributes, live on different domains,
permit different mathematical operations, and require different reference
information. This distinction is:

- usually **decision-invariant for one job, one fixed reference, and a
  correctly transformed threshold** when two quantities are strictly
  monotone transforms (for example TVD and \(1-\mathrm{TVD}\));
- potentially **decision-relevant under aggregation across jobs,
  composition of subsystem errors, ratio or interval interpretations,
  cross-task comparison, and unavailable reference distributions**.

The paper must not claim that the name "metric" alone changes every
pass/fail decision. It will identify precisely where terminology is
harmless, where operations become invalid, and where conclusions can
change.

## Working rule of language

- Reserve **distance metric** for a function \(d: X \times X \to \mathbb{R}\)
  that satisfies M1–M4 on a named space \(X\).
- Call a quantity a **measure**, **score**, or **similarity** only after
  naming its attribute, domain, empirical relation, and scale type
  (Fenton and Bieman 2015, Ch. 2–3).
- Evaluate fitness for a stated decision. A non-metric score can be a
  valid measure of a well-named attribute. A true metric can still be
  the wrong attribute for the decision.

## Section plan

Journal structure in [draft.tex](draft.tex): Introduction,
Background, Related Work, Execution Model, Classification,
Analytical Results, Evaluation, Discussion, Threats to Validity,
Conclusion. Title: *On the Limitations of Metrics for Quantum
Circuit Execution*.

### 1. Introduction

Open with a concrete execution-evaluation decision: a developer has
shots from a QPU job and must mark the execution successful or not.
The first fork is how those shots are to be understood — as a
Sampler histogram / known-answer set, or as an Estimator contraction
against observables. Both start from bitstrings; they measure
different attributes.

State the negative result honestly: monotone-equivalent distance and
similarity pairs produce the same single-job decision after threshold
conversion.

Motivate the real risks:

- aggregation across jobs or providers;
- composition of subsystem errors;
- scale interpretation (ratio statements, universal cutoffs);
- unavailable ideal distributions.

Contributions:

1. consequence analysis of when the distinction changes conclusions;
2. Fenton / GQM audit of job-level success quantities;
3. a practitioner decision guide;
4. a critical typing of the DSR profile.

### 2. When the distinction changes conclusions

- Prove threshold invariance for strictly monotone transforms in the
  single-job, fixed-reference setting.
- Show why arithmetic aggregation is not invariant under nonlinear
  transforms and can change provider or workload rankings.
- Explain what the triangle inequality enables for composition and
  bounds; similarities may require transformation back to a distance.
- Explain why ratio statements and universal cutoffs require a
  defensible scale and empirical calibration (Fenton Ch. 2.4).

### 3. Preliminaries: metric, similarity, divergence, and score

- M1 non-negativity, M2 identity of indiscernibles, M3 symmetry,
  M4 triangle inequality.
- Nearby objects: pseudometric, quasimetric, divergence, similarity /
  fidelity, unary score.
- Why \(1-d\) is not itself a metric, even when \(d\) is.
- Why a similarity cannot "fail the triangle inequality": the axiom is
  not defined for that object. Audit induced dissimilarities separately.

### 4. Representational measurement theory for QC execution

- Fenton Ch. 2: empirical relations, representation condition, five
  scale types, meaningfulness.
- Fenton Ch. 3: GQM; validating a measure versus validating a
  prediction system.
- Direct versus derived measurement: shot counts are closer to
  absolute-scale observations; most published "fidelities" are derived.

### 5. The objects of quantum success evaluation

- Shot, job, batch: the shared raw object is a multiset of
  bitstrings. See [notes/output-types.md](notes/output-types.md).
- **Sampler interpretation:** histogram \(\mathcal{C}\), expected
  set \(E\), full ideal \(P^\star\), inverse-circuit PST.
- **Estimator interpretation:** the same shots contracted to
  \(\widehat{\langle O_i\rangle}\), then success probability,
  observable fidelity, relative error, SNR, depolarization.
- Information requirements (**C7, original text** — the literature
  does not classify quantities this way). State four reference
  classes in the article:
  1. known answer set \(E\) (Sampler scores, coarse distances, DSR);
  2. full ideal histogram \(P^\star\) (HF, TVD, Hellinger distance);
  3. ideal expectation values \(\langle O_i\rangle\) (Estimator
     fidelity / relative error);
  4. tomography or an inverse-circuit construction (DM fidelity,
     PST).
  A quantity is inapplicable when its reference class cannot be
  obtained, even if the formula is a valid metric on paper.
- Mixing a Sampler quantity onto an Estimator result (or the
  reverse) is a representation-condition failure, not a harmless
  change of formula.

### 6. Audit of QC execution quantities

One compact table (see [notes/axiom-audit.md](notes/axiom-audit.md)),
not long textbook subsections. Include:

- success rate and chance-corrected success;
- Hellinger distance and Hellinger fidelity;
- TVD and TVD fidelity;
- Uhlmann / DM fidelity and induced Bures / sine distances;
- ESP and PST;
- Michelson DSR and the coarse DSR quantities;
- Estimator: \(\widehat{\langle O\rangle}\),
  \((1+\hat ev)/2\), observable fidelity, relative error, SNR,
  depolarization factor.

Columns: claimed attribute; domain / arity; reference information;
M1–M4 classification; representation condition; scale type; meaningful
operations; monotone-equivalence class; principal limitation.

### 7. Empirical consequences using existing runs

Select examples only if they demonstrate one of the tests below. Do
not include datasets merely because they exist.

- **Test A — expected invariance:** identical ordering within monotone
  families for single-job comparisons.
- **Test B — aggregation sensitivity:** provider or workload conclusions
  under means versus medians of distance versus similarity transforms.
  A ranking inversion is strong evidence. Absence of inversion is a
  negative result and narrows the empirical claim.
- **Test C — information feasibility:** BV and broad-ideal data, where
  full-reference distances cannot be computed while known-\(E\) scores
  remain available.
- **Worked calculation:** one small histogram with every formula and
  a converted threshold, computed by hand.

Protocol: [notes/consequence-gate.md](notes/consequence-gate.md).
Candidate jobs: [notes/examples.md](notes/examples.md).

### 8. A GQM decision guide

Goal: mark this quantum execution successful or not.

**Question 0.** Are the shots being read as a histogram / known
answer, or as estimates of observables? (Sampler vs Estimator.)

Sampler questions, then quantities:

- How far is the histogram from a reference? Use a distance metric
  (TVD or Hellinger distance) on the space that matches the reference.
- Did enough shots land in \(E\)? Use a score (success rate;
  chance-corrected when \(K/2^m\) is not negligible).
- Do expected peaks dominate competitors? Michelson DSR, with its
  T1-bias and \(K\)-dependence stated.
- Is the ideal histogram computable? If not, only histogram-free
  scores and coarse-space distances remain.

Estimator questions, then quantities:

- How far is \(\hat ev\) from the ideal? Use
  \(\lvert\hat ev-\mathrm{ideal}\rvert\) (a metric on the value
  line) or observable fidelity (a similarity).
- Is the estimate stable enough to decide? SNR and shot budget —
  not a success score.
- If several observables are aggregated, state the scale; do not
  average fidelities as if that were the transform of the mean
  deviation.

Variational and chemistry workloads sit on the Estimator branch.
DSR does not apply there unless an \(E\) is defined.

Thresholds need shots, a chance baseline, and a stated scale type. A
single magic cutoff such as 0.8 is not a meaningful operation on every
scale.

### 9. Critical case study: typing DSR

Lead with weaknesses, then state the defensible scope. Full typing:
[notes/dsr-typing.md](notes/dsr-typing.md).

Weaknesses first:

- \(K=1\) degeneracy of the non-Michelson profile;
- clipping of chance-corrected success below chance;
- Michelson mean-versus-max asymmetry and dependence on \(K\);
- known-\(E\) requirement;
- phase blindness;
- T1 bias on zero-heavy targets;
- exclusion of variational landscapes.

Defensible scope: a known-answer evaluation profile containing scores
and two genuine coarse-space distances.

### 10. Related work

Position Fenton / GQM as the SE basis. Then quantum distance and
fidelity literature, benchmarking reviews, output-validation surveys,
and testing oracles. Do not claim that the M1–M4 distinction itself is
novel. See [notes/literature.md](notes/literature.md).

### 11. Threats to validity and limitations

Sampling uncertainty, calibration drift, reused heterogeneous jobs,
reference-distribution assumptions, aggregation choice, and author
bias from using DSR.

### 12. Conclusion

Choose the attribute and decision first. Then choose a quantity whose
domain, information requirements, scale, and permitted operations fit
that decision.

## Drafting notes for 10 pages

Keep Sections 2–3 and 6 compact. Use one worked Sampler histogram and
one Estimator scalar. Do not include all Grover / QFT / BV /
teleportation figures. Test B remains a protocol in the draft until
the read-only analysis on `DSR_result.csv` is run.

Gates:

1. **Novelty gate.** Status: [notes/literature.md](notes/literature.md).
2. **Consequence gate.** Protocol:
   [notes/consequence-gate.md](notes/consequence-gate.md).
   Read-only analysis of existing rows. No new QPU jobs.
