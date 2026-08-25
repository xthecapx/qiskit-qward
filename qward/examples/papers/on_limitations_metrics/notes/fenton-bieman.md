# Fenton and Bieman 2015 — working notes

Source: [docs/papers/metrics.pdf](../../../../../docs/papers/metrics.pdf),
3rd edition. Page numbers below are **book pages** as printed in the
PDF (Chapter 1 begins on book p. 3).

Do not run a Weyuker complexity-axiom audit. Fenton treats those
properties as the wrong tool unless the attribute is software
complexity. The claim we take from the book is:

> Define the attribute first, then choose a scale, then only then pick
> a formula.

## How the book enters the paper

Two complementary tests for the word "metric":

1. Mathematical M1–M4 (distance on a space).
2. Fenton's representational theory (valid measure of a named
   attribute).

A quantity can pass one test and fail the other. Both failures are
limitations, but they are different limitations.

---

## Chapter 1 — Measurement: What Is It and Why Do It? (pp. 3–22)

### Formal definition (p. 5)

> Measurement is the process by which numbers or symbols are assigned
> to attributes of entities in the real world in such a way so as to
> describe them according to clearly defined rules.

Consequences for QC:

- We do not "measure a circuit" or "measure a job." We measure a
  named attribute of a named entity (histogram closeness to a
  reference; mass on \(E\); peak dominance).
- Loose talk that "HF is the metric of the job" collapses entity and
  attribute, which Fenton calls unacceptable for scientific work
  (p. 5).

### Galileo line (p. 7)

"What is not measurable make measurable" is not permission to assign
any number. Chapter 2 is required before the assignment counts as
measurement.

Hubbard's alternative (quoted p. 8): measurement as a quantitatively
expressed reduction of uncertainty. Useful later for shot-noise and
thresholds, not as a replacement for the representation condition.

### Direct measurement versus calculation (p. 9)

Shot counts are closer to direct observation. Success rate, HF, TVD,
and DSR are **derived** (calculated) quantities. Fenton will later
require meaningfulness arguments for derived measures (Ch. 2.4.4).

### Scope of "software metrics" (pp. 17–21)

The book already warns that the phrase "software metrics" names many
unrelated activities. QC inherited the same baggy vocabulary:
fidelity, success rate, ESP, PST, and DSR are all sold as "metrics."

### Relevance to Section 1 and 4 of the paper

Open the paper with a decision, not with a formula. Fenton p. 16:
measurement is for understanding, control, and improvement — not for
producing a number.

---

## Chapter 2 — The Basics of Measurement (pp. 25–85)

This is the theoretical spine.

### 2.1 Representational theory (pp. 26–40)

**Empirical relations (2.1.1, p. 27).** An attribute is understood
only after we can state empirical relations on entities
("taller than", "indistinguishable from", "closer than").

For QC execution, write the empirical relation before the formula:

| Attribute | Empirical relation we claim to preserve |
|---|---|
| Histogram closeness | Job A is closer to \(P^\star\) than job B |
| Task success | Job A placed more mass on \(E\) than job B |
| Peak dominance | Expected outcomes stand out more in A than in B |
| Hardware reliability | Circuit A is less likely to fail than B (ESP / PST) |

**Representation condition (2.1.3, p. 33).** A mapping \(M\) from
entities to numbers is a measurement of the attribute only if
empirical relations are preserved by numerical relations. Informally:
if we judge A closer than B, then \(M(A)\) must stand in the
corresponding numerical relation to \(M(B)\).

Failures we will use:

- Clipped chance-corrected success maps every below-chance job to 0,
  so it cannot preserve a "worse than chance" ordering.
- Michelson DSR maps every non-dominant expected peak to 0, so it
  cannot preserve fine-grained below-contrast orderings.
- Using HF to answer "did enough shots land in \(E\)?" measures the
  wrong attribute even if HF is a valid similarity of histograms.

### 2.2 Measurement and models (pp. 40–51)

- Define the attribute before the instrument (p. 42).
- Direct versus derived measurement (p. 44). HF, TVD, DSR, and
  \(p_E\) are derived from counts.
- Measurement for prediction (p. 47) is a different activity from
  measurement of a current attribute. ESP is closer to a prediction
  system (Ch. 3.4) than to a job-level success measure.

### 2.3 Scale types (pp. 51–60)

| Scale | Admissible transforms | Meaningful statements |
|---|---|---|
| Nominal | Permutation of labels | Equality only |
| Ordinal | Strictly increasing | Order, not differences or ratios |
| Interval | \(ax+b\), \(a>0\) | Differences; not ratios |
| Ratio | \(ax\), \(a>0\) | Ratios; zero is meaningful |
| Absolute | Identity only | Counts; "there are 1024 shots" |

Consequences for QC:

- A universal cutoff such as "success if score \(> 0.8\)" is
  meaningful only after the scale is defended. On a merely ordinal
  similarity it is a convention, not a measurement fact (p. 61–65).
- "HF = 0.90 is twice as good as HF = 0.45" is a ratio statement.
  It is not automatically meaningful for a nonlinear transform of a
  distance.
- Arithmetic means are not scale-free. Averaging HF across jobs and
  averaging Hellinger distance across the same jobs can rank
  providers differently (paper Section 2).
- \(p_E\) sits awkwardly between absolute (it is a relative
  frequency of counts) and ratio. The paper must pick a position and
  defend it. First-pass position: treat shot counts as absolute and
  \(p_E\) as a derived ratio-scale proportion with a meaningful zero
  and a meaningful unit of "fraction of shots," while refusing ratio
  claims that compare \(p_E\) to HF.

### 2.4 Meaningfulness (pp. 61–78)

- Statistical operations must match the scale (p. 65). Medians are
  safer than means when the scale is in doubt.
- Objective versus subjective measures (p. 68): success criteria
  \(E\) are part of the measure. Changing \(E\) changes the
  attribute.
- Derived measurement and meaningfulness (p. 75): \(1-\mathrm{TVD}\)
  and \(\mathrm{BC}^2\) are derived. Their legal operations are not
  inherited automatically from TVD or Hellinger distance.

### Mapping to QC quantities (first pass)

| Quantity | Entity | Attribute | Tentative scale | Representation risk |
|---|---|---|---|---|
| Success rate \(p_E\) | Job + \(E\) | Mass on the goal set | Derived ratio / proportion | Valid for that attribute; not a distance |
| Chance-corrected success | Job + \(E\) + \(m\) | Above-chance mass | Clipped derived | Clipping breaks below-chance order |
| Hellinger distance | Pair of histograms | Distance on the simplex | Ratio-like distance | Valid metric of closeness |
| Hellinger fidelity | Pair of histograms | Similarity | At least ordinal; ratio claims unsafe | Same attribute as \(H\), different scale story |
| TVD | Pair of histograms | \(L^1\) distance | Ratio-like distance | Valid metric of closeness |
| \(1-\mathrm{TVD}\) | Pair of histograms | Similarity | Complement of a ratio distance | Threshold-equivalent to TVD per job |
| ESP | Circuit + calibration | Predicted success | Prediction system | Not a measure of the observed job |
| PST | Inverse-circuit job | Return-to-zero rate | Derived proportion | Measures \(U^\dagger U\), not \(U\) |
| Michelson DSR | Job + \(E\) | Peak contrast | Clipped derived | Mean-vs-max; \(K\) dependence; T1 bias |

---

## Chapter 3 — A Goal-Based Framework (pp. 87–131)

### 3.1 Classifying software measures (pp. 87–99)

Process / product / resource / change. Job-level success quantities
are **external product attributes of an execution**, not internal
circuit attributes (depth, T-count). Do not mix pre-runtime
complexity metrics with post-runtime success scores in one GQM leaf.

### 3.2 Goal-Question-Metric (pp. 100–108)

GQM is the decision-guide backbone (paper Section 8).

Template we will instantiate:

- **Goal.** Analyze a QPU job for the purpose of deciding success
  from the point of view of the algorithm developer, in the context
  of a known-answer or known-distribution task.
- **Questions.**
  1. How far is the observed histogram from the chosen reference?
  2. Did enough shots land in \(E\)?
  3. Do expected peaks dominate the strongest competitor?
  4. Is the ideal histogram even computable?
- **Metrics / measures.** Chosen only after the question: TVD or
  Hellinger distance; \(p_E\) or chance-corrected success; Michelson
  DSR; histogram-free coarse distances when \(P^\star\) is
  unavailable.

### 3.4–3.5 Validation (pp. 117–126)

Distinguish:

- **Validating a measure** (does the number preserve the empirical
  relations of the named attribute?).
- **Validating a prediction system** (does ESP or a reliability
  model predict later observations?).

Mohapatra et al. 2025 correlate HF / TVD / ESP / PST with DM
fidelity. That is closer to validating a prediction or proxy system
than to validating a measure of "this job succeeded."

"How not to validate" (p. 125): do not declare a quantity valid
because it correlates with another quantity whose own attribute is
undefined.

### 3.4.3 Mathematical perspective of metric validation (p. 120)

Use this subsection when we say a true distance can still fail as a
software measure: M1–M4 are about the numerical mapping on a space,
not about whether the space is the right attribute.

---

## Chapter 4 — Empirical Investigation (pp. 133–182)

We are not designing a new experiment. We reuse existing jobs. The
write-up must still name threats (paper Section 11).

From 4.1.4 (p. 143) and 4.2 (pp. 145–170), the relevant threats are:

- **Conclusion validity.** Shot noise; small \(n\) in some
  (algorithm, qubit) groups (see QFT 5-qubit \(n=4\) in
  `narrative_assessment.md`).
- **Internal validity.** Optimization level, calibration drift, and
  provider toolchain are confounded with "provider."
- **Construct validity.** Using HF to stand for "algorithm success"
  when the empirical relation is "mass on \(E\)."
- **External validity.** Superconducting IBM / Rigetti jobs only;
  known-answer algorithms only.

Study type for Section 7: retrospective analysis of existing
measurements (Fenton 4.3.4, p. 173), not a fresh controlled
experiment. Hypotheses must be stated before looking for inversions
(4.1.2, p. 139):

- H-A: within a monotone family, single-job orderings agree.
- H-B: arithmetic means of a similarity and of its distance partner
  can disagree on provider or workload rank.
- H-C: full-simplex distances become unavailable while known-\(E\)
  scores remain defined.

---

## Chapter 6 — Analyzing Software Measurement Data (pp. 225–289)

Use this chapter when reporting Tests A–C. Do not copy the whole
toolbox.

### Scale-appropriate statistics (pp. 232–243)

- If the scale of HF is in doubt, prefer medians, box plots, and
  rank tests to arithmetic means.
- Test B uses the mean *on purpose*, because the mean is the
  operation that can break monotone equivalence. Report medians as
  the contrast: medians of a monotone transform should not invert
  group order.

### Techniques we already have in the repo

The existing profile comparison uses Mann–Whitney U, Wilcoxon,
Cliff's \(\delta\), and bootstrap CIs
(`statistical_comparison_profile.py`). Those are ordinal-safe and
match Fenton 6.6.2 (two-group tests, p. 281). Keep them for
provider comparisons. Add mean-based rankings only for Test B.

### What not to do

- Do not average HF, TVDF, success rate, and Michelson DSR into one
  "overall metric." That is a derived measure with no stated
  attribute (Ch. 2.4.4).
- Do not treat a 0.8 cutoff as scale-independent.

---

## Chapters we are not using as pillars

- Ch. 5 data-collection forms: only if we later document how
  `DSR_result.csv` was built.
- Ch. 7 causal / Bayesian models: out of scope.
- Ch. 8–11 size, structure, quality models, reliability growth:
  useful for related-work contrast (internal product metrics versus
  execution-success measures), not for the audit table.

## Sentences the paper can quote in paraphrase

1. We measure attributes of entities, not "things" (p. 5).
2. The representation condition is the test of a measure (p. 33).
3. Meaningful statements depend on scale type (pp. 51–65).
4. GQM: goals before questions before measures (pp. 100–108).
5. Validating a measure is not the same as validating a prediction
   system (pp. 117–119).
