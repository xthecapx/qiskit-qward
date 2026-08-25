# Literature and novelty gate

Question the gate must answer:

> Does prior work already apply representational measurement theory
> or GQM specifically to job-level quantum execution-success
> quantities (success rate, HF, TVD, PST, ESP, DSR)?

Standard papers that prove Hellinger distance and TVD are metrics,
or that fidelity is not a distance, **do not** occupy this gap.

Status after the first search pass: **gap tentatively open**.
Confirm or refute with Question C in
[../elicit-prompt.md](../elicit-prompt.md) before drafting.

## What is already standard (do not claim as novel)

| Work | What it already does | Columns C1–C8 |
|---|---|---|
| Gilchrist, Langford, Nielsen, PRA 71:062310 (2005) | Enumerates criteria a QIP distance must meet; assesses many candidates; recommends theoretically and experimentally meaningful distances. Primary Question A source | C1–C3 yes; C4–C6 no |
| Luo and Zhang, PRA 69:032106 (2004) | Quantum Hellinger / affinity vs Bures / fidelity; infinitesimal form is Wigner–Yanase skew information | C1–C3 yes |
| Ma, Zhang, Chen, Phys. Lett. A 373:3407 (2009) | Metrics *induced by* fidelity (trace, sine, Bures); fidelity itself is the generator, not the metric | C1–C3 yes |
| Mendonça et al., PRA 78:052330 (2008) | Alternative fidelity as a *similarity*; Jozsa axioms; then identifies metrics on density matrices | C3 yes (explicit "measure of similarity") |
| Endres and Schindelin, IEEE TIT 49:1858 (2003) | A bounded metric on classical probability distributions (Jensen–Shannon / capacitory family); \(\sqrt{\chi^2}\) as approximation | C1–C2 yes on the simplex; not about quantum fidelity |
| Nielsen and Chuang (2010) | Textbook: Uhlmann fidelity, trace distance, relation to metrics | C1–C3 yes |
| QuantumBenchmarkZoo "Fidelities and Errors" | Table: HF is not a metric; Hellinger distance and TVD are; KL is not | C1–C3 yes; C4–C5 no |
| Qiskit `hellinger_fidelity` docs | Defines \(F_H=(1-H^2)^2=\mathrm{BC}^2\) | Implementation, not a theory paper |
| Fenton and Bieman 2015; Fenton TSE 1994 | Representation condition, scale types, GQM, "when a measure is not a measure" | C4–C5 yes; **not applied to QC execution** |

These sources are Section 3 / 10 background.

## Closest QC reviews (competitors, not occupiers)

| Work | What it does | Why it does not close the gap |
|---|---|---|
| Mohapatra et al., QCE 2025 | Correlates HF, TVD, ESP, PST, Clifford, shadows with DM fidelity; uses the better proxies for error mitigation | Correlation / proxy validation (Fenton: prediction-system validation). No M1–M4 audit, no representation condition, no GQM for "mark this job successful" |
| ASE 2025, *Is Measurement Enough?* | Classifies quantum testing oracles as distribution-level vs output-value-level | Right SE neighborhood. Does not ask whether the oracles are metrics or valid Fenton measures |
| Fortunato et al., arXiv:2410.00650 | Survey of quantum software testing; KS, chi-squared, distances as oracles with thresholds | Testers already use TVD / Hellinger as pass/fail oracles without scale or axiom analysis |
| Andrews et al., IEEE Design & Test 2026 | Survey of functional testing of quantum circuits | Same family as Fortunato; no Fenton audit |
| QUTest distribution asserts | TVD, Hellinger, KL, chi-squared pragmas | Practice, not theory |
| Cruz-Lemus et al. / Zhao QSE line (from thesis notes) | Quality attributes and understandability of *circuits* | Internal product metrics, not job-level execution success |
| Q-COSMIC / FSM for quantum | Functional size of quantum software | Different attribute (size), not execution success |
| Hacaloglu, Soubra, Bourque, Abran, IEEE Access 2026, [11554035](https://ieeexplore.ieee.org/document/11554035) | SLR: *Quantum Software Size: What Do We Measure, How and Why?* (16 studies; LOC, COSMIC, qubits/gates, cyclomatic). PDF: `on_limitations_metrics/pdf/...` | **Question C hit that does not close the gate.** Same "what / how / why" slogan, applied to *size* (internal product / project estimation), not to Sampler or Estimator *execution* attributes. Strengthens the gap: QSE is measuring size rigorously while execution "fidelity" remains untyped |
| Informal blog "How to Run a Meaningful Quantum Experiment" | Hypothesis then primary metric | GQM-like advice, not a peer-reviewed Fenton audit |

QCE26 reviews of our own DSR paper (`main-review.md`) are motivation
for a new question ("why call this a metric, and when does the choice
change a decision?"), not a literature gap by themselves.

## First-pass verdict

- **M1–M4 fidelity-versus-distance:** occupied and now confirmed by
  Elicit Question A. Cite Gilchrist et al. (2005) as the process-
  distance criteria paper; Luo and Zhang (2004) and Ma et al. (2009)
  for Hellinger / Bures / fidelity-induced metrics; Endres and
  Schindelin (2003) if we mention a classical simplex metric besides
  TVD / Hellinger. Demote the axiom list to preliminaries.
- **Fenton representation / scale type applied to HF, TVD, \(p_E\),
  PST, ESP, DSR:** not found in this pass.
- **GQM tree whose goal is "mark this QPU job successful":** not
  found as a peer-reviewed treatment. Closest is generic "choose a
  primary metric" advice and testing-oracle surveys.
- **Consequence analysis** (single-job invariance versus aggregation
  / composition / missing \(P^\star\)): not found as an explicit
  result on quantum execution measures.

Therefore the paper should lead with:

1. when the distinction changes conclusions (Section 2);
2. Fenton / GQM applied to execution-success quantities
   (Sections 4, 6, 8);
3. a critical typing of DSR (Section 9);

and should **not** lead with "we discovered that HF is not a metric."

## What would fail the novelty gate

Any peer-reviewed paper that:

- names Fenton's representation condition or scale types, and
- applies them to at least HF / TVD / success rate as job-level
  success quantities, or
- publishes a GQM model for marking QPU jobs successful and
  classifies those same quantities.

If Elicit Packet C returns such a paper, shrink the contribution to
the decision guide plus DSR typing and rewrite Section 1.

## Papers already on disk

See `docs/papers/` (Mohapatra PDF is linked from the QCE 2025 page;
`metrics.pdf` is Fenton; QCE26 v2 is the previous DSR paper). Log
new Elicit PDFs here as they arrive.

## Elicit log

| Date | Packet | Result | Action |
|---|---|---|---|
| 2026-08-24 | Informal web search (A/B/C) | No Fenton/GQM + QC-execution hit | Gap tentatively open |
| 2026-08-24 | Elicit A | 10 theoretical hits (Gilchrist, Luo, Ma, Mendonça, Endres, Chen, Reeb, Spehner, Zhang–Wu, Toth–Pitrik). See table below | Confirms axiom literature is occupied. Must-cite: Gilchrist 2005, Luo 2004, Ma 2009. No Fenton/GQM, no job-level Sampler/Estimator success |
| 2026-08-24 | Elicit B | Mohapatra et al. QCE 2025 (HF, TVD, ESP, PST vs DM fidelity); Gilchrist 2005 again | Closest competitor on *which number to use*, but the criterion is correlation with DM fidelity, not Fenton/GQM or job-level success. Gilchrist is Question A overlap |
| 2026-08-24 | Elicit C | Hacaloglu et al. IEEE Access 2026 (quantum software *size* SLR) | Logged as competitor on "what/how/why"; does **not** apply Fenton/GQM to HF, TVD, \(p_E\), or Estimator expectations. Gap still open |

## Elicit Question A — paper-by-paper

None of these papers applies Fenton, GQM, Sampler vs Estimator, or a
job-level success criterion. They occupy Section 3 / 10 only.

| Key | Use in our paper | C1–C8 (from abstracts) |
|---|---|---|
| `gilchristDistanceMeasuresComparea_2005` | **Must cite.** Criteria a QIP *distance* must satisfy; fidelity is assessed as a candidate and is not adopted as the gold-standard distance | C1–C3 yes |
| `luoInformationalDistanceQuantumstate_2004` | **Must cite** if we mention quantum Hellinger / affinity vs Bures / fidelity | C1–C3 yes |
| `maFidelityInducedDistance_2009` | **Must cite** for "fidelity induces Bures / sine / trace-type metrics; fidelity is not itself a metric." Replaces the unverified arXiv:0808.0984-only cite | C1–C3 yes |
| `mendoncaAlternativeFidelityMeasure_2008` | Cite as explicit language: fidelity is a *similarity*; metrics are identified separately | C3 yes |
| `endresNewMetricProbability_2003` | Optional classical-simplex sibling of TVD / Hellinger (Jensen–Shannon family). Useful if Section 3 mentions more than two classical metrics | C1–C2 yes |
| `chenSuperFidelityRelated_2011` | Optional. Another fidelity-induced metric (super-fidelity). Same moral as Ma 2009; do not stack both unless discussing superfidelity | C1, C3 |
| `spehnerQuantumCorrelationsDistinguishability_2014` | Optional survey for related work (Bures, relative entropy, distinguishability) | C3; review |
| `zhangLowerBoundFidelity_2013` | Optional. Treats fidelity and Bures as distinct (bounds one by the other) | C3 |
| `reebHilbertsProjectiveMetric_2011` | Peripheral. Hilbert projective metric on cones; not a job-level success measure | skip unless a reviewer asks about cone metrics |
| `tothQuantumWassersteinDistance_2025` | Peripheral. Quantum Wasserstein; triangle inequality only in special cases. Shows the "is this a metric?" question is still live, but not about histogram HF / TVD | skip for the main text |

**Take-away for drafting.** Question A succeeded: we now have citable
proofs that the field already distinguishes fidelity (similarity)
from distance (metric). Do not claim that distinction as a result.
Lock Gilchrist 2005 + Ma 2009 (published PLA, not only the arXiv
note) as the two backbone citations in Section 3.

## Elicit Question B — paper-by-paper

| Key | Use in our paper | Why it does not occupy our gap |
|---|---|---|
| `mohapatraBenchmarkingFidelityMetricsa_2025` | **Must cite** as the closest empirical competitor. HF, TVD, ESP, PST, Clifford, shadows vs DM / Uhlmann fidelity on SupermarQ + IBM noise models; ESP Spearman 0.88, PST Pearson 0.92; RZNE application | Validates *proxies of DM fidelity* (Fenton: prediction-system validation). Does not ask M1–M4, representation condition, scale type, Sampler vs Estimator interpretation, or how to mark a known-answer job successful. Their recommended quantities (ESP, PST) are not even histogram-success scores |
| `gilchristDistanceMeasuresComparea_2005` | Already a Question A must-cite. Question B only re-found it | Process-distance gold standard, not a job-success GQM |

**How to position Mohapatra in Section 10.** They answer “which cheap number tracks DM fidelity?” We answer “which number is a valid measure of the attribute the developer actually decided to read from the shots?” Those can disagree: ESP/PST can correlate with DM fidelity and still be the wrong attribute for a Grover known-answer job or a VQE energy job.
