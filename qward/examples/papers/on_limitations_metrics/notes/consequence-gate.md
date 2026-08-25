# Consequence gate — read-only analysis design

Purpose: decide, **before drafting Section 7**, whether
distance-versus-similarity aggregation changes any provider or
workload ranking on jobs we already have.

This is a design, not a run. No new QPU jobs. No scripts are
executed here. When you run it, use existing
`qward/examples/papers/DSR_result.csv`.

## Hypotheses (state these before looking at rankings)

- **H-A (invariance).** For every job with a defined pair
  \((H,F_H)\) or \((\mathrm{TVD},1-\mathrm{TVD})\), the
  single-job order of two jobs is reversed exactly once when
  switching from the distance to its similarity. Spearman
  correlation of \(H\) with \(F_H\) is \(-1\) up to ties.
- **H-B (aggregation).** There exists a grouping (provider,
  backend, or algorithm × qubit count) such that the ranking
  implied by the **arithmetic mean** of a distance disagrees
  with the ranking implied by the arithmetic mean of its
  monotone similarity.
- **H-C (feasibility).** There exist jobs in the same corpus
  where \(H\) / TVD are missing because \(P^\star\) was not
  built, while \(p_E\) and the coarse profile are defined.

H-A is expected to hold. If it fails, the CSV or the transforms
are wrong.

H-B is the gate. If it fails on this corpus, Section 7 must
report a negative result and the paper becomes a methodological
article plus the teleportation score-versus-score example.

H-C is already supported by the broad-ideal and BV notes; the
run only needs to count rows with null `hellinger_fidelity` and
non-null `success_rate`.

## Data

File: `qward/examples/papers/DSR_result.csv`.

Columns required:

- grouping: `algorithm`, `backend_name`, `backend_type`,
  `num_qubits`, `execution_type`
- Class H: `hellinger_distance`, `hellinger_fidelity`
- Class TV: `tvd`, `tvd_fidelity`
- scores: `success_rate`, `chance_corrected_success`,
  `dsr_michelson`
- optional: `coarse_hellinger_distance`,
  `coarse_hellinger_fidelity`

Filter for Test A / H-B on Class H:

- drop rows where `hellinger_distance` or `hellinger_fidelity`
  is null;
- keep only QPU rows if the claim is about providers
  (`execution_type` in `{IBM_QPU, ...}` and Rigetti / AWS
  equivalents present in the file).

Provider label: map `backend_name` / `backend_type` to
`{IBM, Rigetti}` the same way
`statistical_comparison_profile.py` does. Do not invent a
third "other" bucket without writing it down.

## Test A procedure

1. Restrict to rows with both members of a pair.
2. Confirm the algebraic relation on a sample of rows:
   \(F_H \stackrel{?}{=} (1-H^2)^2\) and
   \(1-\mathrm{TVD} \stackrel{?}{=} \mathrm{tvd\_fidelity}\).
   Record floating-point tolerance.
3. Compute Spearman \(\rho\) of \(H\) vs \(F_H\) and of
   \(\mathrm{TVD}\) vs \(1-\mathrm{TVD}\) globally and per
   `algorithm`.
4. Pass if \(\rho\le -0.999\) aside from documented ties or
   rounding.

Fenton Ch. 6: this is a rank association, legal on ordinal
similarities.

## Test B procedure

Do this separately for Class H and Class TV.

1. For each grouping key below, compute
   - \(\overline{d}\) = mean of the distance,
   - \(\overline{s}\) = mean of the similarity,
   - \(\tilde{d}\) = median of the distance,
   - \(\tilde{s}\) = median of the similarity.
2. Grouping keys, in this order:
   - `provider` (IBM vs Rigetti), all algorithms pooled;
   - `provider` × `algorithm`;
   - `provider` × `algorithm` × `num_qubits` (only groups with
     \(n\ge 5\) per side);
   - `backend_name` within IBM, if at least three backends have
     \(n\ge 5\).
3. For two groups \(A,B\), a **mean inversion** is:
   \(\overline{d}_A < \overline{d}_B\) (A closer) but
   \(\overline{s}_A < \overline{s}_B\) (A less similar),
   or the symmetric case.
4. A **median inversion** should be absent if H-A holds and
   the transform is strictly monotone. If a median inversion
   appears, stop and check the data.

Report, for each grouping:

| Grouping | \(n_A,n_B\) | mean distance order | mean similarity order | inversion? | median order |

### Why the mean can invert

\(F_H=(1-H^2)^2\) is strictly decreasing in \(H\) on \([0,1]\)
but nonlinear. Jensen's inequality implies
\(\mathrm{mean}(F_H) \neq F_H(\mathrm{mean}(H))\) unless \(H\)
is constant. Two providers with the same mean \(H\) can have
different mean \(F_H\) if their spreads differ; with different
means, the inequality can flip.

A minimal constructed example (for the paper text, even if the
CSV does not invert):

- Provider A jobs: \(H=(0.05,0.05,0.80)\)
- Provider B jobs: \(H=(0.30,0.30,0.30)\)

Mean \(H_A=0.30=H_B\), but mean \(F_H\) differs. Adjust one
coordinate slightly to produce a strict ranking inversion if
needed for the theoretical subsection.

### What not to treat as Test B

- IBM vs Rigetti on `success_rate` medians (already in
  `narrative_assessment.md`).
- Michelson vs CCS on teleportation (case study, different
  attributes).
- Any ranking that uses a different subset of rows for the
  distance than for the similarity.

## Test C procedure

1. Count rows with null `hellinger_fidelity` and non-null
   `success_rate`.
2. Cross-tab by `algorithm` and `num_qubits`.
3. Cite the broad-ideal timing table and the BV comment that
   `hellinger_fidelity` needs the \(2^n\) ideal.
4. Pass if those rows exist or if the broad-ideal Stage 2
   numbers are used as the existence proof. They already are.

## Decision rule for the manuscript

| Outcome | Manuscript consequence |
|---|---|
| H-A holds, H-B finds at least one mean inversion with \(n\ge 5\) per side | Section 7 can claim an empirical aggregation effect. Lead the introduction with that pair of groups. |
| H-A holds, H-B finds no inversion | Section 7 reports the negative result. Keep the theoretical aggregation warning. Use teleportation CCS vs Michelson as the empirical "choice of quantity changes the conclusion" example. |
| H-A fails | Fix the data or the transform before any other claim. |
| H-C fails | Should not happen; if it does, Test C rests only on the broad-ideal synthetic stage. |

## Fenton constraints on the write-up

- State the attribute before the ranking ("closeness to
  \(P^\star\)" for Class H / TV; not "success").
- Prefer medians for the provider-success story (ordinal-safe).
- Use means only in Test B, and say why that operation is the
  one under test (Ch. 2.4 meaningfulness).
- Name threats: reused heterogeneous jobs, optimization-level
  confounding, calibration drift, author-selected groupings
  (Ch. 4.1.4).

## Deliverable after you run it

Append a short results block to this file:

- date;
- row counts after each filter;
- inversion table;
- the decision-rule row you took;
- any grouping you looked at and discarded.

## Results, 2026-08-25

The reproducible run is implemented in
`../reproduce_analysis.py`. It reads the existing CSV and does not submit
new jobs.

- The dataset contains 1,478 rows. Both the Hellinger and TVD checks retain
  1,411 paired rows.
- The maximum error in \(F_H=(1-H^2)^2\) is
  \(1.203\times10^{-6}\), with Spearman \(\rho=-0.999999999\).
- The maximum error in \(1-\mathrm{TVD}=\mathrm{tvd\_fidelity}\) is
  \(1.110\times10^{-16}\), with Spearman \(\rho=-1\).
- The provider filter retains 682 records: 559 IBM records and 123 Rigetti
  records.

| Grouping | Comparable pairs per class | Hellinger mean / median inversions | TVD mean / median inversions |
|---|---:|---:|---:|
| provider | 1 | 0 / 0 | 0 / 0 |
| provider × algorithm | 2 | 0 / 0 | 0 / 0 |
| provider × algorithm × qubits | 7 | 0 / 0 | 0 / 0 |
| Grover configuration × qubits | 21 | 0 / 0 | 0 / 0 |
| **Total** | **31** | **0 / 0** | **0 / 0** |

Test C finds 67 rows with a defined success rate and missing full Hellinger
fidelity: 65 Bernstein–Vazirani related rows and two Grover rows. The result
satisfies H-A and H-C but not H-B. The manuscript therefore takes the
negative result row in the decision table: it retains the analytical
aggregation counterexample and does not claim that the corpus exhibits an
aggregation inversion. The earlier optional comparison among three or more
IBM backends was discarded because it does not test the prespecified IBM
versus Rigetti ordering used by the manuscript.
