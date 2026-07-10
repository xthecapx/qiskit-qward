# DSR Profile Narrative Assessment

Gate check required by the revision plan before drafting `main-dsr-profile.tex`:
inspect the recomputed profile and confirm/deny whether the empirical story
changes relative to the original Michelson-only, full-HF/TVD paper.

Data: `DSR_result.csv` rebuilt via `enrich_dsr_profile.py --dataset all` +
`build_csv_from_json.py` (1283 rows: 248 Grover, 239 QFT, 796 Teleportation).
Stats: `statistical_comparison_profile.py` (Mann-Whitney U / Wilcoxon +
Cliff's delta, matched by algorithm and qubit count).

## 1. Grover / QFT: the "Rigetti underperforms IBM" narrative is CONFIRMED, not inverted

At every matched `(algorithm, num_qubits)` group with `n >= 3` per side, IBM's
median is higher than Rigetti's (Ankaa-3 / Forte-1) on **all four** profile
components (`success_rate`, `chance_corrected_success`,
`coarse_tvd_similarity`, `coarse_hellinger_fidelity`), with large effect
sizes and Mann-Whitney `p < 0.005` in every group except one borderline QFT
5-qubit case (`n=4` per side, underpowered):

| Algorithm | q | IBM median (success) | Rigetti median (success) | Cliff's δ | p |
|---|---|---|---|---|---|
| GROVER | 2 | 0.969 | 0.927 | +0.88 | <0.0001 |
| GROVER | 3 | 0.825 | 0.170 | +1.00 | <0.0001 |
| GROVER | 4 | 0.514 | 0.113 | +1.00 | 0.0016 |
| QFT | 2 | 0.965 | 0.845 | +1.00 | <0.0001 |
| QFT | 3 | 0.931 | 0.818 | +0.99 | <0.0001 |
| QFT | 4 | 0.892 | 0.387 | +1.00 | <0.0001 |
| QFT | 5 | 0.701 | 0.039 | +1.00 | 0.0011 |

**Decision: keep the qualitative Rigetti-vs-IBM narrative.** The exact
numbers backing it must change (the old paper's ~0.25 "background overlap"
full-HF/TVD framing is replaced by the profile's success /
chance-corrected-success / coarse-similarity numbers above), and the
"first to signal failure" framing should be softened to "each profile
component signals the same failure; chance-corrected success and the coarse
components are the ones defensible without a simulated ideal histogram."

## 2. Teleportation: a genuinely NEW finding — IBM underperforms Rigetti for larger payloads

Opposite direction from Grover/QFT. For payload sizes 2-4, IBM's median
`success_rate` / `chance_corrected_success` is **lower** than Rigetti's,
significant at payload 3 and 4 (`p < 0.0001`, `δ ≈ -0.8`):

| Payload | IBM median (success) | Rigetti median (success) | δ (IBM vs Rigetti) | p |
|---|---|---|---|---|
| 1 | 0.615 | 0.700 | -0.50 | 0.0041 |
| 2 | 0.251 | 0.400 | -0.34 | 0.0522 (n.s.) |
| 3 | 0.132 | 0.400 | -0.82 | <0.0001 |
| 4 | 0.066 | 0.200 | -0.76 | <0.0001 |

Critically, **Michelson DSR mostly collapses to ~0 for both providers**
(payload 2 and 4 medians are `0.000` on both sides) even though
`success_rate` / `chance_corrected_success` clearly separate them. This is
the T1-bias failure mode documented in the revision plan: teleportation's
"other" mass concentrates on the all-zero attractor state under amplitude
damping, so the strongest *competing* peak (`p_comp`, what Michelson
contrasts against) is close to the expected-success mass itself once
teleportation degrades — Michelson saturates near 0 while chance-corrected
success still tracks the real, provider-dependent gap. This is exactly the
motivating example for keeping Michelson as an optional fifth layer rather
than the headline number, and for the abstract/introduction's motivation
section.

**Decision: report this as a new result**, not previously visible in the
Michelson-only analysis, and use it (rather than Rigetti Grover/QFT) as the
primary illustration of the T1-bias motivation for chance-corrected success.

## 3. The "K=1 redundancy" claim needs a scope correction

The plan's contract states coarse TVD similarity and coarse Hellinger
fidelity collapse exactly to `success_rate` when `K=1`. This is TRUE by
construction *within the coarse computation itself* (proved analytically:
at `K=1`, `coarse_tvd = 1 - success_rate` exactly). It does **not**,
however, mean the coarse profile always matches the *full*-distribution
HF/TVD computed from a statevector-simulated ideal (`enrich_hellinger.py`),
because that comparison implicitly assumes the true ideal probability is a
delta on `E`.

`2_full_vs_coarse_comparison.png` shows this directly: QFT round-trip points
lie exactly on `y=x` (its ideal *is* an exact delta — round-trip QFT is a
deterministic unitary), but Grover points deviate by up to `0.23`
(Hellinger) / `0.05` (TVD) at high success rates, because Grover's true
statevector ideal at a finite iteration count is only *approximately* a
delta on the marked state(s) (residual amplitude leaks to unmarked states;
`theoretical_success` in the Grover configs is `<1`, e.g. `0.9995` for
`S10-1`).

**Decision: correct the manuscript wording.** State the `K=1` collapse as
an exact property of the coarse profile's own construction (idealized
"succeed on E, fail otherwise" target), and separately state that it
coincides with the full-distribution metric only for algorithms whose true
ideal is *exactly* a delta on `E` (QFT round-trip, teleportation) — not for
finite-iteration Grover, where the two are deliberately different
questions: "did we get the right answer" (coarse) vs. "did we reproduce the
exact ideal quantum state" (full).

## Net conclusion for the manuscript delta

- Headline empirical story (QPU noise degrades success in a
  provider-dependent way, worse on Rigetti's Ankaa-3/Forte-1 for
  Grover/QFT) **survives recomputation** — write the delta with confidence,
  but replace every specific number and drop the "background overlap ≈0.25"
  full-HF/TVD framing.
- Add the teleportation IBM-vs-Rigetti reversal as new content; it is a
  better motivating example for chance-corrected success than anything in
  the original paper.
- Narrow the `K=1` redundancy claim as described above.
