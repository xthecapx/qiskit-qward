# Plan: Unified IBM pipeline and `ibm_fez` training dataset

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.
Update the **Progress log** at the bottom whenever a phase moves.

## Goal

Build a fresh, consistent training dataset on **`ibm_fez`** for the thesis:
every row is one QPU job whose pre-runtime features and post-runtime targets
were computed by the same code, from the exact circuit that ran and the chip
state when it ran.

- **Algorithms (thesis scope):** Fourier-based (QFT, PE), oracle-based
  (Grover, BV), entanglement-based (vTP, GHZ).
- **Additional:** coin toss (noise floor), random volumetric (control group),
  Shor (known-failure anchor).
- **Shots:** 1,024 per job.
- **Optimization level:** 3 only (fixed `seed_transpiler`, ISA frozen at prepare).
- **Repetitions:** 2 runs per config per window, on the same frozen ISA circuit.
- **Windows:** 3 calibration windows on different days, so 6 jobs per config.
- **Legacy data:** existing runs are kept untouched as a separate legacy set.
  They are not recovered, rewritten or mixed with the new dataset.
- **New data location:** `data/qpu/v2/` (schema-versioned).
- Rigetti and IonQ come later through the same plan format.

## Feature and target groups recorded per job

| Group | Contents | Computed at |
|---|---|---|
| A. Logical circuit | QWARD pre-runtime metrics (Qiskit, Complexity, Structural, Behavioral, Element, QuantumSpecific*) | prepare |
| B. Transpiled circuit | same metrics on the ISA circuit; layout, physical qubits | prepare |
| C. Hardware at submission | per-qubit/per-gate errors for used qubits, directional readout, T1/T2, idle time, device medians, IBM calibration timestamp | prepare |
| D. Hardware at execution | same as C from `job.properties()` | recover |
| E. Baseline predictions | M1 / M1-mixture / M2 error estimates (features, not targets) | prepare |
| F. Targets | one shared `FidelityMetrics` pass against the ideal distribution (success, DSR, Hellinger, TVD) + `HistogramAssessment` | recover |
| G. Provenance | run/job/batch/campaign IDs, window, code + package versions, transpiler seed, Runtime options, timestamps | throughout |

\* QuantumSpecificMetrics measured cost on the logical circuit: 1.3 s (3 q),
6 s (6 q), 74 s (8 q), 462 s (9 q). Compute once per config up to 8 qubits;
above that, record as missing with reason `size_limit` (not zero).

vTP S-configs are dynamic (`if_else`); M1/M2 reject dynamic circuits, so group
E is recorded as missing for them. All other groups are computed.

## Findings that motivate the redesign

Old pipeline (`ibm_experiment_base.py` + `executor.run_ibm`):
1. Files are written only if the run completes (`ibm_experiment_base.py:356`);
   timeouts lose job/batch IDs (hence hard-coded `get_timed_out_batches`).
2. Gate-error characterization re-transpiles without a seed
   (`ibm_experiment_base.py:325`); on FakeFez, 4 transpiles gave 2–4 distinct
   layouts, so it may not describe the circuit that ran.
3. Calibration is device medians from `backend.target`, captured before
   submission, timestamped `now()`; no per-qubit data.
4. QWARD metrics are computed on the logical circuit only.
5. Post-run targets differ per runner (`evaluate_result` in each of 11
   runners; none uses `FidelityMetrics`); Hellinger/TVD/DSR are added later
   by enrichment scripts.
6. `--recover` assigns optimization level by job order; update script covers
   only Grover/QFT; campaign state is tracked in shell-script comments.
7. `docs/thesis-dataset-plan.md` states per-job historical calibration is
   unavailable; `job.properties()` worked for all 10 `bv-ibm-fez` jobs.

Calibration campaign (`bv_calibration_ibm.py`) has the right lifecycle
(freeze → submit → idempotent recover → run-time calibration) but bypasses
the base class and has bugs (see Phase 1).

## Circuit matrix (within hardware limits)

Measured on FakeFez (opt 2, seed 11; opt 2 and opt 3 gave the same 2Q count
on 23 of 25 sampled configs, median ratio 1.0). Predicted error is M1; real
`ibm_fez` error was 1.5–3× higher in BV runs. Keep circuits between "almost
always succeeds" and "pure noise".

| Algorithm | Keep | Drop / anchor | Configs (≈) | Jobs (×2 runs ×3 windows) |
|---|---|---|---|---|
| GHZ | GHZ2–GHZ16 (2Q 3→15) | — | 15 | 90 |
| BV | ONES/ALT/SINGLE × BV2–BV14 | — | 39 | 234 |
| vTP | V1–V5 static, S1–S5 dynamic × 3 payloads | — | 30 | 180 |
| QFT | SR2–SR8, SP/PV ≤ 9 qubits, IV4 | SR10–14, SP10/12 | 28 | 168 |
| PE | T/S/Z families, 2–8 counting qubits | 9–10 counting | 21 | 126 |
| Grover | S2–S4, M3/M4, H3/H4, SYM/ASYM | S6+ dropped; S5 once as anchor | 23 | 138 |
| Coin toss | all CT configs (noise floor) | — | 11 | 66 |
| Random volumetric | RV2–RV5 × 3 depths; RV6-D6, RV7-D7, RV8-D8 | deeper RV (80–100% predicted) | 15 | 90 |
| Shor | SHOR-N15-M3 (anchor) | other N15 configs | 1 | 6 |
| **Total** | | | **~183** | **~1,100** |

Estimated QPU time: 55–80 min total, 18–27 min per window. Exact counts and
seconds are finalized in the plan files (Phase 6) before approval.

## Phases

### Phase 1 — Fix calibration-campaign bugs (no QPU) ✅
- [x] Rename manifest recovery to `recover_manifest`; base
      `recover_batch(batch_id, ...)` is no longer shadowed.
- [x] Align alpha: `prepare` and analysis share `CLAIMS_PER_RUN` (7);
      records store `claims_per_run`. Archived records keep their frozen alpha.
- [x] `prepare_batch`: manifest existence, `runs`, run budget and config IDs
      validated before any record is written.
- [x] `submit_batch`: per-(campaign, window) limits; one `_submitted_runs`
      helper replaces the duplicated scan.
- [x] One `capture_calibration` per candidate in `rank_layouts`; one
      calibration read per Batch submission.
- [x] Tests for each fix (499 passing).

### Phase 2 — Shared lifecycle `qward/examples/papers/ibm_campaign.py` (no QPU)
- [ ] Record/journal with hashes, atomic writes, JSON encoder.
- [ ] Prepare: transpile once with fixed seed, store logical + ISA QPY,
      calibration snapshot, config, ideal distribution; status `prepared`.
- [ ] Submit: job mode for single runs, one Batch per plan; IDs saved
      immediately; Runtime-options hook (replaces Shor monkey-patch).
- [ ] Recover by saved job ID, idempotent, never resubmits; run-time
      calibration via `job.properties()`; refuse never-submitted records.
- [ ] Plans/manifests with windows and run budgets; `--status`.
- [ ] Compact CLI summary; `--full-json`.
- [ ] `IBMExperimentBase.run()` keeps current CLI (`--config --backend
      --opt-levels --runs`), now prepare → submit → wait; `--no-wait`.
- [ ] Old `--update`/`--recover` kept as read-only legacy importer.

### Phase 3 — Shared feature/target extractor (no QPU)
- [ ] Groups A–G from one function, for any algorithm.
- [ ] Runner contract: configs, circuit, ideal distribution (+ optional
      hooks: Runtime options, prediction, extra assessments).
- [ ] Replace per-runner `evaluate_result` targets with one `FidelityMetrics`
      pass; keep algorithm-specific extras additive.
- [ ] QuantumSpecific size limit with explicit missing reason.
- [ ] Dynamic-circuit handling for group E.

### Phase 4 — Port runners (no QPU)
For each: FakeFez dry run + test that every A–G field is present.
- [ ] GHZ
- [ ] BV
- [ ] vTP (static + dynamic)
- [ ] QFT
- [ ] PE
- [ ] Grover
- [ ] Coin toss (ideal biased distribution)
- [ ] Random volumetric
- [ ] Shor (Runtime-options hook)
- [ ] BV calibration campaign on the shared module; existing
      `bv-ibm-fez-1/2/3` records still load and recover.

### Phase 5 — Exporter (no QPU)
- [ ] One row per job, all groups as columns.
- [ ] Export schema document.
- [ ] Missing-field check that fails loudly.
- [ ] Output under `data/qpu/v2/`; legacy CSV pipeline unchanged.
- [ ] Single driver `run_campaign.sh` (`uv run -m`, `.env`, no tokens on
      the command line).

### Phase 6 — Campaign plan files (no QPU)
- [ ] One plan per algorithm × window (9 algorithms × 3 windows = 27 files).
      Each plan is submitted as **one Batch containing every config of that
      algorithm** (2 runs each). A Batch cannot span days, so each window
      gets its own Batch.
- [ ] Job order inside a Batch: shuffle the configs (seeded, saved in the
      manifest), then submit run 1 of all configs, then run 2 of all configs.
      Size is not then tied to submission time, and the two runs of a config
      are spread across the Batch's duration.
- [ ] Exact job counts and QPU-second estimates per Batch.
- [ ] **User approval.**

### Phase 7 — Pilot (QPU)
- [ ] 1 small + 1 mid config per algorithm, opt 3, 2 runs each; Shor anchor
      once (~34 jobs, ~2 min).
- [ ] Verify export end to end; fix before scaling up.

### Phase 8 — Full campaign (QPU)
- [ ] Window 1, all algorithms.
- [ ] Window 2 (different day).
- [ ] Window 3 (different day).
- [ ] Final export and data description.

## Related experiment: BV calibration prediction

Separate research question (does calibration predict histogram error?).
Details and findings: `qward/examples/papers/bv/CALIBRATION_EXPERIMENT.md`.

- [x] `bv-ibm-fez-1` (2 jobs) and `bv-ibm-fez-2` (8 jobs) completed.
- [~] `bv-ibm-fez-3` window 1 submitted (Batch `7256acf4-41a9-4d4d-9932-c22e444a34df`),
      not yet recovered.
- [ ] `bv-ibm-fez-3` windows 2 and 3.

## Decisions

| Date | Decision |
|---|---|
| 2026-10-29 | Re-run all campaigns on `ibm_fez` for a fresh dataset; legacy data kept separate. |
| 2026-10-29 | 1,024 shots; 3 windows. |
| 2026-10-29 | Optimization level 3 only. Opt 2 ≈ opt 3 (median 2Q ratio 1.0); opt 0 has ~1.9× more 2Q gates and mostly pushes circuits into noise. The level is constant, so the model learns from transpiled-circuit features (group B). |
| 2026-10-29 | 2 runs per config per window (6 per config) to measure within- and between-window variation. |
| 2026-10-29 | One Batch per algorithm per window, containing all of that algorithm's configs; seeded shuffle, run 1 of every config before run 2. |
| 2026-10-29 | Thesis scope: QFT, PE, Grover, BV, vTP, GHZ; also port coin toss, random volumetric, Shor. |
| 2026-10-29 | Compute QuantumSpecificMetrics up to 8 qubits; mark missing above. |
| 2026-10-29 | Circuits bounded to the informative error range (matrix above). |

## Progress log

| Date | Update |
|---|---|
| 2026-10-29 | Plan written. Calibration feature committed (`5ebae5d`). Scaling simulator raw data removed (regenerable; README added). |
| 2026-10-29 | Switched to opt 3 only with 2 runs per config per window (~1,100 jobs). |
| 2026-10-29 | Batches organized per algorithm (one per algorithm per window). |
| 2026-10-29 | Phase 1 done: calibration-campaign bug fixes. |
