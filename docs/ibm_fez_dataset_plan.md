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

**Reference vs success set.** Each record stores two things:
- `reference`: the ideal noiseless output distribution of the logical
  circuit. It is exact (statevector) for static circuits up to 20 qubits;
  dynamic circuits use noiseless Aer sampling (200k shots, `reference_exact:
  false`). Hellinger and TVD are computed against it.
- `success_set`: the labels accepted by the runner's `create_success_criteria`
  (for example PE accepts ±1 around the phase peak). `success_rate` and DSR
  use it. Baselines report `*_error` = predicted 1 − P(success set) and
  `*_tvd` = predicted TVD to the ideal, so each is comparable with its target.

**Where the groups live in a record:**
- A, B, C, E: `artifacts/<stem>/prepared.json` → `payload.features`
- D: journal → `submission_journal[i].hardware_at_execution` and
  `baselines_at_execution`
- F: journal → `individual_results[i].targets`; the legacy `evaluate_result`
  fields are kept alongside
- G: journal and `prepared.json` top-level keys

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

Final matrix (source of truth: `qward/examples/papers/campaign_matrix.py`;
`--report` re-measures sizes locally). Sizes are the ISA at optimization level
3, seed 11, on FakeFez calibration; "max M1 error" is the largest predicted
error in the algorithm's set (real `ibm_fez` ran 1.5–3× higher in BV).

| Algorithm | Configs kept | Dropped / anchor | Configs | Jobs/window | Jobs (3 windows) | Max 2Q | Max M1 err |
|---|---|---|---:|---:|---:|---:|---:|
| GHZ | GHZ2–GHZ16 | GHZ17–20 | 15 | 30 | 90 | 15 | 0.15 |
| BV | ONES/ALT/SINGLE × BV2–BV14 | BV15+ | 39 | 78 | 234 | 38 | 0.21 |
| vTP | V1–V5 static, S1–S5 dynamic × 3 payloads | — | 30 | 60 | 180 | 36 | 0.14 (15 dynamic) |
| QFT | SR2–SR8; SP3–SP8 (≤ 9 q); PV4/PV6; IV4 | SR10–14, SP10/12 | 27 | 54 | 162 | 208 | 0.53 |
| PE | T/S/Z × 2–8 counting qubits | 9–10 counting | 21 | 42 | 126 | 112 | 0.35 |
| Grover | S2–S5, M3/M4, H3/H4, SYM/ASYM | S6+; **S5 = failure anchor** | 24 | 48 | 144 | 530 | 0.82 |
| Random volumetric | RV2–RV5 × 3 depths; RV6-D6, RV7-D7, RV8-D8 | RV9+ | 15 | 30 | 90 | 330 | 0.71 |
| Shor | SHOR-N15-M3 | other N15 configs; **M3 = failure anchor** | 1 | 2 | 6 | 1388 | 0.99 |
| **Total** | | | **172** | **344** | **1,032** | | |

Coin toss is not part of the campaign: GHZ and random volumetric already cover
the noise floor. Its runner stays available, and the 4 pilot jobs already run
(CT3, CT3-B25) stay on disk but are filtered out of the export.

Distribution of predicted M1 error across the 172 configs: 75 below 10%, 59 at
10–30%, 20 at 30–60%, 2 at 60–90%, 1 above 90% (Shor), 15 dynamic (no
prediction). The three high-error entries (Grover S5, Shor M3, RV8-D8) are
deliberate anchors for where output turns into noise.

**QPU time.** Measured on the 34 pilot jobs (IBM `usage.quantum_seconds`,
whole seconds, 1,024 shots): 32 jobs billed 2 s (ISA depth 4–438, circuit
duration 0–14 µs), the two Shor jobs billed 3 s (depth 4,640, 136 µs). A
Batch's usage in the dashboard is the sum of its jobs (4 jobs = 8 s, Shor's
2 jobs = 6 s); the pilot used 62 s in total. The 250 µs rep delay dominates
short circuits, so billing is almost flat in circuit size.
`campaign_matrix.qpu_seconds` is fitted to those numbers (31 ns per layer plus
1.7 s fixed per job) and predicts **~12 min per window, ~35 min for all 1,032
jobs** (≈ 2,100 s of the account's 10,800 s). The earlier "30 min" figure was
this same total; the dashboard figure per Batch is 6–8 s because each Batch
holds 2–4 jobs, not a whole window. `run_campaign.sh usage [window]` reads the
real numbers from IBM next to the saved per-job values.

**Dashboard tags.** Every job carries `<algorithm>` first (`ghz`, `bv` →
`bernstein-vazirani`, `phase-estimation`, …), then `cfg:<config>`,
`campaign:<id>`, `w<window>`, `repeat-<n>` and the run ID. In the IBM
dashboard, filter the Jobs table by the algorithm tag. Jobs submitted before
this change carry only `<run id>` and `repeat-<n>`.

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

### Phase 2 — Shared lifecycle `qward/examples/papers/ibm_campaign.py` (no QPU) ✅
- [x] Record/journal with hashes, atomic UTF-8 writes, JSON encoder
      (`RecordStore`, schema 2.0).
- [x] Prepare: transpile once with fixed seed (default opt 3, seed 11), store
      logical + ISA QPY, used-qubit calibration snapshot (device-wide record
      optional), config, reference, payload; status `prepared`. ALAP for static
      circuits; dynamic circuits (vTP S) transpiled unscheduled. Faulty
      qubits/edges rejected before archiving.
- [x] Submit: job mode for single records; one Batch per plan with seeded
      shuffle, run 1 of every config before run 2; IDs journaled immediately;
      `submission_unknown` on send failure, never retried.
- [x] Runtime-options hook `get_runtime_options(config)` merged onto defaults
      (replaces Shor monkey-patch once Shor is ported).
- [x] Recover by saved job ID, idempotent, never resubmits; skips completed
      jobs; refuses never-submitted records; `after_job` hook for run-time
      calibration (wired in Phase 3).
- [x] Multi-register counts joined like Aer (`"last first"`, space separated),
      verified against Aer.
- [x] Plans/manifests with windows and run budgets; offline `--plan-status`.
- [x] Compact CLI summaries.
- [x] Base CLI: `--prepare-plan/--submit-plan/--recover-plan/--plan-status`,
      `--prepare-only` (with `--config --backend`), `--submit-record`,
      `--recover-record`. New records go to `<algo>/data/qpu/v2/`.
- [x] Legacy `run()` and `--update`/`--recover` unchanged for existing data.
- [x] Dynamic-circuit-safe `circuit_hash(allow_dynamic=True)`; static hashes
      unchanged, so archived snapshots still verify.
- [x] Tests: `tests/test_ibm_campaign.py` (9) with FakeFez + mocked Runtime;
      local Aer dry run of a 3-config GHZ plan end to end. 508 passing.
- Note: the BV calibration campaign still uses its own lifecycle; it moves
  onto `ibm_campaign` in Phase 4 (records stay readable).

### Phase 3 — Shared feature/target extractor (no QPU) ✅
- [x] `qward/examples/papers/dataset_features.py`: groups A–F as functions;
      G comes from the record. A group that cannot be computed is
      `{"available": false, "reason": ...}`, never zeros.
- [x] Group B on the executed circuit: `compact_isa` keeps active qubits in
      physical order and drops delays (QWARD metrics fail on the 156-qubit
      ISA with a `TranspileLayout`); gate depth and 2Q count preserved.
- [x] Runner contract in `IBMExperimentBase`: configs, circuit,
      `create_success_criteria`; `get_reference` defaults to the exact ideal
      distribution; optional `get_runtime_options`, `include_quantum_specific`.
- [x] One shared `FidelityMetrics` + `HistogramAssessment` pass for targets;
      legacy `evaluate_result` fields kept additively. Multi-register labels
      flattened (FidelityMetrics returned 0 success with spaced labels).
- [x] QuantumSpecific ≤ 8 qubits and static only; otherwise explicit reason.
- [x] Group E missing for dynamic circuits; `*_error` vs success set and
      `*_tvd` vs ideal; works for multi-target references.
- [x] Group D at recovery from `job.properties()` (calibration in effect when
      the job ran) plus baselines re-predicted from it.
- [x] Removed the Halstead `difficulty ≤ 1000` / `effort ≤ 1e8` caps in
      `StructuralMetricsSchema`; transpiled Shor exceeded them.
- [x] Tests: `tests/test_dataset_features.py` (13); local Aer dry run of
      all 9 algorithms — every group present (vTP-S: QS and E missing by
      design). 521 passing.

### Phase 4 — Port runners (no QPU) ✅
All runners inherit the lifecycle from `IBMExperimentBase`; each gets a
FakeFez + noisy-Aer test (`tests/test_runner_ports.py`, 22 tests) that checks
groups A–G are present.
- [x] GHZ, BV, QFT, PE, Grover, random volumetric, coin toss (biased
      distribution is the exact reference, not the success set)
- [x] vTP static (V) and dynamic (S): dynamic is transpiled unscheduled;
      group E is missing by design; control-flow ops (`if_else`) have no IBM
      error and are counted separately (`uncalibrated_control_flow_ops`), not
      treated as missing calibration.
- [x] Shor: `get_runtime_options` returns DD (XpXm) + gate twirling, recorded
      in the frozen record; the lifecycle enforces the same QPU-safe guard as
      the legacy `run()` (simulator-only configs are refused). The executor
      monkey-patch stays only on the legacy path.
- [x] BV calibration campaign: shares `write_json`/`json_value`/`now` with
      `ibm_campaign`; its own lifecycle (prediction-specific records, M1/M2
      assessment) stays for now and keeps reading `bv-ibm-fez-1/2/3`.
      Fully merging it into `RecordStore` is deferred: it adds nothing to the
      dataset and risks the archived records.
- [x] BV signal-background (27–29 qubits): no tractable ideal (more than 12
      measured bits), so `get_reference` returns `None`; targets and baselines
      are `{"available": false, "reason": "no_reference"}`. Not part of the
      ibm_fez matrix.
- Output: `ideal_reference` refuses circuits above 12 measured bits instead
  of enumerating them.

### Phase 5 — Exporter and driver (no QPU) ✅
- [x] `qward/examples/papers/dataset_export.py`: one row per completed job,
      groups A–G flattened into prefixed columns (`meta`, `config`, `a`–`f`,
      `e_exec`, `g`); ~450 columns. Also derives `f.observed_error` and
      `f.residual_vs_<model>`, and queue/run seconds from IBM timestamps.
- [x] Output: `dataset.csv`, `schema.json` (column → group, dtype, non-null),
      `export_report.json` (skipped records, errors, warnings).
- [x] Validation fails the export (override: `--allow-incomplete`) on: no rows,
      duplicate job IDs, mixed/unexpected record schema, simulated backends
      (override: `--allow-fake`, tests only), unavailable group a/b/c/d/f, e
      missing without a documented reason, counts ≠ shots, coverage < 1.
      Warnings: undocumented QuantumSpecific gaps, calibration changed between
      prepare and execution, sampled references.
- [x] Reads only `<algo>/data/qpu/v2/`; legacy files and the legacy CSV
      pipeline are untouched. Prepared-but-unsubmitted records are skipped and
      reported.
- [x] `docs/ibm_fez_dataset_schema.md`: column groups, documented reasons for
      missing values, how to compare predictions with targets, plan-file format.
- [x] `qward/examples/papers/run_campaign.sh`: `prepare | submit | recover |
      status | export` per algorithm and window. Credentials come from `.env`
      or the environment (nothing on the command line); `submit` asks for
      confirmation and is the only action that sends jobs; `status` is offline.
- [x] Tests: `tests/test_dataset_export.py` (10); 553 passing overall.

### Phase 6 — Campaign plan files (no QPU) ✅
- [x] `qward/examples/papers/campaign_matrix.py` holds the selection as data,
      validates it against every runner's config registry, and writes the
      plans: `uv run -m qward.examples.papers.campaign_matrix --write`
      (refuses to overwrite a plan that already has a manifest).
- [x] 32 plan files (8 algorithms × windows 0–3), `<algo>/plans/<algo>-w<N>.json`: campaign `fez-dataset-v2`,
      `ibm_fez`, 1,024 shots, optimization level 3, seed 11, 2 runs per config,
      distinct seeded order per plan, run budget = configs × 2.
- [x] One Batch per algorithm per window; seeded shuffle of configs, run 1 of
      every config before run 2 (`job_order` in the manifest).
- [x] Job counts and QPU estimate per window: see the table above;
      `campaign_matrix --report` reproduces them locally.
- [x] `QuantumSpecificMetrics` seeded (`torch` seed 11, RNG state restored):
      it optimizes from random starts, so two computations of the same circuit
      differed. Features are now reproducible.
- [x] QuantumSpecific measured over the matrix: 134 configs have it (static,
      ≤ 8 qubits), **23.8 min of compute per window** (8-qubit circuits take
      60–80 s each, 863 s in total). It depends only on the logical circuit, so
      results are cached per circuit (key: circuit hash + seed + metric source
      + torch version) under `qward/examples/papers/data/quantum_specific_cache/`.
      `campaign_matrix --precompute-quantum-specific` fills it once (~24 min);
      afterwards every `prepare` is a cache hit and the three windows share
      the same values. Tests never touch the repository cache.
- [x] Tests: `tests/test_campaign_matrix.py` (18): configs exist, sizes,
      anchors and cuts, files on disk equal the generator output, every plan
      passes `prepare_plan` validation, driver paths match.
- [ ] **User approval of the matrix before any QPU use.**

### Phase 7 — Pilot (QPU) ✅ run, ✅ export clean
- [x] 1 small + 1 mid config per algorithm, opt 3, 2 runs each; Shor anchor
      once. 30 jobs for the 8 campaign algorithms (the 4 coin-toss jobs run
      before it was dropped stay on disk, excluded from the export).
- [x] Prepared, submitted one Batch at a time, recovered; all 9 manifests
      `completed`, no job errors.
- [x] Export of the real data: 30 rows, 0 errors. Warnings: 2 sampled
      references (the dynamic vTP config, by design) and 4 QuantumSpecific
      failures on PE-S6 and SP6-P4 (see Findings).
- Findings:
  - IBM advanced its calibration timestamp 2–3 times between prepare and
    execution, but the values of the qubits used did not change in any of the
    17 records. The notice is now one line per submission, and group D has
    `changed_values` (compares what the circuit depends on); the export warns
    only when values change.
  - `QuantumSpecificMetrics` gave NaN coherence on PE-S6 and SP6-P4:
    `torch.abs` backpropagates NaN through exactly-zero off-diagonals of
    structured unitaries (first at step 120 of 300). Fixed with an exact
    zero-safe absolute value; values elsewhere are unchanged.
  - IBM limits active Batches; `submit all` is sequential and resumable.
- [ ] Re-run `prepare` for PE-S6 and SP6-P4 features if the pilot data is
      kept (their `a.quantum_specific` is unavailable in those 4 rows).

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
| 2026-10-29 | Phase 2 done: shared lifecycle `ibm_campaign.py` + base CLI flags. |
| 2026-10-29 | Phase 3 done: feature/target groups A–G (`dataset_features.py`); exact ideal references. |
| 2026-10-29 | Phase 4 done: all runners on the lifecycle, Shor options hook, 543 tests passing. |
| 2026-10-29 | Phase 5 done: `dataset_export.py`, schema doc, `run_campaign.sh`; 553 tests passing. |
| 2026-10-29 | Phase 6 done; coin toss dropped (GHZ and RV cover the noise floor): 8 algorithms, 172 configs, 1,032 jobs. |
| 2026-10-29 | Phase 7 pilot run on ibm_fez: 34 jobs, export clean; calibration notice, QuantumSpecific NaN and sequential Batch fixes. |
