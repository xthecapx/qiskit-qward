# Plan: Extend QWARD for Complete Thesis Dataset Capture

## Context

The thesis (Section `\subsection{Dataset Structure}`, line 591 in `thesis.tex`) defines four feature groups that MUST be captured for every experiment:

| Group | Status | Gap |
|-------|--------|-----|
| 1. Pre-runtime metrics | 13/13 implemented | Only QiskitMetrics + ComplexityMetrics used in experiments (missing 4 strategies) |
| 2. Backend calibration | 0/6 implemented | Entirely missing — no infrastructure exists |
| 3. Post-runtime targets | 6/6 implemented | Hellinger/TVD in ad-hoc script, not core library |
| 4. Gate error characterization | 0/3 implemented | Entirely missing |

Additionally, 2 of 6 algorithms are missing (BV, GHZ), and 3 algorithms lack QPU experiment scripts (PE, BV, GHZ).

---

## Phase 1: Backend Calibration Collector (P0)

- [x] **1A. New ABC**: `qward/metrics/backend_metric_base.py`
  - New `BackendMetricCollector` ABC parallel to `MetricCalculator` — takes a backend object instead of a circuit
  - Methods: `get_metrics() -> Dict`, `is_available() -> bool`

- [x] **1B. IBM + AWS Calibration**: `qward/metrics/backend_calibration.py`
  - Uses Qiskit 2.x API (`backend.target`):
    - `backend.target.qubit_properties[q].t1` / `.t2` → median T₁, T₂
    - `backend.target[gate_name][(qubits)].error` → gate errors
    - `backend.target['measure'][(q,)].error` → readout error
    - `backend.num_qubits` filtered by non-None properties → operational qubits
  - For AWS/Braket: extract from `device.properties` dict

- [x] **1C. Schema**: `qward/schemas/backend_calibration_schema.py`
  - Fields: `median_single_qubit_gate_error`, `median_two_qubit_gate_error`, `median_readout_error`, `median_t1_us`, `median_t2_us`, `num_operational_qubits`, `backend_name`, `calibration_timestamp`, `provider`

---

## Phase 2: Gate Error Characterization (P0)

- [x] **2A. New class**: `qward/metrics/gate_error_characterization.py`
  - Takes transpiled circuit + backend
  - Extract physical qubit layout via `transpiled_circuit.layout.final_index_layout(filter_ancillas=True)`
  - For each gate in transpiled circuit, query `backend.target[gate_name][(physical_qubits)].error`
  - Return per-gate error entries + aggregate statistics (mean 1Q error, mean 2Q error, max error, weighted mean)

- [x] **2B. Schema**: `qward/schemas/gate_error_characterization_schema.py`
  - `GateErrorEntry`: gate_name, physical_qubits, error_rate, duration_ns
  - `GateErrorCharacterizationSchema`: entries, mean_single_qubit_error, mean_two_qubit_error, max_error, weighted_mean_error, num_distinct_physical_qubits, physical_qubits_used

---

## Phase 3: Fix Experiment Metrics Capture (P0)

- [x] **3A. Update `qward/examples/papers/experiment_helpers.py`**
  - `calculate_qward_metrics()` currently uses only QiskitMetrics + ComplexityMetrics
  - Add: `StructuralMetrics` (Halstead volume, etc.), `BehavioralMetrics` (critical depth, liveness, parallelism), `ElementMetrics`
  - **QuantumSpecificMetrics** (magic, coherence, sensitivity): Precompute separately via enrichment script. These are deterministic from the circuit and expensive (PyTorch). A separate `enrich_quantum_specific.py` runs after experiments complete.

- [x] **3B. Update `ibm_experiment_base.py`**
  - Before execution: get backend → `BackendCalibrationCollector(backend).get_metrics()`
  - After transpilation (per opt_level): `GateErrorCharacterization(isa_circuit, backend).get_metrics()`
  - Store both in JSON output under `"backend_calibration"` and `"gate_error_characterization"` keys

- [x] **3C. Update `aws_experiment_base.py`**
  - Same pattern adapted for AWS Braket's calibration API
  - Rigetti exposes calibration via `device.properties` in Braket SDK
  - Uses Braket `standardized` format (cross-provider) with fallback to `provider.specs` (Rigetti-specific)
  - Unwraps qiskit-braket-provider `_device` to access raw AwsDevice

- [ ] **3D. Update JSON output schema**
  ```json
  {
    "backend_calibration": {
      "median_single_qubit_gate_error": 0.00023,
      "median_two_qubit_gate_error": 0.0045,
      "median_readout_error": 0.012,
      "median_t1_us": 280.5,
      "median_t2_us": 120.3,
      "num_operational_qubits": 156,
      "backend_name": "ibm_fez",
      "calibration_timestamp": "2026-02-05T10:30:00Z",
      "provider": "ibm"
    },
    "gate_error_characterization": {
      "per_opt_level": {
        "3": { "entries": [...], "mean_single_qubit_error": ..., ... }
      }
    }
  }
  ```

---

## Phase 4: Integrate Hellinger/TVD into Core (P1)

- [ ] **4A. Extend `qward/metrics/circuit_performance.py`**
  - Add optional `ideal_distribution` parameter
  - When provided, compute: Hellinger fidelity, Hellinger distance, TVD, TVD fidelity
  - Move logic from `examples/papers/enrich_hellinger.py` into the core class

- [ ] **4B. New schema fields in `circuit_performance_schema.py`**
  - `FidelityMetricsSchema`: hellinger_fidelity, hellinger_distance, tvd, tvd_fidelity

---

## Phase 5: Missing Algorithms (P1)

- [x] **5A. Bernstein-Vazirani**: `qward/algorithms/bernstein_vazirani.py`
  - Class `BernsteinVazirani(secret_string: str)`
  - Circuit: H⊗n → Oracle (CNOT per '1' bit) → H⊗n → measure
  - Expected output: the secret string with P=1 (ideal)
  - Scaling: 2–14 qubits, various secret strings (all-ones, alternating, single-bit, random)

- [x] **5B. GHZ State**: `qward/algorithms/ghz.py`
  - Class `GHZ(num_qubits: int)`
  - Circuit: H on qubit 0, CNOT chain 0→1, 0→2, ..., 0→(n-1), measure all
  - Expected outcomes: `"0"*n` and `"1"*n` each at 50%
  - Scaling: 2–20+ qubits (shallow circuit, depth=n)

- [x] **5C. Random Volumetric Circuit (GHZ control group)**: `qward/algorithms/random_volumetric.py`
  - Class `RandomVolumetric(num_qubits: int, depth: int = None, seed: int = None, native_gates: list = None)`
  - **Purpose**: Control group for Hypothesis 4 — random-circuit-based hardware benchmarking
  - **Strategy**: Mirror circuit (Loschmidt echo) — apply random layers then their inverse → expected output = |0...0⟩
  - **Circuit structure**:
    1. For each layer: randomly pair qubits, apply random SU(4) gates (or native 2-qubit gates + random 1-qubit gates)
    2. Barrier
    3. Apply inverse of all layers in reverse
    4. Measure all → expected outcome `"0"*n`
  - **Scaling**: volumetric — increase BOTH qubit count and depth independently
  - **Native gate variant**: When `native_gates` provided (from backend), uses native 1Q+2Q gates instead of SU(4) — matches thesis: "drawn from the backend's native gate set"
  - **Reference implementation**: `Quantum_Benchmark_26/benchmark.ipynb` → `create_RND_circuit()`
  - **Relation to vTP**: vTP also uses random gates but within teleportation protocol. Random Volumetric is the pure random baseline without algorithm structure.

- [x] **5D. Register in `qward/algorithms/__init__.py`** — export BernsteinVazirani, GHZ, RandomVolumetric

---

## Phase 6: QPU Experiment Scripts (P2) — SCRIPTS ONLY, NO EXECUTION

**Order: BV first** (shallowest circuit O(1) depth, most likely to succeed at high qubit counts), then GHZ (control group), then PE, then vTP.

All scripts validated via **simulator dry-run** before any real QPU execution.

- [ ] **6A. BV experiment scripts** (FIRST)
  - `qward/examples/papers/bv/__init__.py`
  - `qward/examples/papers/bv/bv_configs.py`
  - `qward/examples/papers/bv/bv_ibm.py`
  - `qward/examples/papers/bv/bv_aws.py`
  - Validate: run smallest config on `AerSimulator`, confirm JSON output structure

- [ ] **6B. GHZ experiment scripts**
  - `qward/examples/papers/ghz/__init__.py`
  - `qward/examples/papers/ghz/ghz_configs.py`
  - `qward/examples/papers/ghz/ghz_ibm.py`
  - `qward/examples/papers/ghz/ghz_aws.py`
  - Validate: run 2-qubit GHZ on simulator, confirm correct expected outcomes

- [ ] **6C. Phase Estimation experiment scripts**
  - `qward/examples/papers/phase_estimation/__init__.py`
  - `qward/examples/papers/phase_estimation/pe_configs.py`
  - `qward/examples/papers/phase_estimation/pe_ibm.py`
  - `qward/examples/papers/phase_estimation/pe_aws.py`
  - Validate: run T-gate PE on simulator, confirm phase detection

- [ ] **6D. vTP experiment scripts** (IBM only — dynamic circuits)
  - `qward/examples/papers/teleportation/__init__.py`
  - `qward/examples/papers/teleportation/tp_configs.py`
  - `qward/examples/papers/teleportation/tp_ibm.py`
  - Validate: run 1-qubit teleportation on simulator

- [ ] **6E. Random Volumetric experiment scripts** (control group for H4)
  - `qward/examples/papers/random_volumetric/__init__.py`
  - `qward/examples/papers/random_volumetric/rv_configs.py`
  - `qward/examples/papers/random_volumetric/rv_ibm.py`
  - `qward/examples/papers/random_volumetric/rv_aws.py`
  - Configs: scale both n (2–14 qubits) and depth (n, 2n, 3n) — volumetric grid
  - Expected outcome: `"0"*n` (mirror circuit guarantees this)
  - Validate: run 3-qubit depth=3 on simulator, verify |000⟩ dominates

Each subclasses `IBMExperimentBase`/`AWSExperimentBase` following `grover_ibm.py` / `grover_aws.py` pattern.

---

## Phase 7: Historical Data Enrichment (P3)

- [ ] **7A. `enrich_calibration.py`** — Calibration reference for existing data
  - IBM API doesn't expose historical per-job calibration
  - Capture CURRENT calibration as reference, annotated as `"calibration_note": "post_hoc_reference"`
  - For future runs: calibration captured at execution time (Phase 3B)

- [x] **7B. `enrich_full_metrics.py`** — Re-compute pre-runtime metrics
  - Re-run StructuralMetrics + BehavioralMetrics + ElementMetrics on circuits reconstructed from configs
  - Pre-runtime metrics are deterministic, safe to backfill
  - Enriched 130/301 files (171 already had valid metrics)

- [ ] **7C. `enrich_quantum_specific.py`** — PyTorch metrics (magic, coherence, sensitivity)
  - Separate from main experiment runs due to cost
  - Reconstructs circuits, runs QuantumSpecificMetrics, backfills into JSON

---

## Random Volumetric Circuit Design Decision

### Thesis requirement (line 628)
> GHZ follows a volumetric growth strategy that increases both qubit count and gate depth by appending random single-qubit and two-qubit operations drawn from the backend's native gate set. The GHZ random-circuit variant serves as the control group for Hypothesis 4.

### Three candidate strategies analyzed:

| Strategy | Source | How it works | Known output? | Volumetric? |
|----------|--------|-------------|---------------|-------------|
| **Mirror SU(4)** | `Quantum_Benchmark_26/benchmark.ipynb` | Random SU(4) layers + inverse → |0⟩ⁿ | Yes (Loschmidt echo) | Yes (n × depth) |
| **vTP random gates** | `qward/algorithms/v_tp.py` | Random 1Q gates on payload, then teleport+nullify | Yes (nullification) | Partial (only 1Q gates, depth tied to protocol) |
| **Native-gate variant** | Thesis description | Random 1Q+2Q gates from backend's native set + inverse | Yes (mirror) | Yes (n × depth) |

### Recommended: **Native-gate mirror circuit** (hybrid of notebook + thesis)

- Start from GHZ base (H + CNOT chain) to establish entanglement
- Append random layers using backend native gates (not SU(4) — matches thesis "drawn from backend's native gate set")
- Apply inverse of appended layers + inverse GHZ → expected output = |0⟩ⁿ
- Scale: increase qubit count AND depth independently (volumetric grid)
- Native gates make transpiled depth predictable (no decomposition explosion from SU(4))

**Why not pure SU(4)?** Each SU(4) decomposes into ~3 CX + several 1Q gates. For a 10-qubit depth-10 circuit with SU(4), transpiled depth explodes (~15× actual depth). This makes the "circuit depth" feature noisy. Native gates keep transpiled circuit closer to logical circuit.

**Why not vTP random?** vTP uses only single-qubit gates (no random 2Q ops). Doesn't vary circuit depth independently from protocol structure. Not a true volumetric benchmark.

**Implementation**: `RandomVolumetric` class accepts optional `native_gates` list. When None, uses SU(4) (for simulator testing). When provided, generates circuits using only those gates (for QPU matching thesis description).

---

## Calibration Data Investigation Summary

| Approach | Feasibility | Notes |
|----------|-------------|-------|
| `backend.target` at execution time | **Best** | Captures live calibration. Must be done DURING the run. |
| `service.job(job_id)` for historical | **No calibration** | Returns counts/status only, not backend properties at time of execution |
| IBM REST API historical calibration | **Uncertain** | `quantum.cloud.ibm.com` shows history in UI, no documented public API |
| Current calibration as reference | **Fallback** | `service.backend(name).target` gives TODAY's calibration — imperfect but usable |

**Decision**: For new runs → capture at execution time. For historical data → annotate with current calibration as imperfect reference.

---

## Execution Order

```
Phase 1 + 2 (backend calibration + gate errors)  ─┐
Phase 3 (fix experiment_helpers)                   ├─→ Phase 6 (QPU scripts) → Phase 7 (enrichment)
Phase 5 (BV + GHZ algorithms)                     ─┘
Phase 4 (Hellinger/TVD core integration)           ─── independent
```

---

## Key Architecture Decisions

1. **Do NOT modify `MetricCalculator` ABC** — it works well for circuit-only analysis. Create parallel `BackendMetricCollector` ABC for backend-dependent metrics.
2. **Capture calibration at execution time** — modify experiment bases to capture BEFORE calling executor.
3. **Gate error characterization per optimization level** — each opt level produces different transpiled circuit with different physical qubit assignments.
4. **QuantumSpecificMetrics precomputed separately** — expensive PyTorch metrics run as enrichment after experiments.
5. **Don't break existing data** — new fields are additive. Existing JSON files remain valid.
6. **For historical data** — accept that execution-time calibration is lost. Add reference calibration with clear annotations.

---

## Verification Checklist (ALL LOCAL — NO QPU RUNS)

- [ ] Unit tests for `BackendCalibrationCollector` with mock backend
- [ ] Unit tests for `GateErrorCharacterization` with mock transpiled circuit
- [ ] Integration test: run small circuit on `AerSimulator` (has `.target`), verify calibration extraction
- [ ] Integration test: run small circuit on `GenericBackendV2` (fake backend with calibration data)
- [ ] Smoke test: run Grover 2-qubit with updated experiment base using simulator, verify JSON has all 4 feature groups
- [ ] BV algorithm unit tests (secret string recovery on simulator)
- [ ] GHZ algorithm unit tests (expected outcomes on simulator)
- [ ] End-to-end: full experiment pipeline on simulator producing complete JSON with all feature groups

**NO QPU EXECUTIONS until all simulation tests pass and output is confirmed correct.**

---

## Key Files Reference

| File | Action |
|------|--------|
| `qward/metrics/backend_metric_base.py` | CREATE — new ABC |
| `qward/metrics/backend_calibration.py` | CREATE — calibration collector |
| `qward/metrics/gate_error_characterization.py` | CREATE — per-gate errors |
| `qward/schemas/backend_calibration_schema.py` | CREATE |
| `qward/schemas/gate_error_characterization_schema.py` | CREATE |
| `qward/algorithms/bernstein_vazirani.py` | CREATE |
| `qward/algorithms/ghz.py` | CREATE |
| `qward/algorithms/random_volumetric.py` | CREATE — mirror circuit generator |
| `qward/algorithms/__init__.py` | MODIFY — add exports |
| `qward/metrics/__init__.py` | MODIFY — add exports |
| `qward/metrics/types.py` | MODIFY — new MetricsId entries |
| `qward/examples/papers/experiment_helpers.py` | MODIFY — use all strategies |
| `qward/examples/papers/ibm_experiment_base.py` | MODIFY — capture calibration |
| `qward/examples/papers/aws_experiment_base.py` | MODIFY — capture calibration |
| `qward/metrics/circuit_performance.py` | MODIFY — add Hellinger/TVD |
| `qward/examples/papers/bv/` | CREATE — BV experiment scripts |
| `qward/examples/papers/ghz/` | CREATE — GHZ experiment scripts |
| `qward/examples/papers/phase_estimation/` | CREATE — PE experiment scripts |
| `qward/examples/papers/random_volumetric/` | CREATE — H4 control group scripts |
