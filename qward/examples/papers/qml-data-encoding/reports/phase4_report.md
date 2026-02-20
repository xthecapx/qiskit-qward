# Phase 4 Report: Implementation (TDD Green Phase)

## Summary

Phase 4 implemented the complete QML data encoding library to satisfy all 179 tests from Phase 3. This is the TDD "green" phase -- every test that previously failed with `ModuleNotFoundError` (111 tests) now passes, and all 68 previously-passing tests continue to pass. The implementation covers five quantum encoding methods, three metric computations, and an experiment pipeline, all built on Qiskit 2.1.2 and Qiskit Aer 0.17.1.

## Token Usage

| Metric | Count |
|--------|-------|
| Input tokens | ~250,000 |
| Output tokens | ~45,000 |
| Total tokens | ~295,000 |
| Estimated cost | ~$7.13 (Opus: $3.75 input + $3.38 output) |
| Agents spawned | 0 |
| Agent sessions | python-architect (sole agent) |

**Note**: Token counts are estimates. The architect read all Phase 2 and Phase 3 deliverables, all 9 test files, conftest.py, and the existing eigen-solver implementation patterns before producing the implementation.

## Key Results

### Test Results

```
179 passed in 21.47s
```

All 179 tests pass, including:
- 15 basis encoding tests
- 17 amplitude encoding tests
- 25 angle encoding tests
- 22 IQP encoding tests
- 19 re-uploading encoding tests
- 22 expressibility and statistical tests
- 24 pipeline integration tests
- 24 preprocessing tests
- 11 classical baseline tests

### Package Structure

```
qml-data-encoding/
  src/
    qml_data_encoding/
      __init__.py
      encodings/
        __init__.py
        base.py          # Abstract base class with kernel, MW, statevector helpers
        basis.py          # BasisEncoding: X gates for binary vectors
        amplitude.py      # AmplitudeEncoding: StatePreparation with decomposition
        angle.py          # AngleEncoding: Ry/Rx/Rz rotations per qubit
        iqp.py            # IQPEncoding: H-D(x)-H with ZZ interactions
        reuploading.py    # ReuploadingEncoding: L layers of [CX, S(x), W(theta)]
      metrics/
        __init__.py
        expressibility.py  # KL divergence from Haar, fidelity distributions
        entanglement.py    # Meyer-Wallach via reduced density matrices
        kernel.py          # Kernel-target alignment with rescaled kernel
      pipeline/
        __init__.py
        experiment.py      # ExperimentPipeline: preprocessing + encoding
        config.py          # ExperimentResult dataclass
```

### Implementation Decisions

#### 1. Base Class Architecture

All five encodings extend `BaseEncoding` (ABC), which provides:
- `_statevector(x, theta)` -- statevector simulation via AerSimulator
- `kernel(x, x', theta)` -- fidelity-based quantum kernel
- `kernel_matrix(X, theta)` -- Gram matrix with symmetric optimization
- `meyer_wallach(x, theta)` -- delegates to `metrics.entanglement`

This eliminates code duplication and ensures consistent kernel/MW computation across all encodings.

#### 2. Amplitude Encoding: StatePreparation + 4x Decompose

Qiskit's `initialize` instruction produces custom gates (`isometry_to_uncompute_dg`) that Aer cannot simulate directly. The solution uses `StatePreparation` from `qiskit.circuit.library` with 4 rounds of decomposition to reduce to base gates (`u`, `cx`) that Aer handles natively.

#### 3. Angle Encoding: Reversed Qubit Mapping

The TDD tests verify the angle encoding statevector against a hand-built tensor product `kron(s_0, s_1, ..., s_{n-1})`. Qiskit's statevector uses reversed ordering `kron(s_{n-1}, ..., s_0)`. To match, the angle encoding maps feature `x[i]` to qubit `n-1-i`. This is transparent to kernel and entanglement computations (which are ordering-invariant) and correctly produces the expected statevector.

#### 4. IQP Encoding: Decomposed RZZ Gates

IQP uses `RZZ(2*x_i*x_j)` for pairwise interactions, decomposed into `CX-Rz-CX`. This produces exactly `d*(d-1)` CX gates for full connectivity, matching the Phase 2 resource analysis.

#### 5. Re-uploading: CX Chain at Layer Start

The re-uploading circuit places the entangling CX chain at the START of each layer (before data encoding and trainable rotations):
```
Layer l: CX_chain -> S(x) -> W(theta_l)
```
This ensures:
- L=1 with theta=0 produces product states (CX on |0...0> is identity) -- MW = 0
- L>=2 creates entanglement (CX acts on non-trivial states from previous layer)
- Gate count: L*(d-1) CX gates, 2*L*d Ry gates

#### 6. Kernel-Target Alignment: Rescaled Kernel Formula

The KTA implementation uses `K_r = 2*K - 1` to rescale kernel entries from [0,1] to [-1,1] before computing the Frobenius alignment with `Y = outer(y, y)`. This ensures:
- Perfect kernel (same-class=1, different-class=0) gives alignment = 1.0
- Identity kernel gives alignment = 1/sqrt(n)

#### 7. Expressibility: Full 2pi Parameter Range

The fidelity distribution sampling uses `Uniform[0, 2*pi]` per feature (not `[0, pi]`) to match the theoretical prediction `E[F] = (1/2)^d` for angle encoding. This comes from the requirement that `cos^2(z/2)` has expected value 0.5 when z is uniform on `[-pi, pi]` (which requires x, x' uniform on `[0, 2*pi]`).

#### 8. Pipeline: Separated Preprocessing and Encoding

`ExperimentPipeline.build_circuit(x)` accepts pre-processed data directly (does not re-apply preprocessing). The caller is responsible for calling `fit_preprocessing()` and `transform()` first. This avoids double-application of PCA and handles the case where preprocessing is not yet fitted.

### Conftest.py Modification

Added `src/` directory to `sys.path` in `conftest.py` so that `from qml_data_encoding import ...` works:
```python
_SRC_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
```

## Figures & Tables

No figures generated. Key tables in this report:
- Test results (above)
- Package structure (above)

## Handoff Notes

### For Phase 5 (Data Scientist)

The implementation is complete and all 179 tests pass. Key usage patterns:

```python
# Encodings
from qml_data_encoding.encodings import (
    BasisEncoding, AmplitudeEncoding, AngleEncoding,
    IQPEncoding, ReuploadingEncoding,
)

# Create encoding
enc = AngleEncoding(n_features=4, rotation_axis="y")
circuit = enc.encode(x)
kernel = enc.kernel(x, x_prime)
K = enc.kernel_matrix(X)
mw = enc.meyer_wallach(x)

# Metrics
from qml_data_encoding.metrics import (
    compute_expressibility, compute_fidelity_distribution,
    meyer_wallach_from_statevector, kernel_target_alignment,
)

# Pipeline
from qml_data_encoding.pipeline import ExperimentPipeline, ExperimentResult
pipeline = ExperimentPipeline(
    encoding="angle_ry", preprocessing="minmax_pi",
    n_qubits=4, ansatz_reps=2,
)
pipeline.fit_preprocessing(X_train)
X_proc = pipeline.transform(X_test)
circuit = pipeline.build_circuit(X_proc[0])
```

### For Phase 6 (Review)

All Phase 2 theoretical predictions are verified by the tests:
- Angle kernel factorizability: CONFIRMED
- Rz trivial kernel: CONFIRMED
- IQP entanglement creation: CONFIRMED
- Angle product states: CONFIRMED
- Re-uploading L=1 matches angle: CONFIRMED
- Amplitude scale invariance: CONFIRMED
- IQP kernel non-factorizability: CONFIRMED

---

*Phase 4 completed by: python-architect*
*Date: 2026-02-19*
*Status: Complete -- 179/179 tests pass*
