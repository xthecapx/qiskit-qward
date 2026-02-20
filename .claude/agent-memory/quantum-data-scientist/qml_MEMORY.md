# Quantum Data Scientist Memory

## Project: QWARD Eigen-Solver (COMPLETED 2026-02-19)

### Final Import Path (after Phase 7 integration)
```python
from qward.algorithms import QuantumEigensolver, ClassicalEigensolver
from qward.algorithms.eigensolver import pauli_decompose
```

### Key Findings (Phase 5)
- VQE achieves machine precision for 1-qubit systems (2-parameter RY+RZ ansatz)
- 2-qubit systems (EfficientSU2, 12 params) achieve ~1e-6 accuracy ideal, <3% noisy
- Noise ranking: IBM-HERON-R3 (best) < R1 ~ R2 < RIGETTI-ANKAA3 (worst)
- Noise impact correlates directly with 2-qubit gate error rates
- Deflation protocol works perfectly under ideal conditions including 3-fold degeneracy
- QuantumEigensolver.solve() uses fixed seed (rng=42) -- for statistical trials, call _run_single_vqe directly with varying seeds
- 129 total tests passing (108 sandbox + 21 integration)

### QWARD Infrastructure
- `qward.algorithms.NoiseModelGenerator` + `get_preset_noise_config()` for noise presets
- `qward.algorithms.experiment_analysis.compute_descriptive_stats()` for trial statistics
- Noise preset keys: "IBM-HERON-R1", "IBM-HERON-R2", "IBM-HERON-R3", "RIGETTI-ANKAA3"

### File Locations
- Eigen-solver example: `qward/examples/papers/eigen-solver/`
- Library integration: `qward/algorithms/eigensolver/`
- Phase 5 outputs: `results/` (CSV/JSON) and `img/` (PNG plots)

## Project: QML Data Encoding (COMPLETED 2026-02-19)

### Critical: Qiskit 2.1 Ansatz Decomposition
**RealAmplitudes MUST be decomposed before AerSimulator can run them.**
```python
# WRONG - causes "unknown instruction: RealAmplitudes"
bound_ansatz = ansatz.assign_parameters(param_dict)
# CORRECT - decompose after binding
bound_ansatz = ansatz.assign_parameters(param_dict).decompose()
```
Class `RealAmplitudes` deprecated in Qiskit 2.1; use `real_amplitudes` function.

### AmplitudeEncoding Qubit Count
AmplitudeEncoding computes n_qubits = ceil(log2(n_features)) internally.
Pass n_features to constructor. Handles L2 normalization + padding.

### Key Findings (Phase 5)
- 67 valid configs out of 160 (58% excluded by NISQ feasibility)
- Classical >> quantum at 4-qubit scale (p < 0.001)
- **Sparsity predicts best encoding (rho=0.90, p=0.002)**
- Basis encoding = classical parity on MNIST binary (0.992)
- 18.7% gap between encodings on Iris (amplitude best, IQP worst)
- PCA + MinMax is most universally applicable preprocessing

### File Locations
- Source: `qward/examples/papers/qml-data-encoding/src/qml_data_encoding/`
- Results: `qward/examples/papers/qml-data-encoding/results/`
- Figures: `qward/examples/papers/qml-data-encoding/img/`

### Analysis Patterns
- Always use `os.environ["MPLBACKEND"] = "Agg"` before matplotlib import
- Use `uv run` (not `python`) per CLAUDE.md guidelines
- Friedman test requires >= 3 treatments and >= 2 blocks
- Bonferroni correction: alpha_adj = alpha / n_comparisons
- Save figures at 300 DPI for publication quality
- VQC training: COBYLA + mini-batch (size=20) is stable
- For n-trial stats: compute mean, std, SEM, 95% CI, pass rate
