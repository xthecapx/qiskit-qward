# Python Architect Agent Memory

## Project: QWARD Eigen-Solver

### Key Architecture Notes
- **Executor**: `qward.algorithms.executor.QuantumCircuitExecutor` - unified interface for Aer, IBM, AWS Braket
  - `simulate()` returns dict with `counts` and `qward_metrics`
  - Noise via `noise_model` param: string type, `NoiseModel`, or `NoiseConfig`
- **Noise**: `qward.algorithms.noise_generator` - factory pattern with presets
  - Hardware presets: `IBM-HERON-R1`, `IBM-HERON-R2`, `IBM-HERON-R3`, `RIGETTI-ANKAA3`
  - `get_preset_noise_config(name)` returns `NoiseConfig`
  - `NoiseModelGenerator.create_from_config(config)` returns `NoiseModel`
- **Scanner/Metrics**: `qward.Scanner` with strategy pattern (`QiskitMetrics`, `ComplexityMetrics`, `CircuitPerformanceMetrics`)

### Eigensolver Library Module (COMPLETE)
- Library location: `qward/algorithms/eigensolver/`
- Sandbox location: `qward/examples/papers/eigen-solver/eigen_solver/src/`
- Public API: `from qward.algorithms import QuantumEigensolver, ClassicalEigensolver`
- Submodule API: `from qward.algorithms.eigensolver import pauli_decompose`
- Library tests: `tests/test_eigensolver.py` (21 tests)
- Sandbox tests: `qward/examples/papers/eigen-solver/tests/` (108 tests)

### Qiskit Version Constraints
- qiskit==2.1.2, qiskit-aer==0.17.1, qiskit-ibm-runtime==0.41.1
- **qiskit_algorithms is NOT installed** - must implement VQE from scratch
- Qiskit 2.x uses Primitives V2 (EstimatorV2, SamplerV2)
- Available: `StatevectorEstimator`, `SparsePauliOp`, `RealAmplitudes`
- `SparsePauliOp.from_operator(Operator(matrix))` handles Pauli decomposition
- Use `scipy.optimize.minimize` with COBYLA for classical optimization loop
- **DEPRECATION**: `EfficientSU2` class deprecated in Qiskit 2.1 -- use `efficient_su2()` function
- AerEstimator: `from qiskit_aer.primitives import EstimatorV2 as AerEstimator`

### Eigensolver Implementation (Phase 4 COMPLETE)
- All 108 tests pass: classical baseline (33), convergence (14), pauli (23), ideal (24), noisy (14)
- Import path: `from eigen_solver.src.<module> import ...`
- conftest adds `eigen-solver/` dir to sys.path; package is `eigen_solver/` (underscore)
- PauliDecomposition: dict-like interface + SparsePauliOp for quantum primitives
- VQE: StatevectorEstimator for ideal, AerEstimator for noisy
- Deflation: penalty-based using density matrix projectors decomposed into Pauli basis

### Code Style
- Max line length: 100 chars
- Use `uv run` for all Python execution
- Follow `.pylintrc` configuration

### Existing Algorithm Patterns
- Algorithms module uses flat structure with `__init__.py` re-exports
- Dataclass-based result types (e.g., `IBMJobResult`, `AWSJobResult`)
- Scanner integration via strategy pattern
