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

### Analysis Patterns
- Always use `os.environ["MPLBACKEND"] = "Agg"` before matplotlib import for headless rendering
- Use `uv run` (not `python`) per CLAUDE.md guidelines
- For n-trial stats: compute mean, std, SEM, 95% CI, pass rate
- Normalized error = |E_VQE - E_exact| / spectral_range
- For independent trials with QuantumEigensolver, bypass solve() fixed seed by calling _run_single_vqe with trial-specific rng seeds
