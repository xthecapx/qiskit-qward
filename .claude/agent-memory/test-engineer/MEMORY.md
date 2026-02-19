# Test Engineer Memory

## Project Structure
- Project root: `/Users/cristianmarquezbarrios/Documents/code/qiskit-qward`
- Existing tests: `tests/` (uses both pytest and unittest)
- Eigensolver tests: `qward/examples/papers/eigen-solver/tests/`
- Eigensolver implementation: `qward/examples/papers/eigen-solver/eigen_solver/src/`
- Package manager: `uv` (use `uv run -m pytest` not `python -m pytest`)
- conftest.py pattern: set `MPLBACKEND=Agg` before any matplotlib import

## Eigensolver Test Matrices (Corrected Eigenvalues from Phase 1)
- M1 (Pauli Z): eigenvalues {-1, 1}, spectral range 2.0
- M2 (Pauli X): eigenvalues {-1, 1}, spectral range 2.0
- M3 ([[2,1-1j],[1+1j,3]]): eigenvalues {1, 4}, spectral range 3.0
- M4 ([[2,1,0],[1,3,1],[0,1,2]]): eigenvalues {1, 2, 4}, spectral range 3.0
- M5 (Heisenberg XXX): eigenvalues {-3, 1, 1, 1}, spectral range 4.0

## Key Patterns
- `pauli_decompose()` returns `PauliDecomposition` object with dict-like interface
- Success criteria: 1% ideal, 5% noisy (relative to spectral range)
- Convergence budget: 200 iterations max
- Import path requires `sys.path` setup for `eigen_solver.src` (conftest handles this)
- Use `shots=None` for statevector (exact) simulation in ideal tests
- Noise presets: "IBM-HERON-R2", "RIGETTI-ANKAA3"
