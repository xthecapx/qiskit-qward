# Quantum Computing Researcher - Agent Memory

## Project: QWARD Eigensolver

### Key Files
- Plan: `qward/examples/papers/eigen-solver/plan.md`
- Phase 1: `qward/examples/papers/eigen-solver/phase1_problem_statement.md`
- Phase 2: `qward/examples/papers/eigen-solver/phase2_theoretical_design.md`

### Eigenvalue Corrections (Plan had errors)
- M3 = [[2, 1-1j], [1+1j, 3]]: eigenvalues are {1, 4} (NOT {~1.27, ~3.73})
- M4 = [[2,1,0],[1,3,1],[0,1,2]]: eigenvalues are {1, 2, 4} (NOT {~0.59, ~2.00, ~4.41})
- Characteristic polynomials verified analytically

### VQE Design Decisions
- 1-qubit ansatz: RY+RZ (2 params, full Bloch sphere)
- 2-qubit ansatz: EfficientSU2, L=2 reps (12 params, 2 CX gates)
- Optimizer: COBYLA (ideal), SPSA (noisy)
- 3x3 embedding penalty: p = lambda_max + 2*spectral_range (=10 for M4)
- Large penalties cause huge Pauli coefficients -> excessive shot noise

### Shot Noise Insight
- M4 with penalty=100: sum(c^2) = 1789, needs ~2M shots for 1% accuracy
- M4 with penalty=10: sum(c^2) = 12.2, needs ~30K shots for 1% accuracy
- Penalty optimization is critical for shot-based simulations

### Pauli Decomposition Formula
- c_P = Tr(M * P) / 2^q for q-qubit Pauli string P
- Identity term handled analytically (no measurement needed)

### Tools/Environment
- Use `uv run` not `python` directly
- Qiskit 2.1.2, qiskit-aer 0.17.1
