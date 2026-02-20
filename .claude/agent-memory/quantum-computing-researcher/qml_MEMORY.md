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

## Project: QML Data Encoding Research

### Key Files
- Plan: `qward/examples/papers/qml-data-encoding/plan.md`
- Phase 2: `phase2_encoding_theory.md`, `phase2_expressibility_analysis.md`,
  `phase2_preprocessing_theory.md`, `phase2_experimental_design.md`
- Reports: `reports/phase2_report.md`, `reports/token_budget.md`

### Kernel Derivations (verified analytically)
- Angle (Ry): K = prod_i cos^2((x_i - x_i')/2) -- factorizable, classical
- IQP: non-factorizable, sum over 2^d terms, likely #P-hard
- Amplitude: squared cosine similarity -- classical
- Rz alone: K = 1 (constant!) -- DO NOT USE without H layers
- Re-uploading: adaptive kernel, parameter-dependent

### Expressibility Hierarchy (proved)
- Basis << Angle << IQP << Re-upload(L->inf)
- Proof via entanglement: IQP produces entangled states, Angle only product states

### Critical Preprocessing Insight
- PCA decorrelates features -> neutralizes IQP cross-terms
- Prediction: PCA+Angle ~ PCA+IQP (testable)
- Z-score+sigmoid > MinMax for heavy-tailed data
- Normalization range controls kernel bandwidth

### Experimental Design
- 160 configs -> ~125 valid after exclusions
- Fixed: VQC + RealAmplitudes(reps=2) + COBYLA(maxiter=200) + 1024 shots

### Tools/Environment
- Use `uv run` not `python` directly
- Qiskit 2.1.2, qiskit-aer 0.17.1
