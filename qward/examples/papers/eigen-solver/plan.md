# Quantum Eigensolver Project Plan

## Project Overview

**Objective**: Implement and validate a quantum eigensolver (VQE-based) for small Hermitian matrices (2x2, 3x3, 4x4) with rigorous classical validation.

**Location**: `qward/examples/papers/eigen-solver/`

**Key Constraint**: All quantum results must be validated against classical eigensolvers (NumPy/SciPy) to ensure correctness before scaling.

---

## Team & Responsibilities

| Agent | Phase | Primary Deliverables |
|-------|-------|---------------------|
| `quantum-research-lead` | 1 (Ideation) | Problem scoping, feasibility assessment, success criteria |
| `quantum-computing-researcher` | 2 (Theoretical Design) | VQE formulation, Hamiltonian encoding, ansatz design |
| `test-engineer` | 3 (Test Design) | Pytest suite defining expected behaviors (TDD) |
| `python-architect` | 4 (Implementation) | Qiskit VQE implementation passing all tests |
| `quantum-data-scientist` | 5 (Execution & Analysis) | Results analysis, quantum vs classical comparison |
| All | 6 (Review) | Iteration decisions, final validation |
| `python-architect` | 7 (Library Integration) | Move code to `qward/algorithms/`, update exports, docs |

---

## Phase 1: Ideation (Lead + Researcher)

### Objective
Define the problem scope, validate quantum approach, and establish success criteria.

### Tasks

**Research Lead:**
- [ ] Assess VQE feasibility for small matrix eigenvalue problems
- [ ] Define success criteria (energy accuracy threshold)
- [ ] Identify NISQ-era constraints relevant to this problem
- [ ] Establish test matrix suite for validation

**Researcher (support):**
- [ ] Confirm VQE is appropriate for general Hermitian matrices
- [ ] Preliminary qubit/depth estimates for 2x2, 3x3, 4x4 cases
- [ ] Identify potential barren plateau risks for chosen ansatz

### Deliverables
```
eigen-solver/
  phase1_problem_statement.md   # Problem definition & success criteria
```

### Success Criteria
- Energy error ≤ 0.01 (1% of spectral range) for ideal simulation
- Energy error ≤ 0.05 (5% of spectral range) with noise

### Test Matrices (Validation Suite)

**2x2 Matrices:**
```python
# Pauli Z (diagonal, trivial)
M1 = [[1, 0], [0, -1]]  # eigenvalues: 1, -1

# Pauli X (off-diagonal)
M2 = [[0, 1], [1, 0]]   # eigenvalues: 1, -1

# General Hermitian
M3 = [[2, 1-1j], [1+1j, 3]]  # eigenvalues: ~1.27, ~3.73
```

**3x3 Matrix:**
```python
# Symmetric real
M4 = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]  # eigenvalues: ~0.59, ~2.00, ~4.41
```

**4x4 Matrix:**
```python
# 2-qubit Heisenberg model (physically relevant)
M5 = Heisenberg_XXX(J=1)  # eigenvalues: -3, 1, 1, 1
```

### Handoff Checklist
- [ ] Problem statement document approved
- [ ] Test matrices defined with classical eigenvalues
- [ ] Success thresholds agreed upon
- [ ] Risk assessment complete

---

## Phase 2: Theoretical Design (Researcher)

### Objective
Design the VQE circuit, Hamiltonian encoding, and optimization strategy.

### Tasks

- [ ] Define Hamiltonian encoding for arbitrary Hermitian matrices
  - Pauli decomposition: H = Σᵢ cᵢ Pᵢ
  - Calculate Pauli coefficients from matrix elements
- [ ] Design hardware-efficient ansatz
  - RealAmplitudes or EfficientSU2 for NISQ compatibility
  - Parameterize depth based on matrix size
- [ ] Define measurement strategy
  - Shots required for energy estimation
  - Error bounds from shot statistics
- [ ] Specify classical optimizer
  - COBYLA or SPSA for noisy environments
  - Convergence criteria
- [ ] Prove correctness
  - Show that ground state of encoded H corresponds to minimum eigenvalue
  - Analyze ansatz expressibility

### Deliverables
```
eigen-solver/
  phase2_theoretical_design.md  # Full theoretical specification
```

**Contents:**
1. Pauli decomposition algorithm for n×n Hermitian matrix
2. Qubit requirements: ⌈log₂(n)⌉ qubits for n×n matrix
3. Ansatz circuit diagram with parameterization
4. Cost function: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
5. Convergence criteria: |E_{k+1} - E_k| < ε
6. Complexity analysis (gates, depth, shots)

### Mathematical Specification

**Pauli Decomposition:**
For n×n Hermitian matrix M:
```
H = (1/n) Σᵢⱼ Tr(M · Pᵢⱼ) × Pᵢⱼ
```
where Pᵢⱼ are n-qubit Pauli strings.

**Ansatz Structure:**
```
|ψ(θ)⟩ = U(θ)|0⟩^⊗n

U(θ) = ∏ₗ [∏ᵢ RY(θₗᵢ) · ∏ᵢⱼ CX(i,j)]
```

### Handoff Checklist
- [ ] Hamiltonian encoding complete with formulas
- [ ] Ansatz specified with circuit diagram
- [ ] Expected eigenvalue ranges documented
- [ ] Edge cases identified (degenerate eigenvalues, etc.)

---

## Phase 3: Test Design (Test Engineer)

### Objective
Write failing tests that define expected VQE behavior (TDD approach).

### Tasks

- [ ] Create test fixtures for test matrices
- [ ] Write classical baseline tests (NumPy eigensolvers)
- [ ] Write VQE correctness tests
- [ ] Write convergence tests
- [ ] Write noise robustness tests

### Deliverables
```
eigen-solver/
  tests/
    conftest.py                    # Shared fixtures
    test_pauli_decomposition.py    # Hamiltonian encoding tests
    test_classical_baseline.py     # NumPy validation (must pass)
    test_vqe_ideal.py              # Ideal VQE (no noise)
    test_vqe_noisy.py              # Noisy simulation tests
    test_convergence.py            # Optimizer convergence tests
```

### Test Categories

**1. Pauli Decomposition Tests:**
```python
def test_pauli_decomposition_2x2():
    """Verify Pauli decomposition reconstructs original matrix."""
    M = np.array([[1, 0], [0, -1]])  # Pauli Z
    H = pauli_decompose(M)
    reconstructed = sum(c * P.to_matrix() for c, P in H.terms())
    assert np.allclose(M, reconstructed)

def test_pauli_decomposition_hermitian():
    """Verify decomposition preserves Hermiticity."""
    M = random_hermitian(4)
    H = pauli_decompose(M)
    assert H.is_hermitian()
```

**2. Classical Baseline Tests (Must Pass First):**
```python
@pytest.fixture
def test_matrices():
    return {
        'pauli_z': (np.array([[1, 0], [0, -1]]), [-1, 1]),
        'pauli_x': (np.array([[0, 1], [1, 0]]), [-1, 1]),
        'general_2x2': (np.array([[2, 1-1j], [1+1j, 3]]), [1.268, 3.732]),
    }

def test_numpy_eigenvalues(test_matrices):
    """Validate classical eigenvalues for all test matrices."""
    for name, (matrix, expected) in test_matrices.items():
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert np.allclose(sorted(eigenvalues), sorted(expected), atol=0.01)
```

**3. VQE Ideal Tests:**
```python
def test_vqe_finds_ground_state_2x2():
    """VQE finds minimum eigenvalue for 2x2 matrix (ideal)."""
    M = np.array([[1, 0], [0, -1]])  # Ground state energy = -1
    vqe = QuantumEigensolver(M)
    result = vqe.solve(shots=None)  # Statevector simulation

    classical_min = np.min(np.linalg.eigvalsh(M))
    assert abs(result.eigenvalue - classical_min) < 0.01

def test_vqe_all_eigenvalues_4x4():
    """VQE finds all eigenvalues via deflation (4x4)."""
    M = heisenberg_xxz(4)
    vqe = QuantumEigensolver(M)
    quantum_eigenvalues = vqe.solve_all()

    classical_eigenvalues = np.linalg.eigvalsh(M)
    assert np.allclose(sorted(quantum_eigenvalues), sorted(classical_eigenvalues), atol=0.05)
```

**4. Noisy Tests:**
```python
@pytest.mark.parametrize("noise_preset", ["IBM-HERON-R2", "RIGETTI-ANKAA3"])
def test_vqe_with_noise(noise_preset):
    """VQE converges within tolerance under realistic noise."""
    M = np.array([[1, 0], [0, -1]])
    vqe = QuantumEigensolver(M, noise_preset=noise_preset)
    result = vqe.solve(shots=8192)

    classical_min = np.min(np.linalg.eigvalsh(M))
    assert abs(result.eigenvalue - classical_min) < 0.10  # Relaxed for noise
```

### Handoff Checklist
- [ ] All test files created
- [ ] Tests run and FAIL (red phase)
- [ ] Classical baseline tests PASS
- [ ] Test documentation complete
- [ ] Fixtures are reusable

---

## Phase 4: Implementation (Python Architect)

### Objective
Implement the VQE eigensolver to pass all tests.

### Tasks

- [ ] Implement Pauli decomposition for Hermitian matrices
- [ ] Implement VQE wrapper using Qiskit
- [ ] Integrate with QWARD executor
- [ ] Implement eigenvalue deflation for finding all eigenvalues
- [ ] Add noise model support

### Deliverables
```
eigen-solver/
  src/
    __init__.py
    pauli_decomposition.py     # Matrix → Hamiltonian conversion
    quantum_eigensolver.py     # VQE-based eigensolver class
    ansatz.py                  # Parameterized circuit builders
    classical_baseline.py      # NumPy reference implementation
```

### Class Architecture

```python
# quantum_eigensolver.py
from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qward.algorithms import QuantumCircuitExecutor

class EigensolverBase(ABC):
    """Abstract base for eigensolvers."""

    @abstractmethod
    def solve(self) -> EigensolverResult:
        """Find minimum eigenvalue."""
        pass

    @abstractmethod
    def solve_all(self) -> List[float]:
        """Find all eigenvalues."""
        pass


class ClassicalEigensolver(EigensolverBase):
    """NumPy-based classical eigensolver for validation."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def solve(self) -> EigensolverResult:
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        return EigensolverResult(
            eigenvalue=eigenvalues[0],
            eigenvector=eigenvectors[:, 0],
            optimal_parameters=None,
            iterations=0
        )

    def solve_all(self) -> List[float]:
        return list(np.linalg.eigvalsh(self.matrix))


class QuantumEigensolver(EigensolverBase):
    """VQE-based quantum eigensolver."""

    def __init__(
        self,
        matrix: np.ndarray,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: Optional[Optimizer] = None,
        noise_preset: Optional[str] = None,
        shots: int = 4096
    ):
        self.matrix = matrix
        self.hamiltonian = pauli_decompose(matrix)
        self.ansatz = ansatz or self._default_ansatz()
        self.optimizer = optimizer or COBYLA(maxiter=200)
        self.noise_preset = noise_preset
        self.shots = shots

    def solve(self) -> EigensolverResult:
        """Find minimum eigenvalue using VQE."""
        # Implementation using Qiskit's VQE
        ...

    def solve_all(self) -> List[float]:
        """Find all eigenvalues using deflation."""
        ...
```

### Integration with QWARD

```python
from qward.algorithms import QuantumCircuitExecutor, get_preset_noise_config

executor = QuantumCircuitExecutor(shots=4096)

# Ideal simulation
result = executor.simulate(circuit)

# Noisy simulation
noise_config = get_preset_noise_config("IBM-HERON-R2")
result = executor.simulate(circuit, noise_model=noise_config)
```

### Handoff Checklist
- [ ] All tests pass (green phase)
- [ ] Code follows project style (.pylintrc)
- [ ] Type hints complete
- [ ] Docstrings present
- [ ] Integration with QWARD verified

---

## Phase 5: Execution & Analysis (Data Scientist)

### Objective
Run comprehensive experiments and compare quantum vs classical results.

### Tasks

- [ ] Run VQE on all test matrices (ideal + noisy)
- [ ] Generate comparison tables: quantum vs classical eigenvalues
- [ ] Visualize convergence behavior
- [ ] Characterize noise impact
- [ ] Statistical significance analysis

### Deliverables
```
eigen-solver/
  results/
    comparison_table.csv           # Quantum vs classical eigenvalues
    convergence_plots/             # Loss vs iteration plots
    noise_analysis.md              # Noise impact characterization
  img/
    eigenvalue_comparison.png      # Bar chart: quantum vs classical
    convergence_2x2.png            # Convergence for 2x2 matrices
    convergence_4x4.png            # Convergence for 4x4 matrices
    noise_impact.png               # Error vs noise level
```

### Analysis Plan

**1. Accuracy Comparison Table:**
| Matrix | Classical λ_min | Quantum λ_min (ideal) | Error | Quantum λ_min (noisy) | Error |
|--------|-----------------|----------------------|-------|----------------------|-------|
| Pauli Z | -1.000 | -0.998 | 0.002 | -0.942 | 0.058 |
| ... | ... | ... | ... | ... | ... |

**2. Convergence Analysis:**
- Plot cost function E(θ) vs optimizer iteration
- Identify convergence rate differences across matrix sizes
- Detect barren plateau signatures (if any)

**3. Noise Characterization:**
- Compare results across noise presets (IBM Heron R1-R3, Rigetti Ankaa-3)
- Quantify error scaling with circuit depth
- Recommend optimal shot counts for each matrix size

**4. Statistical Validation:**
- Run multiple trials (n=10) per configuration
- Report mean ± std for eigenvalue estimates
- Chi-squared test for measurement distribution validity

### Visualization Requirements

```python
from qward import Scanner, Visualizer
from qward.metrics import CircuitPerformanceMetrics

# Circuit analysis
Scanner(vqe_circuit).scan().summary().visualize(save=True, show=False)

# Custom success criteria
def eigenstate_criteria(outcome):
    return outcome == target_eigenstate

perf = CircuitPerformanceMetrics(
    circuit=vqe_circuit,
    job=job_result,
    success_criteria=eigenstate_criteria
)
```

### Handoff Checklist
- [ ] All experiments completed
- [ ] Comparison tables generated
- [ ] Visualizations saved to img/
- [ ] Statistical analysis complete
- [ ] Noise characterization documented

---

## Phase 6: Review (All Agents)

### Objective
Evaluate results against success criteria and iterate if needed.

### Decision Points

| Criterion | Threshold | Action if Failed |
|-----------|-----------|-----------------|
| Ideal accuracy | < 1% error | Researcher: review ansatz expressibility |
| Noisy accuracy | < 5% error | Architect: add error mitigation |
| Convergence | < 200 iterations | Researcher: review optimizer choice |
| Statistical significance | p > 0.01 | Data Scientist: increase shots |

### Iteration Routing

```
IF ideal_error > threshold:
    → Researcher: analyze circuit expressibility
    → Architect: increase ansatz depth

IF noisy_error > threshold:
    → Architect: implement error mitigation (ZNE, PEC)
    → Data Scientist: characterize specific noise sources

IF convergence_fails:
    → Researcher: review cost landscape
    → Architect: try different optimizer (SPSA, ADAM)
```

### Final Deliverables

```
eigen-solver/
  README.md                        # Project overview & usage
  phase1_problem_statement.md      # Problem definition
  phase2_theoretical_design.md     # Mathematical specification
  src/                             # Implementation code
  tests/                           # Test suite
  results/                         # Analysis outputs
  img/                             # Visualizations
  final_report.md                  # Summary & conclusions
```

---

## Phase 7: QWARD Library Integration (Python Architect)

### Objective
Integrate the validated eigensolver into the main QWARD library for public use.

### Rationale
The `eigen-solver/src/` folder is a development sandbox. Once validated, the code should be:
1. Moved into the main `qward/` package structure
2. Properly exposed via `__init__.py` exports
3. Documented in the library's public API
4. Tested as part of the main test suite

### Tasks

- [ ] Create `qward/algorithms/eigensolver/` module structure
- [ ] Move and refactor validated code from `eigen-solver/src/`
- [ ] Update `qward/algorithms/__init__.py` to expose new classes
- [ ] Move tests to `tests/algorithms/test_eigensolver.py`
- [ ] Update QWARD documentation with eigensolver usage examples
- [ ] Add eigensolver to the QWARD skill reference

### Target Library Structure

**From (development sandbox):**
```
qward/examples/papers/eigen-solver/
  src/
    __init__.py
    pauli_decomposition.py
    quantum_eigensolver.py
    ansatz.py
    classical_baseline.py
  tests/
    ...
```

**To (integrated library):**
```
qward/
  algorithms/
    __init__.py                    # Add: from .eigensolver import *
    eigensolver/
      __init__.py                  # Public API exports
      pauli_decomposition.py       # Matrix → Hamiltonian
      quantum_eigensolver.py       # QuantumEigensolver class
      classical_eigensolver.py     # ClassicalEigensolver for validation
      ansatz.py                    # Ansatz builders
      _utils.py                    # Internal utilities

tests/
  algorithms/
    test_eigensolver.py            # Main test file
    test_pauli_decomposition.py    # Unit tests for decomposition
```

### Public API Design

```python
# qward/algorithms/eigensolver/__init__.py
"""Quantum Eigensolver module for computing eigenvalues of Hermitian matrices."""

from .quantum_eigensolver import QuantumEigensolver, EigensolverResult
from .classical_eigensolver import ClassicalEigensolver
from .pauli_decomposition import pauli_decompose, PauliDecomposition
from .ansatz import (
    build_hardware_efficient_ansatz,
    build_real_amplitudes_ansatz,
)

__all__ = [
    "QuantumEigensolver",
    "ClassicalEigensolver",
    "EigensolverResult",
    "pauli_decompose",
    "PauliDecomposition",
    "build_hardware_efficient_ansatz",
    "build_real_amplitudes_ansatz",
]
```

### Usage After Integration

```python
# Users can import directly from qward.algorithms
from qward.algorithms import QuantumEigensolver, ClassicalEigensolver
from qward.algorithms.eigensolver import pauli_decompose

import numpy as np

# Define a Hermitian matrix
M = np.array([[2, 1-1j], [1+1j, 3]])

# Classical baseline
classical = ClassicalEigensolver(M)
classical_result = classical.solve()
print(f"Classical min eigenvalue: {classical_result.eigenvalue}")

# Quantum eigensolver
quantum = QuantumEigensolver(M, noise_preset="IBM-HERON-R2")
quantum_result = quantum.solve(shots=4096)
print(f"Quantum min eigenvalue: {quantum_result.eigenvalue}")

# Compare
error = abs(quantum_result.eigenvalue - classical_result.eigenvalue)
print(f"Error: {error:.4f}")
```

### Documentation Updates

- [ ] Add `docs/algorithms/eigensolver.md` with:
  - Overview and mathematical background
  - API reference
  - Usage examples
  - Performance considerations
- [ ] Update `skills/qward-development/references/` with eigensolver reference
- [ ] Add eigensolver example to `qward/examples/`

### Migration Checklist

| Source (Development) | Target (Library) | Action |
|---------------------|------------------|--------|
| `eigen-solver/src/pauli_decomposition.py` | `qward/algorithms/eigensolver/pauli_decomposition.py` | Move + refactor |
| `eigen-solver/src/quantum_eigensolver.py` | `qward/algorithms/eigensolver/quantum_eigensolver.py` | Move + refactor |
| `eigen-solver/src/classical_baseline.py` | `qward/algorithms/eigensolver/classical_eigensolver.py` | Move + rename |
| `eigen-solver/src/ansatz.py` | `qward/algorithms/eigensolver/ansatz.py` | Move |
| `eigen-solver/tests/*.py` | `tests/algorithms/test_eigensolver*.py` | Move + consolidate |

### Backward Compatibility

The development sandbox (`qward/examples/papers/eigen-solver/`) will be preserved as:
- A reference implementation with full documentation
- An example of the research-to-library workflow
- A location for the research artifacts (phase documents, results, images)

Add a deprecation notice to the sandbox `__init__.py`:
```python
# qward/examples/papers/eigen-solver/src/__init__.py
import warnings
warnings.warn(
    "Importing from qward.examples.papers.eigen_solver is deprecated. "
    "Use 'from qward.algorithms import QuantumEigensolver' instead.",
    DeprecationWarning
)
from qward.algorithms.eigensolver import *
```

### Handoff Checklist
- [ ] All code moved to `qward/algorithms/eigensolver/`
- [ ] `qward/algorithms/__init__.py` updated
- [ ] Tests pass from new location
- [ ] Documentation added
- [ ] Deprecation notice in sandbox
- [ ] Example script works with new imports

---

## Success Metrics Summary

| Metric | Target | Measured By |
|--------|--------|-------------|
| Ideal VQE accuracy | ≤ 1% error vs classical | Data Scientist |
| Noisy VQE accuracy | ≤ 5% error vs classical | Data Scientist |
| All tests pass | 100% | Test Engineer |
| Code coverage | ≥ 80% | Architect |
| Convergence iterations | ≤ 200 | Data Scientist |
| Library integration | Public API exposed | Architect |
| Documentation complete | Usage examples + API docs | Architect |

---

## Timeline & Dependencies

```
Phase 1 (Ideation)
    │
    ▼
Phase 2 (Theoretical Design)
    │
    ▼
Phase 3 (Test Design) ←── Classical baseline tests must pass first
    │
    ▼
Phase 4 (Implementation) ←── Tests define implementation contract
    │
    ▼
Phase 5 (Execution & Analysis)
    │
    ▼
Phase 6 (Review) ──────┐
    │                  │
    │ ◄────────────────┘ (iterate if needed)
    │
    ▼
Phase 7 (Library Integration)
    │
    ▼
  [DONE] → Eigensolver available in qward.algorithms
```

---

## Quick Start Commands

```bash
# Run classical baseline tests (Phase 3 validation)
pytest qward/examples/papers/eigen-solver/tests/test_classical_baseline.py -v

# Run all VQE tests
pytest qward/examples/papers/eigen-solver/tests/ -v

# Run with coverage
pytest qward/examples/papers/eigen-solver/tests/ --cov=qward/examples/papers/eigen-solver/src

# Run noisy tests (slower)
pytest qward/examples/papers/eigen-solver/tests/test_vqe_noisy.py -v --slow
```

---

## Appendix: Reference Materials

- **VQE Theory**: [Peruzzo et al., Nature Communications 5, 4213 (2014)](https://www.nature.com/articles/ncomms5213)
- **Qiskit VQE**: [Qiskit Algorithms Documentation](https://qiskit-community.github.io/qiskit-algorithms/)
- **QWARD Documentation**: See `skills/qward-development/` for API reference
- **Project Standards**: See `.pylintrc` and `requirements.qward.txt`
