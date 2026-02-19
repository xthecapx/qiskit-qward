"""
VQE-based quantum eigensolver for Hermitian matrices.

Implements the Variational Quantum Eigensolver (VQE) algorithm to find
eigenvalues of small Hermitian matrices (2x2, 3x3, 4x4). Supports both
ideal (statevector) and noisy (shot-based) simulation.

The implementation uses Qiskit Primitives V2 (StatevectorEstimator) for
statevector simulation and Aer with noise models for noisy simulation.

References:
    - Peruzzo et al., Nature Communications 5, 4213 (2014)
    - Phase 2: phase2_theoretical_design.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.optimize import minimize

from .ansatz import build_ansatz
from .pauli_decomposition import pauli_decompose, PauliDecomposition


@dataclass
class EigensolverResult:
    """Result from an eigensolver computation.

    Attributes:
        eigenvalue: The computed eigenvalue.
        eigenvector: The eigenvector (statevector) if available.
        optimal_parameters: Optimal variational parameters (VQE only).
        iterations: Number of optimizer iterations.
        cost_history: History of cost function values during optimization.
        converged: Whether the optimizer reported convergence.
    """

    eigenvalue: float
    eigenvector: Optional[np.ndarray] = None
    optimal_parameters: Optional[np.ndarray] = None
    iterations: int = 0
    cost_history: Optional[List[float]] = None
    converged: bool = False


class EigensolverBase(ABC):
    """Abstract base class for eigensolvers."""

    @abstractmethod
    def solve(self, **kwargs) -> EigensolverResult:
        """Find the minimum eigenvalue."""

    @abstractmethod
    def solve_all(self) -> List[float]:
        """Find all eigenvalues."""


class QuantumEigensolver(EigensolverBase):
    """VQE-based quantum eigensolver for Hermitian matrices.

    Uses the variational principle to find eigenvalues by minimizing
    the expectation value E(theta) = <psi(theta)|H|psi(theta)> over
    parameterized quantum circuits.

    Args:
        matrix: Hermitian matrix (numpy ndarray).
        ansatz: Custom ansatz circuit. If None, auto-selected.
        optimizer: Optimizer name ('COBYLA' or 'SPSA').
        noise_preset: Noise preset name for noisy simulation
            (e.g., 'IBM-HERON-R2', 'RIGETTI-ANKAA3').
        shots: Default number of shots for noisy simulation.
        maxiter: Maximum optimizer iterations.
        num_restarts: Number of random restarts for robustness.

    Raises:
        ValueError: If matrix is not Hermitian.
        TypeError: If matrix is not a numpy ndarray.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        ansatz: Optional[QuantumCircuit] = None,
        optimizer: str = "COBYLA",
        noise_preset: Optional[str] = None,
        shots: int = 4096,
        maxiter: int = 200,
        num_restarts: int = 3,
    ):
        self._validate_matrix(matrix)
        self.matrix = matrix
        self._decomposition = pauli_decompose(matrix)
        self.hamiltonian = self._decomposition.sparse_pauli_op
        self.num_qubits = self._decomposition.num_qubits
        self.ansatz = ansatz or build_ansatz(self.num_qubits)
        self.optimizer = optimizer
        self.noise_preset = noise_preset
        self.shots = shots
        self.maxiter = maxiter
        self.num_restarts = num_restarts

    @staticmethod
    def _validate_matrix(matrix: np.ndarray) -> None:
        """Validate input matrix is Hermitian."""
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"Expected numpy ndarray, got {type(matrix).__name__}")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
            raise ValueError("Matrix is not Hermitian (M != M^dagger)")

    def _cost_function_statevector(
        self, params: np.ndarray, estimator, cost_history: list
    ) -> float:
        """Evaluate the VQE cost function using statevector simulation.

        Args:
            params: Current variational parameters.
            estimator: Qiskit StatevectorEstimator instance.
            cost_history: List to append cost values to.

        Returns:
            Energy expectation value.
        """
        bound_circuit = self.ansatz.assign_parameters(params)
        pub = (bound_circuit, self.hamiltonian)
        job = estimator.run([pub])
        result = job.result()
        energy = float(result[0].data.evs)
        cost_history.append(energy)
        return energy

    def _cost_function_noisy(self, params: np.ndarray, noise_model, cost_history: list) -> float:
        """Evaluate the VQE cost function using noisy shot-based simulation.

        Uses Aer simulator with a noise model to simulate realistic
        hardware conditions.

        Args:
            params: Current variational parameters.
            noise_model: Qiskit Aer NoiseModel instance.
            cost_history: List to append cost values to.

        Returns:
            Estimated energy from shot statistics.
        """
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator

        if noise_model is not None:
            backend = AerSimulator(noise_model=noise_model)
        else:
            backend = AerSimulator()

        estimator = AerEstimator.from_backend(backend)
        estimator.options.default_shots = self.shots

        bound_circuit = self.ansatz.assign_parameters(params)
        pub = (bound_circuit, self.hamiltonian)
        job = estimator.run([pub])
        result = job.result()
        energy = float(result[0].data.evs)
        cost_history.append(energy)
        return energy

    def _get_noise_model(self):
        """Get the noise model from a preset name.

        Returns:
            NoiseModel instance, or None for ideal simulation.
        """
        if self.noise_preset is None:
            return None

        from qward.algorithms import (
            get_preset_noise_config,
            NoiseModelGenerator,
        )

        config = get_preset_noise_config(self.noise_preset)
        return NoiseModelGenerator.create_from_config(config)

    def _run_single_vqe(
        self,
        hamiltonian: SparsePauliOp,
        shots: Optional[int],
        initial_params: np.ndarray,
    ) -> EigensolverResult:
        """Run a single VQE optimization from given initial parameters.

        Args:
            hamiltonian: The Hamiltonian to minimize.
            shots: Number of shots (None for statevector).
            initial_params: Initial variational parameters.

        Returns:
            EigensolverResult from this single run.
        """
        cost_history = []
        original_hamiltonian = self.hamiltonian
        self.hamiltonian = hamiltonian

        try:
            if shots is None and self.noise_preset is None:
                # Ideal statevector simulation
                estimator = StatevectorEstimator()

                def cost_fn(params):
                    return self._cost_function_statevector(params, estimator, cost_history)

            else:
                # Noisy or shot-based simulation
                noise_model = self._get_noise_model()

                def cost_fn(params):
                    return self._cost_function_noisy(params, noise_model, cost_history)

            optimizer_options = {"maxiter": self.maxiter}
            if self.optimizer.upper() == "COBYLA":
                method = "COBYLA"
                optimizer_options["rhobeg"] = 0.5
            elif self.optimizer.upper() == "SPSA":
                # SPSA is not directly available in scipy.optimize.minimize.
                # Use Nelder-Mead as a gradient-free alternative that
                # handles noisy cost functions well.
                method = "Nelder-Mead"
                optimizer_options = {
                    "maxiter": self.maxiter,
                    "xatol": 1e-4,
                    "fatol": 1e-4,
                }
            elif self.optimizer.upper() == "L-BFGS-B":
                method = "L-BFGS-B"
            else:
                method = self.optimizer

            opt_result = minimize(
                cost_fn,
                initial_params,
                method=method,
                options=optimizer_options,
            )

            eigenvalue = float(opt_result.fun)

            # Extract eigenvector from the optimal circuit
            eigenvector = self._extract_statevector(opt_result.x)

            return EigensolverResult(
                eigenvalue=eigenvalue,
                eigenvector=eigenvector,
                optimal_parameters=opt_result.x,
                iterations=int(opt_result.nfev),
                cost_history=cost_history,
                converged=opt_result.success,
            )
        finally:
            self.hamiltonian = original_hamiltonian

    def _extract_statevector(self, params: np.ndarray) -> Optional[np.ndarray]:
        """Extract the statevector from the optimized ansatz.

        Args:
            params: Optimal variational parameters.

        Returns:
            Statevector as numpy array, or None if extraction fails.
        """
        try:
            from qiskit.quantum_info import Statevector

            bound_circuit = self.ansatz.assign_parameters(params)
            sv = Statevector.from_instruction(bound_circuit)
            return sv.data
        except Exception:
            return None

    def solve(self, shots: Optional[int] = None, **kwargs) -> EigensolverResult:
        """Find the minimum eigenvalue using VQE.

        Runs multiple random restarts and returns the best result.

        Args:
            shots: Number of shots per evaluation. None for statevector
                (exact expectation values). Overrides self.shots if provided
                explicitly; if not provided and noise_preset is set, uses
                self.shots.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            EigensolverResult with the minimum eigenvalue found.
        """
        if shots is None and self.noise_preset is not None:
            shots = self.shots

        num_params = self.ansatz.num_parameters
        rng = np.random.default_rng(42)

        best_result = None
        for _ in range(self.num_restarts):
            initial_params = rng.uniform(0, np.pi, size=num_params)
            result = self._run_single_vqe(self.hamiltonian, shots, initial_params)
            if best_result is None or result.eigenvalue < best_result.eigenvalue:
                best_result = result

        return best_result

    def solve_all(self) -> List[float]:
        """Find all eigenvalues using VQE with deflation.

        Iteratively finds each eigenvalue by adding a penalty term
        that shifts previously found eigenstates above the spectrum.

        For embedded (padded) matrices, only the original eigenvalues
        are returned (penalty artifacts are discarded).

        Returns:
            List of all eigenvalues (sorted ascending).
        """
        n = self._decomposition.original_dimension
        num_eigenvalues = n

        eigenvalues = []
        found_states = []
        current_hamiltonian = self.hamiltonian

        # Estimate spectral norm for penalty strength
        ham_matrix = current_hamiltonian.to_matrix()
        eigs_est = np.linalg.eigvalsh(ham_matrix)
        lam_max_est = eigs_est[-1]

        for k in range(num_eigenvalues):
            # Build deflated Hamiltonian
            if found_states:
                deflated_matrix = current_hamiltonian.to_matrix().copy()
                for j, state in enumerate(found_states):
                    alpha = 2.0 * (lam_max_est - eigenvalues[j])
                    projector = np.outer(state, state.conj())
                    deflated_matrix += alpha * projector
                deflated_hamiltonian = SparsePauliOp.from_operator(Operator(deflated_matrix))
            else:
                deflated_hamiltonian = current_hamiltonian

            num_params = self.ansatz.num_parameters
            rng = np.random.default_rng(42 + k)
            restarts = self.num_restarts + 2 if k > 0 else self.num_restarts

            best_result = None
            for _ in range(restarts):
                initial_params = rng.uniform(0, np.pi, size=num_params)
                result = self._run_single_vqe(deflated_hamiltonian, None, initial_params)
                if best_result is None or result.eigenvalue < best_result.eigenvalue:
                    best_result = result

            eigenvalues.append(best_result.eigenvalue)
            if best_result.eigenvector is not None:
                found_states.append(best_result.eigenvector)

        return sorted(eigenvalues)
