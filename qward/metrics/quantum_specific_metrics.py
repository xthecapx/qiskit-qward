"""
Quantum Specific Metrics implementation for QWARD.

This module provides the QuantumSpecificMetrics class for extracting metrics
that quantify intrinsically quantum properties of a circuit within the QWARD
framework. These metrics capture phenomena that have no classical analogue and
directly reflect the quantum computational resources present in a circuit.

Quantum-specific metrics include measures of entanglement, coherence, sensitivity,
and non-Clifford “magic,” along with other indicators of quantum behavior that
define the circuit’s potential for quantum advantage and its complexity under
classical simulation. By focusing on fundamental quantum effects rather than
gate counts or structural arrangement, these metrics offer insight into the
circuit’s true quantum nature and resource requirements.

[1] K. Bu, R. J. Garcia, A. Jaffe, D. E. Koh y L. Li, “Complexity of quantum
circuits via sensitivity, magic, and coherence,” Communications in Mathematical
Physics, vol. 405, no. 7, 2024, doi:10.1007/s00220-024-05030-6.

[2] T. Tomesh, P. Gokhale, V. Omole, G. S. Ravi, K. N. Smith, J. Viszlai,
X.-C. Wu, N. Hardavellas, M. R. Martonosi y F. T. Chong, “SupermarQ: A scalable
quantum benchmark suite,” in Proc. 2022 IEEE International Symposium on
High-Performance Computer Architecture (HPCA), 2022, doi:
10.1109/HPCA53966.2022.00050.

[3] J. A. Cruz-Lemus, L. A. Marcelo, and M. Piattini, "Towards a set of metrics for
quantum circuits understandability," in *Quality of Information and Communications
Technology. QUATIC 2021 (Communications in Computer and Information Science, vol. 1439)
*, A. C. R. Paiva, A. R. Cavalli, P. Ventura Martins, and R. Pérez-Castillo, Eds. Cham:
 Springer, 2021, pp. 238–253. doi: 10.1007/978-3-030-85347-1_18.


"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, partial_trace, DensityMatrix
from qiskit.converters import circuit_to_dag

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

# Try to import torch, but handle gracefully if not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import schemas for structured data validation
try:
    from qward.schemas.quantum_specific_metrics_schema import QuantumSpecificMetricsSchema

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# Gates that represent two-qubit interactions
TWO_QUBIT_GATES = {
    "cx",
    "cy",
    "cz",
    "swap",
    "iswap",
    "dcx",
    "ecr",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "xx_minus_yy",
    "xx_plus_yy",
    "ccx",
    "cswap",
    "mcx",
    "mcphase",
    "mcu1",
    "mcu2",
    "mcu3",
    "mcrx",
    "mcry",
    "mcrz",
    "mcp",
    "mcu",
    "mcswap",
}

PAULI_SINGLE = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class QuantumSpecificMetrics(MetricCalculator):
    """
    Extract intrinsically quantum metrics from QuantumCircuit objects.

    Quantum-specific metrics provide insight into the true quantum behavior of
    a circuit, its resource requirements, and its expected difficulty for
    classical simulation. They complement structural and element-level metrics
    by focusing exclusively on quantum effects arising from the circuit’s
    transformations and state evolution.

    Attributes:
        circuit (QuantumCircuit):
            The quantum circuit to analyze (inherited from MetricCalculator).

        _circuit_dag (DAGCircuit | None):
            DAG representation of the circuit, used for structural traversal
            when computing quantum-specific metrics.

        _torch_available (bool):
            Indicates whether PyTorch is available for metrics that rely on
            differentiable simulation or gradient-based optimization.

        _max_steps (int):
            Maximum number of optimization steps used in iterative or
            differentiable metrics (e.g., magic or sensitivity estimation).

        _lr (float):
            Learning rate used for gradient-based optimization of certain metrics.

        _device (str):
            Device specification ("cpu" or "cuda") for metrics that may leverage
            PyTorch computation.

        _use_trace_norm (bool):
            Whether to use the trace norm when computing specific quantum
            sensitivity or distance-based metrics.
    """

    def __init__(self, circuit: QuantumCircuit):
        """
        Inicializa el calculador de métricas cuánticas específicas.
        """
        super().__init__(circuit)
        self._circuit_dag = circuit_to_dag(circuit) if circuit else None
        self._ensure_schemas_available()
        self._torch_available = TORCH_AVAILABLE
        # Parámetros de optimización para métricas diferenciales
        self._max_steps = 300
        self._lr = 0.05
        self._device = "cpu"
        self._use_trace_norm = False

    def _remove_measurements(self, circuit):
        qc_unitary = QuantumCircuit(circuit.num_qubits)
        for ci in circuit.data:
            instr = ci.operation
            qargs = ci.qubits
            if instr.name not in ["measure", "barrier", "reset"]:
                qc_unitary.append(instr, qargs)
        return qc_unitary

    def _make_pauli_x_on_n(self, n_qubits: int, target: int) -> np.ndarray:
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        mats = [np.eye(2, dtype=complex)] * n_qubits
        mats[target] = pauli_x
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def _frobenius_norm(self, mat):
        return torch.sqrt(torch.sum(torch.abs(mat) ** 2))

    def _trace_norm(self, mat):
        sv = torch.linalg.svdvals(mat)  # pylint: disable=not-callable
        return torch.sum(sv)

    # --- Magic ---
    def _calculate_magic(self) -> float:
        if not self._torch_available:
            print(
                "Warning: Magic metric requires PyTorch. Install torch>=1.12.0 to enable this metric."
            )
            return 0.0
        try:
            return self._magic_metric()
        except Exception as e:
            print(f"Warning: Magic calculation failed: {e}")
            return 0.0

    def _magic_metric(self) -> float:
        circuit = self._remove_measurements(self.circuit)
        unitary = Operator(circuit).data
        unitary_t = torch.tensor(unitary, dtype=torch.complex64, device=self._device)
        return self._magic_optimize(unitary_t)

    def _magic_proxy(self, rho_out):
        off_diag = rho_out - torch.diag(torch.diag(rho_out))
        return torch.sum(torch.abs(torch.imag(off_diag)))

    def _magic_optimize(self, unitary):
        dim = unitary.shape[0]
        x = torch.randn(dim, requires_grad=True, device=self._device)
        optimizer = torch.optim.Adam([x], lr=self._lr)
        best_val = 0.0
        for _ in range(self._max_steps):
            p = torch.softmax(x, dim=0)
            rho = torch.diag(p).to(torch.complex64)
            rho_out = unitary @ rho @ torch.conj(unitary.T)
            magic_val = self._magic_proxy(rho_out)
            loss = -magic_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val = float(magic_val.detach().cpu().item())
            best_val = max(best_val, val)
        return best_val

    # --- Coherence ---
    def _calculate_coherence(self) -> float:
        if not self._torch_available:
            print(
                "Warning: Coherence metric requires PyTorch. Install torch>=1.12.0 to enable this metric."
            )
            return 0.0
        try:
            return self._coherence_metric()
        except Exception as e:
            print(f"Warning: Coherence calculation failed: {e}")
            return 0.0

    def _coherence_metric(self) -> float:
        circuit = self._remove_measurements(self.circuit)
        unitary = Operator(circuit).data
        unitary_t = torch.tensor(unitary, dtype=torch.complex64, device=self._device)
        return self._coherence_optimize(unitary_t)

    def _coherence_l1(self, rho):
        off_diag = rho - torch.diag(torch.diag(rho))
        return torch.sum(torch.abs(off_diag))

    def _coherence_optimize(self, unitary):
        dim = unitary.shape[0]
        x = torch.randn(dim, requires_grad=True, device=self._device)
        optimizer = torch.optim.Adam([x], lr=self._lr)
        for _ in range(self._max_steps):
            p = torch.softmax(x, dim=0)
            rho = torch.diag(p).to(torch.complex64)
            rho_out = unitary @ rho @ torch.conj(unitary.T)
            coherence = self._coherence_l1(rho_out)
            loss = -coherence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return float(-loss.item())

    # --- Sensitivity ---
    def _generate_pauli_labels(self, n_qubits: int, w_max: int) -> List[Tuple[str, ...]]:
        """Generate Pauli labels of weight ≤ w_max (e.g. ('X','I','Z'))."""
        import itertools

        labels = []
        for w in range(0, w_max + 1):
            for positions in itertools.combinations(range(n_qubits), w):
                for prod in itertools.product(["X", "Y", "Z"], repeat=w):
                    label = ["I"] * n_qubits
                    for pos, sym in zip(positions, prod):
                        label[pos] = sym
                    labels.append(tuple(label))
        return labels

    def _pauli_label_to_matrix(self, label: Tuple[str, ...]) -> np.ndarray:
        """Convert a tuple of ('I','X','Y','Z') to its matrix via kron."""
        mats = [PAULI_SINGLE[s] for s in label]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def _influence_from_coeffs(
        self, coeffs: "torch.Tensor", labels: List[Tuple[str, ...]], n_qubits: int, dim_factor: int
    ) -> "torch.Tensor":
        """Compute total influence as in Bu et al. but on restricted basis."""
        q_a = (coeffs.abs() ** 2) * float(dim_factor)
        per_qubit = torch.zeros(n_qubits, dtype=torch.float32, device=self._device)
        for idx, label in enumerate(labels):
            for i, sym in enumerate(label):
                if sym != "I":
                    per_qubit[i] += q_a[idx].real
        return torch.sum(per_qubit)

    def _pauli_coeffs_restricted(
        self, obs_t: "torch.Tensor", pauli_t_list: List["torch.Tensor"], dim_factor: int
    ) -> "torch.Tensor":
        """Return coefficients c_a = Tr(P_a^† O)/2^n for a restricted list of Paulis."""
        coeffs = []
        denom = float(dim_factor)
        for pauli in pauli_t_list:
            coeff = torch.trace(torch.conj(pauli).T @ obs_t) / denom
            coeffs.append(coeff)
        return torch.stack(coeffs)

    def _calculate_sensitivity(self) -> float:
        """Approximate Circuit Sensitivity (CiS) from Bu et al., practical version."""
        if not self._torch_available:
            print("Warning: Sensitivity metric requires PyTorch.")
            return 0.0

        # --- Precompute data ---
        circuit = self._remove_measurements(self.circuit)
        unitary_np = Operator(circuit).data
        unitary_t = torch.tensor(unitary_np, dtype=torch.complex64, device=self._device)
        dim = unitary_np.shape[0]
        n_qubits = int(np.log2(dim))
        dim_factor = 2**n_qubits

        # Restricted Pauli sets
        w_max_obs = 1  # parametrize O as combo of weight-1 Paulis
        w_max_eval = 2  # evaluate expansion up to weight 2
        labels_obs = self._generate_pauli_labels(n_qubits, w_max_obs)
        labels_eval = self._generate_pauli_labels(n_qubits, w_max_eval)

        # Precompute torch Paulis
        def np_to_torch(mat_np):
            real = torch.tensor(mat_np.real, dtype=torch.float32, device=self._device)
            imag = torch.tensor(mat_np.imag, dtype=torch.float32, device=self._device)
            return torch.complex(real, imag)

        pauli_eval_t = [np_to_torch(self._pauli_label_to_matrix(lbl)) for lbl in labels_eval]
        pauli_obs_t = [np_to_torch(self._pauli_label_to_matrix(lbl)) for lbl in labels_obs]

        # Parameters (real coefficients α_j for O)
        m = len(labels_obs)
        params = torch.randn(m, device=self._device, dtype=torch.float32) * 0.1
        params.requires_grad_(True)
        opt = torch.optim.Adam([params], lr=self._lr)

        best_val = 0.0

        for _ in range(self._max_steps):
            opt.zero_grad()
            # Construct O = sum α_j P_j
            obs_t = torch.zeros((dim, dim), dtype=torch.complex64, device=self._device)
            for a, alpha in enumerate(params):
                obs_t = obs_t + alpha * pauli_obs_t[a]

            # Normalize Hilbert–Schmidt
            hs_norm = torch.sqrt(torch.real(torch.trace(torch.conj(obs_t).T @ obs_t)))
            obs_t = obs_t / (hs_norm + 1e-12)

            # Influence before
            coeffs_obs = self._pauli_coeffs_restricted(obs_t, pauli_eval_t, dim_factor)
            influence_obs = self._influence_from_coeffs(
                coeffs_obs, labels_eval, n_qubits, dim_factor
            )

            # After conjugation
            obs_prime = unitary_t @ obs_t @ torch.conj(unitary_t).T
            coeffs_obs_prime = self._pauli_coeffs_restricted(obs_prime, pauli_eval_t, dim_factor)
            influence_obs_prime = self._influence_from_coeffs(
                coeffs_obs_prime, labels_eval, n_qubits, dim_factor
            )

            val = torch.abs(influence_obs_prime - influence_obs)
            loss = -val
            loss.backward()
            opt.step()

            current_val = float(val.detach().cpu().item())
            best_val = max(best_val, current_val)

        return float(best_val)

    def _get_metric_type(self) -> MetricsType:
        """
        Get the type of this metric.

        Returns:
            MetricsType: PRE_RUNTIME (can be calculated without execution)
        """
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """
        Get the ID of this metric.

        Returns:
            MetricsId: QUANTUM_SPECIFIC
        """
        return MetricsId.QUANTUM_SPECIFIC

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the circuit is available, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> QuantumSpecificMetricsSchema:
        """
        Calculate and return quantum specific metrics.

        Returns:
            QuantumSpecificMetricsSchema: Validated schema with all metrics

        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError("QuantumSpecificMetricsSchema is not available")

        # Calculate %SpposQ metric
        spposq_ratio = self._calculate_spposq_ratio()

        # Calculate Magic metric
        magic_value = self._calculate_magic()

        # Calculate Coherence metric
        coherence_value = self._calculate_coherence()

        # Calculate Sensitivity metric
        sensitivity_value = self._calculate_sensitivity()

        # Calculate Entanglement-Ratio metric
        entanglement_ratio = self._calculate_entanglement_ratio()

        return QuantumSpecificMetricsSchema(
            spposq_ratio=spposq_ratio,
            magic=magic_value,
            coherence=coherence_value,
            sensitivity=sensitivity_value,
            entanglement_ratio=entanglement_ratio,
        )

    def _calculate_spposq_ratio(self) -> float:
        """
        Calculate %SpposQ: Ratio of qubits with a Hadamard gate as initial gate.

        Returns:
            float: Ratio of qubits that start with a Hadamard gate
        """
        if not self.circuit or self.circuit.num_qubits == 0:
            return 0.0

        qubits_with_hadamard_start = 0
        total_qubits = self.circuit.num_qubits

        # For each qubit, check if the first operation is a Hadamard gate
        active_qubits = 0
        for qubit_index in range(total_qubits):
            qubit = self.circuit.qubits[qubit_index]
            for instruction in self.circuit.data:
                if qubit in instruction.qubits:
                    active_qubits += 1
                    gate_name = instruction.operation.name.lower()
                    if gate_name == "h":
                        qubits_with_hadamard_start += 1
                    break  # first operation found
        if active_qubits == 0:
            return 0.0
        return qubits_with_hadamard_start / active_qubits

    def _calculate_entanglement_ratio(self) -> float:
        """
        Calculate Entanglement-Ratio: Ratio of two-qubit interactions to total operations.

        Formula: E = ne/ng
        where:
        - ne: number of two-qubit interactions
        - ng: total number of gate operations

        Returns:
            float: Entanglement ratio
        """
        if not self.circuit:
            return 0.0

        two_qubit_gates = 0
        total_gate_operations = 0

        for instruction in self.circuit.data:
            gate_name = instruction.operation.name.lower()

            # Skip measurement, barrier, and reset operations
            if gate_name in {"measure", "barrier", "reset"}:
                continue

            total_gate_operations += 1

            # Check if it's a two-qubit gate
            if gate_name in TWO_QUBIT_GATES:
                two_qubit_gates += 1
            elif len(instruction.qubits) >= 2:
                # If gate operates on 2 or more qubits, count as two-qubit interaction
                two_qubit_gates += 1

        if total_gate_operations == 0:
            return 0.0

        return two_qubit_gates / total_gate_operations

    def _ensure_schemas_available(self):
        """
        Ensure that the required schemas are available.

        Raises:
            ImportError: If schemas are not available
        """
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "QuantumSpecificMetricsSchema is not available. "
                "Please ensure that the schemas module is properly imported."
            )
