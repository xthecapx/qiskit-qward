"""
Quantum Specific Metrics implementation for QWARD.

This module provides the QuantumSpecificMetrics class that calculates:
- %SpposQ: Ratio of qubits with a Hadamard gate as initial gate
- Magic: Quantum magic measure (non-Cliffordness)
- Coherence: Coherence power measure
- Sensitivity: Circuit sensitivity measure
- Entanglement-Ratio: Ratio of two-qubit interactions to total operations

These metrics provide quantum-specific analysis focusing on quantum properties
that are critical for quantum advantage and quantum computing performance.
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
    'cx', 'cy', 'cz', 'swap', 'iswap', 'dcx', 'ecr',
    'rxx', 'ryy', 'rzz', 'rzx', 'xx_minus_yy', 'xx_plus_yy',
    'ccx', 'cswap', 'mcx', 'mcphase', 'mcu1', 'mcu2', 'mcu3',
    'mcrx', 'mcry', 'mcrz', 'mcp', 'mcu', 'mcswap'
}



class QuantumSpecificMetrics(MetricCalculator):
    """
    Quantum Specific Metrics calculator for QWARD.
    
    This class provides quantum-specific metrics that focus on quantum properties
    critical for quantum advantage and quantum computing performance.
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
            if instr.name not in ['measure', 'barrier', 'reset']:
                qc_unitary.append(instr, qargs)
        return qc_unitary

    def _make_pauli_x_on_n(self, n_qubits: int, target: int) -> np.ndarray:
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        mats = [np.eye(2, dtype=complex)] * n_qubits
        mats[target] = X
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def _frobenius_norm(self, A):
        return torch.sqrt(torch.sum(torch.abs(A)**2))

    def _trace_norm(self, A):
        sv = torch.linalg.svdvals(A)
        return torch.sum(sv)

    # --- Magic ---
    def _calculate_magic(self) -> float:
        if not self._torch_available:
            print("Warning: Magic metric requires PyTorch. Install torch>=1.12.0 to enable this metric.")
            return 0.0
        try:
            return self._magic_metric()
        except Exception as e:
            print(f"Warning: Magic calculation failed: {e}")
            return 0.0

    def _magic_metric(self) -> float:
        circuit = self._remove_measurements(self.circuit)
        U = Operator(circuit).data
        U_t = torch.tensor(U, dtype=torch.complex64, device=self._device)
        return self._magic_optimize(U_t)

    def _magic_proxy(self, rho_out):
        off_diag = rho_out - torch.diag(torch.diag(rho_out))
        return torch.sum(torch.abs(torch.imag(off_diag)))

    def _magic_optimize(self, U):
        d = U.shape[0]
        x = torch.randn(d, requires_grad=True, device=self._device)
        optimizer = torch.optim.Adam([x], lr=self._lr)
        best_val = 0.0
        for _ in range(self._max_steps):
            p = torch.softmax(x, dim=0)
            rho = torch.diag(p).to(torch.complex64)
            rho_out = U @ rho @ torch.conj(U.T)
            Mval = self._magic_proxy(rho_out)
            loss = -Mval
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val = float(Mval.detach().cpu().item())
            if val > best_val:
                best_val = val
        return best_val

    # --- Coherence ---
    def _calculate_coherence(self) -> float:
        if not self._torch_available:
            print("Warning: Coherence metric requires PyTorch. Install torch>=1.12.0 to enable this metric.")
            return 0.0
        try:
            return self._coherence_metric()
        except Exception as e:
            print(f"Warning: Coherence calculation failed: {e}")
            return 0.0

    def _coherence_metric(self) -> float:
        circuit = self._remove_measurements(self.circuit)
        U = Operator(circuit).data
        U_t = torch.tensor(U, dtype=torch.complex64, device=self._device)
        return self._coherence_optimize(U_t)

    def _coherence_l1(self, rho):
        off_diag = rho - torch.diag(torch.diag(rho))
        return torch.sum(torch.abs(off_diag))

    def _coherence_optimize(self, U):
        d = U.shape[0]
        x = torch.randn(d, requires_grad=True, device=self._device)
        optimizer = torch.optim.Adam([x], lr=self._lr)
        for _ in range(self._max_steps):
            p = torch.softmax(x, dim=0)
            rho = torch.diag(p).to(torch.complex64)
            rho_out = U @ rho @ torch.conj(U.T)
            C = self._coherence_l1(rho_out)
            loss = -C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return float(-loss.item())

    # --- Sensitivity ---
    def _calculate_sensitivity(self) -> float:
        if not self._torch_available:
            print("Warning: Sensitivity metric requires PyTorch. Install torch>=1.12.0 to enable this metric.")
            return 0.0
        try:
            return self._sensitivity_metric()
        except Exception as e:
            print(f"Warning: Sensitivity calculation failed: {e}")
            return 0.0

    def _sensitivity_metric(self) -> float:
        circuit = self._remove_measurements(self.circuit)
        U_np = Operator(circuit).data
        U_t = torch.tensor(U_np, dtype=torch.complex64, device=self._device)
        d = U_np.shape[0]
        n_qubits = int(np.log2(d))
        best_overall = 0.0
        for j in range(n_qubits):
            Xj_np = self._make_pauli_x_on_n(n_qubits, j)
            Xj_t = torch.tensor(Xj_np, dtype=torch.complex64, device=self._device)
            val_j = self._sensitivity_optimize(U_t, Xj_t)
            if val_j > best_overall:
                best_overall = val_j
        return float(best_overall)

    def _sensitivity_optimize(self, U, X_j_t):
        d = U.shape[0]
        n_qubits = int(np.log2(d))
        x = torch.randn(d, requires_grad=True, device=self._device)
        opt = torch.optim.Adam([x], lr=self._lr)
        best_val = 0.0
        for step in range(self._max_steps):
            p = torch.softmax(x, dim=0)
            rho = torch.diag(p).to(torch.complex64)
            rho_flip = X_j_t @ rho @ torch.conj(X_j_t.T)
            rho_out = U @ rho @ torch.conj(U.T)
            rho_out_flip = U @ rho_flip @ torch.conj(U.T)
            d_half = 2 ** (n_qubits - 1)
            def partial_trace_over_qubit(mat):
                mat = mat.reshape([2, d_half, 2, d_half])
                res = torch.zeros((d_half, d_half), dtype=torch.complex64, device=self._device)
                for k in range(2):
                    res = res + mat[k, :, k, :]
                return res
            red = partial_trace_over_qubit(rho_out)
            red_flip = partial_trace_over_qubit(rho_out_flip)
            Delta = red - red_flip
            if self._use_trace_norm:
                val = self._trace_norm(Delta)
            else:
                val = self._frobenius_norm(Delta)
            loss = -val
            opt.zero_grad()
            loss.backward()
            opt.step()
            current_val = float(val.detach().cpu().item())
            if current_val > best_val:
                best_val = current_val
        return best_val

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
            entanglement_ratio=entanglement_ratio
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
                    if gate_name == 'h':
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
            if gate_name in {'measure', 'barrier', 'reset'}:
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
