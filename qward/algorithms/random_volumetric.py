"""
Random volumetric circuit generator (mirror / Loschmidt echo).

Generates random circuits with controllable width (qubits) and depth, then
applies the inverse — guaranteeing expected output |0...0>. This serves as
the hardware benchmarking control group for Hypothesis 4.

Two modes:
- SU(4) mode: random Haar-distributed 2-qubit unitaries (for simulation)
- Native-gate mode: random gates drawn from the backend's native set (for QPU)
"""

from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


class RandomVolumetric:
    """Mirror circuit for volumetric hardware benchmarking.

    Applies random layers of 2-qubit gates, then their inverse.
    Expected output is always |0...0> (identity operation).

    Args:
        num_qubits: Number of qubits (>= 2).
        depth: Number of random layers. If None, defaults to num_qubits.
        seed: Random seed for reproducibility.
        native_gates: Optional list of native gate names for the backend.
            When None, uses random SU(4) unitaries.
        use_barriers: Whether to add barriers between layers.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: Optional[int] = None,
        seed: int = 42,
        native_gates: Optional[List[str]] = None,
        use_barriers: bool = True,
    ):
        if num_qubits < 2:
            raise ValueError(f"Need at least 2 qubits, got {num_qubits}")

        self._num_qubits = num_qubits
        self._depth = depth if depth is not None else num_qubits
        self._seed = seed
        self._native_gates = native_gates
        self._use_barriers = use_barriers
        self._circuit = self._build_circuit()

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def expected_outcomes(self) -> List[str]:
        """Mirror circuit always returns to |0...0>."""
        return ["0" * self._num_qubits]

    def _build_circuit(self) -> QuantumCircuit:
        n = self._num_qubits
        qc = QuantumCircuit(n, n)
        rng = np.random.default_rng(self._seed)

        layers = []

        for _ in range(self._depth):
            perm = rng.permutation(n)
            layer_gates = []

            for i in range(0, n - 1, 2):
                q1 = int(perm[i])
                q2 = int(perm[i + 1])

                if self._native_gates is None:
                    gate = self._random_su4(rng)
                else:
                    self._random_native_block(rng, qc, q1, q2)
                    layer_gates.append((None, q1, q2))
                    continue

                qc.append(gate, [q1, q2])
                layer_gates.append((gate, q1, q2))

            if self._use_barriers:
                qc.barrier()
            layers.append(layer_gates)

        # Apply inverse layers in reverse order
        for layer_gates in reversed(layers):
            for gate, q1, q2 in reversed(layer_gates):
                if gate is not None:
                    qc.append(gate.inverse(), [q1, q2])
            if self._use_barriers:
                qc.barrier()

        qc.measure(range(n), range(n))
        return qc

    def _random_su4(self, rng: np.random.Generator) -> UnitaryGate:
        """Generate a random SU(4) unitary gate."""
        from scipy.stats import unitary_group

        gate_seed = int(rng.integers(0, 2**31))
        U = unitary_group.rvs(4, random_state=gate_seed)
        return UnitaryGate(U, label="SU4")

    def _random_native_block(
        self, rng: np.random.Generator, qc: QuantumCircuit, q1: int, q2: int
    ) -> None:
        """Apply a random block using native gates directly to the circuit."""
        single_q_gates = [g for g in self._native_gates if g in ("rx", "ry", "rz", "sx", "x", "h")]
        two_q_gates = [g for g in self._native_gates if g in ("cx", "cz", "ecr", "iswap")]

        if not single_q_gates:
            single_q_gates = ["rx", "ry", "rz"]
        if not two_q_gates:
            two_q_gates = ["cx"]

        # Random 1Q gate on q1
        g1 = rng.choice(single_q_gates)
        self._apply_single_gate(qc, g1, q1, rng)

        # Random 1Q gate on q2
        g2 = rng.choice(single_q_gates)
        self._apply_single_gate(qc, g2, q2, rng)

        # Random 2Q entangling gate
        g_ent = rng.choice(two_q_gates)
        self._apply_two_qubit_gate(qc, g_ent, q1, q2)

        # Another random 1Q on each
        g3 = rng.choice(single_q_gates)
        self._apply_single_gate(qc, g3, q1, rng)
        g4 = rng.choice(single_q_gates)
        self._apply_single_gate(qc, g4, q2, rng)

    def _apply_single_gate(
        self, qc: QuantumCircuit, gate_name: str, qubit: int, rng: np.random.Generator
    ):
        """Apply a single-qubit gate (with random angle for parametric gates)."""
        angle = float(rng.uniform(0, 2 * np.pi))
        if gate_name == "rx":
            qc.rx(angle, qubit)
        elif gate_name == "ry":
            qc.ry(angle, qubit)
        elif gate_name == "rz":
            qc.rz(angle, qubit)
        elif gate_name == "sx":
            qc.sx(qubit)
        elif gate_name == "x":
            qc.x(qubit)
        elif gate_name == "h":
            qc.h(qubit)
        else:
            qc.rz(angle, qubit)

    def _apply_two_qubit_gate(self, qc: QuantumCircuit, gate_name: str, q1: int, q2: int):
        """Apply a two-qubit entangling gate."""
        if gate_name == "cx":
            qc.cx(q1, q2)
        elif gate_name == "cz":
            qc.cz(q1, q2)
        elif gate_name == "ecr":
            qc.ecr(q1, q2)
        elif gate_name == "iswap":
            qc.iswap(q1, q2)
        else:
            qc.cx(q1, q2)


class RandomVolumetricCircuitGenerator:
    """Generator for random volumetric circuits at various scales."""

    @staticmethod
    def generate(
        num_qubits: int,
        depth: Optional[int] = None,
        seed: int = 42,
        native_gates: Optional[List[str]] = None,
        use_barriers: bool = True,
    ) -> QuantumCircuit:
        return RandomVolumetric(num_qubits, depth, seed, native_gates, use_barriers).circuit

    @staticmethod
    def get_scaling_configs(max_qubits: int = 14) -> List[dict]:
        """Generate volumetric scaling configurations (qubit × depth grid)."""
        configs = []
        for n in range(2, max_qubits + 1):
            for depth_mult in [1, 2, 3]:
                d = n * depth_mult
                configs.append(
                    {
                        "config_id": f"RV{n}-D{d}",
                        "num_qubits": n,
                        "depth": d,
                        "description": f"{n} qubits, depth {d} ({depth_mult}×n)",
                    }
                )
        return configs
