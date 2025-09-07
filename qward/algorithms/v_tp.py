from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import random
from abc import ABC, abstractmethod


class QuantumGate:
    """Represents a quantum gate with its name and parameters."""

    def __init__(self, name, params=None):
        self.name = name
        self.params = params or []

    def apply(self, qc, qubit):
        getattr(qc, self.name)(qubit)

    def apply_conjugate(self, qc, qubit):
        if self.name == "s":
            qc.sdg(qubit)  # S-dagger is the conjugate of S
        elif self.name == "sdg":
            qc.s(qubit)  # S is the conjugate of S-dagger
        else:
            # x, z, h, y are self-inverse
            self.apply(qc, qubit)


class BaseTeleportation(ABC):
    """Abstract base class for teleportation protocols.

    This class defines the common structure for all teleportation protocols:
    1. Bell state preparation between Alice and Bob
    2. Alice's operations on message and entangled qubits
    3. Protocol-specific correction operations
    """

    def __init__(self, use_barriers: bool = True):
        self.message_qubit = QuantumRegister(1, "M")
        self.alice_entangled = QuantumRegister(1, "A")
        self.bob_entangled = QuantumRegister(1, "B")
        self.use_barriers = use_barriers
        self.circuit = QuantumCircuit(self.message_qubit, self.alice_entangled, self.bob_entangled)
        self._create_protocol()

    def _create_protocol(self):
        """Create the complete teleportation protocol."""
        self._prepare_bell_state()
        self._alice_operations()
        self._apply_corrections()

    def _prepare_bell_state(self):
        """Prepare the entangled pair (Bell state) between Alice and Bob."""
        self.circuit.h(self.alice_entangled)
        self.circuit.cx(self.alice_entangled, self.bob_entangled)

    def _alice_operations(self):
        """Alice's operations on her qubits."""
        self.circuit.cx(self.message_qubit, self.alice_entangled)
        self.circuit.h(self.message_qubit)
        if self.use_barriers:
            self.circuit.barrier()

    @abstractmethod
    def _apply_corrections(self):
        """Apply protocol-specific corrections. Must be implemented by subclasses."""
        pass

    def draw(self):
        return self.circuit.draw(output="mpl")


class StandardTeleportationProtocol(BaseTeleportation):
    """Standard quantum teleportation protocol with conditional corrections.

    This implements the traditional teleportation protocol where Bob's corrections
    are applied conditionally based on Alice's measurement results using classical
    control flow (if statements in the circuit).
    """

    def __init__(self, use_barriers: bool = True):  # pylint: disable=super-init-not-called
        # Create classical register for Alice's measurements
        self.alice_measurements = ClassicalRegister(2, "alice_meas")

        # Initialize base attributes first
        self.message_qubit = QuantumRegister(1, "M")
        self.alice_entangled = QuantumRegister(1, "A")
        self.bob_entangled = QuantumRegister(1, "B")
        self.use_barriers = use_barriers

        # Create circuit with all registers including classical
        self.circuit = QuantumCircuit(
            self.message_qubit, self.alice_entangled, self.bob_entangled, self.alice_measurements
        )

        # Now create the protocol
        self._create_protocol()

    def _alice_operations(self):
        """Alice's operations including measurements."""
        # Alice's Bell measurement operations
        self.circuit.cx(self.message_qubit, self.alice_entangled)
        self.circuit.h(self.message_qubit)

        # Alice measures her qubits
        self.circuit.measure(self.message_qubit, self.alice_measurements[0])  # M -> c[0]
        self.circuit.measure(self.alice_entangled, self.alice_measurements[1])  # A -> c[1]

        if self.use_barriers:
            self.circuit.barrier()

    def _apply_corrections(self):
        """Apply conditional corrections based on Alice's measurement results.

        Following IBM's standard teleportation protocol:
        - Apply X correction if Alice's A qubit measurement (c[1]) is 1
        - Apply Z correction if Alice's M qubit measurement (c[0]) is 1
        """
        # Conditional gates using Qiskit v2.0 if_test syntax
        # Apply X correction if Alice's A qubit measurement (c[1]) is 1
        with self.circuit.if_test((self.alice_measurements[1], 1)):
            self.circuit.x(self.bob_entangled[0])

        # Apply Z correction if Alice's M qubit measurement (c[0]) is 1
        with self.circuit.if_test((self.alice_measurements[0], 1)):
            self.circuit.z(self.bob_entangled[0])


class VariationTeleportationProtocol(BaseTeleportation):
    """Variation teleportation protocol with fixed corrections.

    This implements a variation of the teleportation protocol where the corrections
    applied to Bob's qubit are fixed (CX and CZ operations) rather than being
    dependent on Alice's measurement results. This creates a different quantum
    algorithm that can be used for testing and analysis purposes.
    """

    def _apply_corrections(self):
        """Apply fixed corrections (variation from standard teleportation)."""
        # Instead of measurement-dependent corrections, we apply fixed operations
        self.circuit.cx(self.alice_entangled, self.bob_entangled)
        self.circuit.cz(self.message_qubit, self.bob_entangled)


class TeleportationCircuitGenerator:
    """Circuit generator for testing teleportation protocols.

    This class generates quantum circuits that test teleportation protocols by:
    1. Creating a payload of quantum gates applied to auxiliary qubits
    2. Using the teleportation protocol to transfer the quantum state
    3. Applying inverse operations to validate the transfer

    The generated circuit can be executed using the QuantumCircuitExecutor class
    for comprehensive analysis with qward metrics and visualization.

    This pattern can be extended to other quantum algorithm families for
    systematic testing and analysis.

    Args:
        protocol: Optional protocol instance to use (overrides protocol_type)
        payload_size: Number of auxiliary qubits to use for testing
        gates: List of gate names or number of random gates to apply
        use_barriers: Whether to add barriers for visualization
        save_statevector: Whether to save intermediate statevectors
        protocol_type: 'variation' or 'standard' teleportation protocol (ignored if protocol is provided)
    """

    def __init__(
        self,
        protocol=None,
        *,
        payload_size: int = 3,
        gates: list | int = None,
        use_barriers: bool = True,
        save_statevector: bool = False,
        protocol_type: str = "variation",
    ):
        self.gates: dict = {}
        self.payload_size = payload_size
        self.use_barriers = use_barriers
        self.save_statevector = save_statevector
        self.gate_types = {
            "x": lambda: QuantumGate("x"),  # Pauli-X (self-inverse)
            "y": lambda: QuantumGate("y"),  # Pauli-Y (self-inverse)
            "z": lambda: QuantumGate("z"),  # Pauli-Z (self-inverse)
            "h": lambda: QuantumGate("h"),  # Hadamard (self-inverse)
            "s": lambda: QuantumGate("s"),  # Phase gate
            "sdg": lambda: QuantumGate("sdg"),  # S-dagger gate (inverse of S)
        }

        # Handle gates parameter
        if isinstance(gates, int):
            # Generate N random gates if gates is a number
            available_gates = list(self.gate_types.keys())
            self.input_gates = [random.choice(available_gates) for _ in range(gates)]
        elif gates is None:
            # Generate 1 random gate per qubit if no gates provided
            available_gates = list(self.gate_types.keys())
            self.input_gates = [random.choice(available_gates) for _ in range(payload_size)]
        else:
            # Use provided gates list
            self.input_gates = gates

        self.auxiliary_qubits = QuantumRegister(payload_size, "R")

        # Select protocol - use provided protocol or create based on type
        if protocol is not None:
            self.protocol = protocol
        elif protocol_type.lower() == "standard":
            self.protocol = StandardTeleportationProtocol(use_barriers=use_barriers)
        elif protocol_type.lower() == "variation":
            self.protocol = VariationTeleportationProtocol(use_barriers=use_barriers)
        else:
            raise ValueError(
                f"Unknown protocol_type: {protocol_type}. Use 'standard' or 'variation'"
            )

        self.result = ClassicalRegister(payload_size, "test_result")
        self.circuit = self._create_test_circuit()

    def _create_test_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(
            self.auxiliary_qubits,
            self.protocol.message_qubit,
            self.protocol.alice_entangled,
            self.protocol.bob_entangled,
        )

        self._create_payload(circuit)
        if self.save_statevector:
            circuit.save_statevector(label="after_payload")  # pylint: disable=no-member
        if self.use_barriers and not self.save_statevector:
            circuit.barrier()

        circuit = circuit.compose(
            self.protocol.circuit, qubits=range(self.payload_size, self.payload_size + 3)
        )
        if self.use_barriers and not self.save_statevector:
            circuit.barrier()

        if self.save_statevector:
            circuit.save_statevector(label="before_validation")  # pylint: disable=no-member

        self._create_validation(circuit)
        if self.save_statevector:
            circuit.save_statevector(label="after_validation")  # pylint: disable=no-member
        if self.use_barriers and not self.save_statevector:
            circuit.barrier()

        circuit.add_register(self.result)
        circuit.measure(self.auxiliary_qubits, self.result)
        # circuit.measure_all()

        return circuit

    def _create_payload(self, circuit: QuantumCircuit):
        # First apply initial operations to all qubits
        for qubit in self.auxiliary_qubits:
            circuit.h(qubit)
            circuit.cx(qubit, self.protocol.message_qubit)

        # Calculate gates per qubit
        gates_per_qubit = len(self.input_gates) // self.payload_size
        remaining_gates = len(self.input_gates) % self.payload_size

        # Distribute gates across qubits
        gate_index = 0
        for i, qubit in enumerate(self.auxiliary_qubits):
            # Calculate how many gates this qubit should get
            num_gates = gates_per_qubit + (1 if i < remaining_gates else 0)

            # Apply the gates assigned to this qubit
            qubit_gates = []
            for _ in range(num_gates):
                if gate_index < len(self.input_gates):
                    gate_name = self.input_gates[gate_index]
                    if gate_name in self.gate_types:
                        gate = self.gate_types[gate_name]()
                        qubit_gates.append(gate)
                        gate.apply(circuit, qubit)
                    gate_index += 1

            if qubit_gates:
                self.gates[qubit] = qubit_gates if len(qubit_gates) > 1 else qubit_gates[0]

    def _create_validation(self, circuit: QuantumCircuit):
        for qubit in reversed(self.auxiliary_qubits):
            # Get gates for this qubit if any were applied
            if qubit in self.gates:
                gates = self.gates[qubit]
                if isinstance(gates, list):
                    # Apply conjugates of all gates in reverse order
                    for gate in reversed(gates):
                        gate.apply_conjugate(circuit, qubit)
                else:
                    # Single gate case
                    gates.apply_conjugate(circuit, qubit)

            # Apply the inverse of the initial operations
            circuit.cx(qubit, self.protocol.bob_entangled)
            circuit.h(qubit)

    def draw(self):
        return self.circuit.draw(output="mpl")
