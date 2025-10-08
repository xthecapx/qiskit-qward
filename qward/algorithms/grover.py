from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import math


class GroverOracle:
    """Grover oracle for marking specific quantum states.

    This class builds a Grover oracle for multiple marked states based on the
    IBM Quantum tutorial implementation. The oracle marks target states by
    applying a phase flip (-1) to them.

    Args:
        marked_states: List of marked states (bit strings) to search for
    """

    def __init__(self, marked_states):
        if not isinstance(marked_states, list):
            marked_states = [marked_states]

        self.marked_states = marked_states
        self.num_qubits = len(marked_states[0])

        # Validate that all marked states have the same length
        if not all(len(state) == self.num_qubits for state in marked_states):
            raise ValueError("All marked states must have the same number of bits")

        # Create the oracle circuit
        self.circuit = self._build_oracle()

    def _build_oracle(self):
        """Build the oracle circuit that marks the target states."""
        qc = QuantumCircuit(self.num_qubits)

        # Mark each target state in the input list
        for target in self.marked_states:
            # Flip target bit-string to match Qiskit bit-ordering
            rev_target = target[::-1]
            # Find the indices of all the '0' elements in bit-string
            zero_inds = [ind for ind in range(self.num_qubits) if rev_target.startswith("0", ind)]
            # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
            # where the target bit-string has a '0' entry
            if zero_inds:
                qc.x(zero_inds)
            qc.compose(MCMTGate(ZGate(), self.num_qubits - 1, 1), inplace=True)
            if zero_inds:
                qc.x(zero_inds)
        return qc

    def draw(self):
        """Draw the oracle circuit."""
        return self.circuit.draw(output="mpl")

    def get_circuit(self):
        """Get the oracle quantum circuit."""
        return self.circuit


class Grover:
    """Grover's algorithm implementation.

    This class implements Grover's quantum search algorithm following the pattern
    established in the teleportation protocols. It creates a complete Grover circuit
    with oracle, superposition initialization, and optimal number of iterations.

    Based on the IBM Quantum tutorial: https://quantum.cloud.ibm.com/docs/en/tutorials/grovers-algorithm

    Args:
        marked_states: List of marked states (bit strings) to search for
        use_barriers: Whether to add barriers for visualization
    """

    def __init__(self, marked_states, use_barriers: bool = True):
        if marked_states is None:
            raise ValueError("marked_states must be provided and cannot be None")

        if not marked_states:
            raise ValueError("marked_states cannot be empty")

        if not isinstance(marked_states, list):
            marked_states = [marked_states]

        self.marked_states = marked_states
        self.use_barriers = use_barriers
        self.num_qubits = len(marked_states[0])

        # Validate that all marked states have the same length
        if not all(len(state) == self.num_qubits for state in marked_states):
            raise ValueError("All marked states must have the same number of bits")

        # Create oracle from marked states
        self.oracle = GroverOracle(marked_states)

        # Create Grover operator
        self.grover_op = grover_operator(self.oracle.circuit)

        # Calculate optimal number of iterations
        self.optimal_iterations = math.floor(
            math.pi / (4 * math.asin(math.sqrt(len(self.marked_states) / 2**self.num_qubits)))
        )

        # Create the complete circuit
        self.circuit = self._create_grover_circuit()

    def _create_grover_circuit(self):
        """Create the complete Grover circuit following IBM tutorial pattern."""
        # Create circuit with same number of qubits as the Grover operator
        circuit = QuantumCircuit(self.grover_op.num_qubits)

        # Step 1: Create even superposition of all basis states
        circuit.h(range(self.grover_op.num_qubits))

        if self.use_barriers:
            circuit.barrier()

        # Step 2: Apply Grover operator the optimal number of times
        if self.optimal_iterations > 0:
            circuit.compose(self.grover_op.power(self.optimal_iterations), inplace=True)

        if self.use_barriers:
            circuit.barrier()

        # Step 3: Measure all qubits
        circuit.measure_all()

        return circuit

    def draw(self):
        """Draw the circuit."""
        return self.circuit.draw(output="mpl")

    def get_oracle(self):
        """Get the oracle instance."""
        return self.oracle

    def get_oracle_circuit(self):
        """Get the oracle quantum circuit."""
        return self.oracle.circuit

    def get_grover_operator(self):
        """Get the Grover operator circuit."""
        return self.grover_op

    def success_criteria(self, outcome: str) -> bool:
        """Success criteria function for use with QuantumCircuitExecutor.

        This function determines if a single measurement outcome represents
        a successful Grover algorithm execution by checking if the outcome
        matches any of the marked states.

        Args:
            outcome: Single measurement outcome string (e.g., "011", "100")

        Returns:
            bool: True if the outcome is one of the marked states
        """
        # Clean the outcome string (remove spaces)
        clean_outcome = outcome.replace(" ", "")

        # Check if this outcome is one of our marked states
        return clean_outcome in self.marked_states

    def get_theoretical_success_probability(self) -> float:
        """Calculate the theoretical success probability for the current configuration.

        Returns:
            float: Theoretical probability of measuring a marked state
        """
        num_marked = len(self.marked_states)
        total_states = 2**self.num_qubits

        # Theoretical probability after optimal iterations
        theta = math.asin(math.sqrt(num_marked / total_states))
        prob = math.sin((2 * self.optimal_iterations + 1) * theta) ** 2
        return prob

    def create_isa_circuit(self, backend=None, optimization_level=3):
        """Create ISA (Instruction Set Architecture) circuit optimized for execution.

        Following the IBM Quantum tutorial pattern, this method transpiles the Grover circuit
        for optimal execution using a backend target.

        Args:
            backend: Target backend (if None, uses AerSimulator as default)
            optimization_level: Transpiler optimization level (0-3, default: 3)

        Returns:
            QuantumCircuit: Transpiled ISA circuit
        """
        # Use provided backend or default to AerSimulator
        if backend is None:
            backend = AerSimulator()

        # Get the target from the backend
        target = backend.target
        pm = generate_preset_pass_manager(target=target, optimization_level=optimization_level)
        circuit_isa = pm.run(self.circuit)
        return circuit_isa

    def create_rigetti_isa_circuit(self, optimization_level=3):
        """Create ISA circuit specifically optimized for Rigetti quantum devices and qBraid compatibility.

        This method creates a circuit that is compatible with qBraid's transpiler by using
        only basic gates that can be reliably converted from Qiskit to Braket format.

        Args:
            optimization_level: Transpiler optimization level (0-3, default: 1 for maximum compatibility)

        Returns:
            QuantumCircuit: Rigetti-optimized ISA circuit compatible with qBraid
        """
        from qiskit.transpiler import CouplingMap

        try:
            # Create a conservative transpilation for qBraid compatibility
            # Use only the most basic gates that qBraid can handle reliably

            # Create a simple linear coupling map for Rigetti-like connectivity
            num_qubits = self.num_qubits
            coupling_list = []

            # Create a linear coupling map as a starting point
            for i in range(num_qubits - 1):
                coupling_list.append([i, i + 1])
                coupling_list.append([i + 1, i])  # Bidirectional

            # coupling_map = CouplingMap(coupling_list)

            # Create pass manager with very conservative settings for qBraid compatibility
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level,  # Use level 1 for better compatibility
                # coupling_map=coupling_map,
                # Use only the most basic gates that qBraid supports well
                basis_gates=["h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "measure", "reset"],
                seed_transpiler=42,  # For reproducible results
                # Additional settings to avoid problematic gates
                # translation_method='translator',  # Use translator instead of synthesis
                # approximation_degree=0.99999  # High fidelity approximation
            )

            circuit_isa = pm.run(self.circuit)

            display(circuit_isa.draw(output="mpl"))

            # Additional cleanup: decompose any remaining composite gates
            from qiskit.transpiler.passes import Decompose
            from qiskit.transpiler import PassManager

            # Create a cleanup pass to decompose any remaining problematic gates
            cleanup_pm = PassManager(
                [Decompose(["u1", "u2", "u3"])]  # Decompose U gates that cause qBraid issues
            )

            circuit_isa = cleanup_pm.run(circuit_isa)

            return circuit_isa

        except Exception as e:
            print(
                f"Warning: Rigetti-specific transpilation failed ({e}), falling back to standard transpilation"
            )
            # Fallback to standard transpilation
            return self.create_isa_circuit(optimization_level=optimization_level)

    def expected_distribution(self) -> dict:
        """Get the theoretical expected probability distribution for this Grover instance.

        For optimal Grover iterations, the marked states should have high probability
        while unmarked states should have low probability.

        Returns:
            dict: Expected probability distribution for marked states
        """
        num_marked = len(self.marked_states)
        total_states = 2**self.num_qubits

        # Calculate theoretical success probability
        theoretical_prob = self.get_theoretical_success_probability()

        # For multiple marked states, distribute probability equally among them
        prob_per_marked_state = theoretical_prob / num_marked

        # Create expected distribution
        expected_dist = {}

        # Add marked states with their expected probabilities
        for state in self.marked_states:
            expected_dist[state] = prob_per_marked_state

        # Add unmarked states with remaining probability
        remaining_prob = 1.0 - theoretical_prob
        unmarked_states = total_states - num_marked

        if unmarked_states > 0:
            prob_per_unmarked_state = remaining_prob / unmarked_states

            # Generate all possible states and add unmarked ones
            for i in range(total_states):
                state = format(i, f"0{self.num_qubits}b")
                if state not in self.marked_states:
                    expected_dist[state] = prob_per_unmarked_state

        return expected_dist


class GroverCircuitGenerator:
    """Circuit generator for testing Grover's algorithm.

    This class generates quantum circuits that implement Grover's search algorithm
    for systematic testing and analysis with qward metrics and visualization.

    Following the same pattern as TeleportationCircuitGenerator, this allows for
    comprehensive analysis of quantum search algorithms.

    Args:
        marked_states: List of marked states to search for (default: ["011", "100"])
        use_barriers: Whether to add barriers for visualization
        save_statevector: Whether to save intermediate statevectors
    """

    def __init__(
        self,
        marked_states=None,
        *,
        use_barriers: bool = True,
        save_statevector: bool = False,
    ):
        if marked_states is None:
            marked_states = ["011", "100"]

        self.marked_states = marked_states
        self.use_barriers = use_barriers
        self.save_statevector = save_statevector

        # Create the Grover algorithm
        self.grover = Grover(marked_states, use_barriers=use_barriers)
        self.num_qubits = self.grover.num_qubits

        # Direct access to the circuit
        self.circuit = self.grover.circuit

        # Create and expose ISA circuit (standard)
        self.circuit_isa = self.grover.create_isa_circuit()

        # Create and expose Rigetti-optimized ISA circuit
        self.circuit_isa_rigetti = self.grover.create_rigetti_isa_circuit()

    def draw(self):
        """Draw the circuit."""
        return self.circuit.draw(output="mpl")

    def get_marked_states(self):
        """Get the marked states being searched for."""
        return self.marked_states

    def get_oracle(self):
        """Get the oracle instance."""
        return self.grover.oracle

    def get_success_probability(self):
        """Calculate theoretical success probability."""
        return self.grover.get_theoretical_success_probability()

    def get_rigetti_circuit(self):
        """Get the Rigetti-optimized ISA circuit."""
        return self.circuit_isa_rigetti

    def draw_rigetti_circuit(self):
        """Draw the Rigetti-optimized ISA circuit."""
        return self.circuit_isa_rigetti.draw(output="mpl")
