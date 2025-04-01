from qiskit_qward.scanning_quantum_circuit import ScanningQuantumCircuit
from qiskit_qward.analysis.success_rate import SuccessRate


class QuantumEnigmaValidator(ScanningQuantumCircuit):
    def __init__(self):
        # Initialize with named registers
        super().__init__(num_qubits=3, num_clbits=3, name="quantum_enigma")

        # Define success criteria: both guardians point to the same door
        # and that door is the one NOT to open (the door without the treasure)
        def success_criteria(state):
            # print(f"Analyzing state: {state}")  # Debug print
            lie_qubit = int(state[0])  # Which guardian is lying (0=q0, 1=q1)
            q1_guardian = int(state[1])  # q1 guardian's answer (0=right door, 1=left door)
            q0_guardian = int(state[2])  # q0 guardian's answer (0=right door, 1=left door)

            # Both guardians should point to the same door
            if q0_guardian != q1_guardian:
                print(f"  Failed: Guardians point to different doors")
                return False

            result = q1_guardian == q0_guardian
            return result

        # Add the success rate analysis with the criteria
        success_analyzer = SuccessRate()
        success_analyzer.set_success_criteria(success_criteria)
        self.add_analyzer(success_analyzer)

        # Setup the circuit
        self._setup_circuit()

    def _setup_circuit(self):
        # Place treasure randomly using Hadamard gate on q0
        self.h(0)  # q0: Right guardian's knowledge

        # Ensure both guardians know the same thing using CNOT
        self.cx(0, 1)  # q1: Left guardian's knowledge

        if self.use_barriers:
            self.barrier()

        # Add lie qubit in superposition
        self.h(2)  # q2: Which guardian is lying (0=right, 1=left)

        # First liar detection step
        self.cx(2, 1)  # If q2 is 1, left guardian lies
        self.x(2)  # Flip the lie qubit
        self.cx(2, 0)  # If q2 is 0, right guardian lies
        self.x(2)  # Flip the lie qubit back

        if self.use_barriers:
            self.barrier()

        # Which door would the other guardian tell me not to open?
        # Apply the "what would the other guardian say" logic
        # First, swap the guardians' knowledge
        self.swap(0, 1)  # Swap right and left guardian knowledge

        # Apply NOT gates to represent "not to open"
        self.x(0)  # NOT on right guardian's answer
        self.x(1)  # NOT on left guardian's answer

        if self.use_barriers:
            self.barrier()

        # Second liar detection step
        self.cx(2, 1)  # If q2 is 1, left guardian lies
        self.x(2)  # Flip the lie qubit
        self.cx(2, 0)  # If q2 is 0, right guardian lies
        self.x(2)  # Flip the lie qubit back

        # Measure all qubits
        self.measure([0, 1, 2], [0, 1, 2])
