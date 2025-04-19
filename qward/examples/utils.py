from typing import Any, Callable
from qiskit import QuantumCircuit


def get_display() -> Callable[..., Any]:
    """
    Returns the best display function for the environment (IPython or print).
    """
    try:
        from IPython.display import display as ipython_display

        return ipython_display
    except ImportError:
        return print


def create_example_circuit() -> QuantumCircuit:
    """
    Create a simple quantum circuit for demonstration (2-qubit GHZ state).
    Returns:
        QuantumCircuit: The circuit
    """
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit
