"""
Example demonstrating Scanner metrics shortcut via constructor.
"""

from typing import Any, Callable
from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics

try:
    from IPython.display import display as ipython_display

    display: Callable[..., Any] = ipython_display
except ImportError:
    display = print


def create_example_circuit():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def example_metrics_with_classes():
    circuit = create_example_circuit()
    print("\nExample: Metrics via Scanner constructor (metric classes)")
    scanner = Scanner(circuit=circuit, metrics=[QiskitMetrics, ComplexityMetrics])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def example_metrics_with_instances():
    circuit = create_example_circuit()
    print("\nExample: Metrics via Scanner constructor (metric instances)")
    qm = QiskitMetrics(circuit)
    cm = ComplexityMetrics(circuit)
    scanner = Scanner(circuit=circuit, metrics=[qm, cm])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def example_metrics_with_instance_mismatch():
    circuit = create_example_circuit()
    print("\nExample: Metrics via Scanner constructor (instance with different circuit)")
    circuit2 = QuantumCircuit(2, 2)
    circuit2.x(0)
    qm_diff = QiskitMetrics(circuit2)
    try:
        Scanner(circuit=circuit, metrics=[qm_diff])
    except ValueError as e:
        print("Expected error (different circuit):", e)


def main():
    example_metrics_with_classes()
    example_metrics_with_instances()
    example_metrics_with_instance_mismatch()


if __name__ == "__main__":
    main()
