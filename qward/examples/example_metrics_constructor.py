"""
Example demonstrating Scanner metrics shortcut via constructor.
"""

from qward.examples.utils import get_display, create_example_circuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qiskit import QuantumCircuit

display = get_display()


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
