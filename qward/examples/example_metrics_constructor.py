"""
Example demonstrating Scanner metric calculators shortcut via constructor.
"""

from qward.examples.utils import get_display, create_example_circuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qiskit import QuantumCircuit

display = get_display()


def example_calculators_with_classes():
    circuit = create_example_circuit()
    print("\nExample: Calculators via Scanner constructor (calculator classes)")
    scanner = Scanner(circuit=circuit, calculators=[QiskitMetrics, ComplexityMetrics])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def example_calculators_with_instances():
    circuit = create_example_circuit()
    print("\nExample: Calculators via Scanner constructor (calculator instances)")
    qm = QiskitMetrics(circuit)
    cm = ComplexityMetrics(circuit)
    scanner = Scanner(circuit=circuit, calculators=[qm, cm])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def example_calculators_with_instance_mismatch():
    circuit = create_example_circuit()
    print("\nExample: Calculators via Scanner constructor (instance with different circuit)")
    circuit2 = QuantumCircuit(2, 2)
    circuit2.x(0)
    qm_diff = QiskitMetrics(circuit2)
    try:
        Scanner(circuit=circuit, calculators=[qm_diff])
    except ValueError as e:
        print("Expected error (different circuit):", e)


def example_backward_compatibility():
    circuit = create_example_circuit()
    print("\nExample: Backward compatibility - using 'metrics' parameter")
    scanner = Scanner(circuit=circuit, metrics=[QiskitMetrics, ComplexityMetrics])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def main():
    example_calculators_with_classes()
    example_calculators_with_instances()
    example_calculators_with_instance_mismatch()
    example_backward_compatibility()


if __name__ == "__main__":
    main()
