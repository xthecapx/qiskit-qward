"""
Example demonstrating Scanner metric strategies shortcut via constructor.
"""

from qward.examples.utils import get_display, create_example_circuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qiskit import QuantumCircuit

display = get_display()


def example_strategies_with_classes():
    circuit = create_example_circuit()
    print("\nExample: Strategies via Scanner constructor (strategy classes)")
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def example_strategies_with_instances():
    circuit = create_example_circuit()
    print("\nExample: Strategies via Scanner constructor (strategy instances)")
    qm = QiskitMetrics(circuit)
    cm = ComplexityMetrics(circuit)
    scanner = Scanner(circuit=circuit, strategies=[qm, cm])
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def example_strategies_with_instance_mismatch():
    circuit = create_example_circuit()
    print("\nExample: Strategies via Scanner constructor (instance with different circuit)")
    circuit2 = QuantumCircuit(2, 2)
    circuit2.x(0)
    qm_diff = QiskitMetrics(circuit2)
    try:
        Scanner(circuit=circuit, strategies=[qm_diff])
    except ValueError as e:
        print("Expected error (different circuit):", e)


def example_add_strategy_method():
    circuit = create_example_circuit()
    print("\nExample: Adding strategies manually with add_strategy method")
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    metrics_dict = scanner.calculate_metrics()
    for metric_name, df in metrics_dict.items():
        print(f"{metric_name} DataFrame:")
        display(df)


def main():
    example_strategies_with_classes()
    example_strategies_with_instances()
    example_strategies_with_instance_mismatch()
    example_add_strategy_method()


if __name__ == "__main__":
    main()
