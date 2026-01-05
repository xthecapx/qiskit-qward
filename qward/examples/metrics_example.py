#!/usr/bin/env python3
"""
QWARD Metrics Example

This example demonstrates:
1. Schema-based metrics with validation
2. Different metric types (Qiskit, Complexity, etc.)
3. Scanner strategy patterns
4. Metrics conversion and serialization
"""

from typing import Dict, Any
import json

from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics
from qward.metrics.complexity_metrics import ComplexityMetrics as ComplexityMetricsClass


def create_test_circuit() -> QuantumCircuit:
    """Create a test quantum circuit with various gate types."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.t(1)
    qc.cx(0, 1)
    qc.ry(0.5, 2)
    qc.cz(1, 2)
    qc.swap(2, 3)
    qc.barrier()
    qc.measure_all()
    return qc


# =============================================================================
# Example 1: Schema-Based Metrics
# =============================================================================


def schema_based_metrics_example():
    """Demonstrate schema-based metrics with validation."""
    print("=== Example 1: Schema-Based Metrics ===\n")

    circuit = create_test_circuit()
    qiskit_metrics = QiskitMetrics(circuit)

    # Get schema-based metrics (structured object)
    schema_metrics = qiskit_metrics.get_metrics()

    print("Schema-based metrics (structured object):")
    print(f"  Type: {type(schema_metrics)}")
    print(f"  Depth: {schema_metrics.basic_metrics.depth}")
    print(f"  Qubits: {schema_metrics.basic_metrics.num_qubits}")
    print(f"  Operations: {schema_metrics.basic_metrics.count_ops}")

    print("\n✓ Benefits of schema approach:")
    print("  • Full type hints and IDE autocomplete")
    print("  • Automatic validation of data types")
    print("  • Clear documentation of available fields")

    # Convert to flat dictionary for DataFrame compatibility
    flat_dict = schema_metrics.to_flat_dict()
    print(f"\nFlat dictionary has {len(flat_dict)} keys")
    print(f"Sample keys: {list(flat_dict.keys())[:3]}...")


def validation_example():
    """Demonstrate schema validation capabilities."""
    print("\n=== Example 2: Schema Validation ===\n")

    try:
        from qward.metrics.schemas import BasicMetricsSchema

        print("Testing validation with invalid data...")

        try:
            BasicMetricsSchema(
                depth=-1,  # Invalid - should be >= 0
                width=2,
                size=5,
                num_qubits=2,
                num_clbits=2,
                num_ancillas=0,
                num_parameters=0,
                has_calibrations=False,
                has_layout=False,
                count_ops={},
            )
            print("  ✗ Validation should have caught invalid data")
        except Exception as e:
            print(f"  ✓ Validation caught invalid data: {type(e).__name__}")

    except ImportError:
        print("  Pydantic not available for validation demo")


def json_schema_example():
    """Demonstrate JSON schema generation for documentation."""
    print("\n=== Example 3: JSON Schema Generation ===\n")

    try:
        from qward.metrics.schemas import BasicMetricsSchema

        json_schema = BasicMetricsSchema.model_json_schema()

        print("Generated JSON schema for BasicMetrics:")
        print(f"  Title: {json_schema.get('title', 'N/A')}")
        print(f"  Properties: {list(json_schema.get('properties', {}).keys())}")

        properties = json_schema.get("properties", {})
        depth_schema = properties.get("depth", {})
        print(f"  Depth field type: {depth_schema.get('type', 'N/A')}")
        print(f"  Depth field minimum: {depth_schema.get('minimum', 'N/A')}")

    except ImportError:
        print("  Pydantic not available for JSON schema generation")


# =============================================================================
# Example 2: Complexity Metrics
# =============================================================================


def complexity_metrics_example():
    """Demonstrate complexity metrics calculation."""
    print("\n=== Example 4: Complexity Metrics ===\n")

    circuit = create_test_circuit()
    print(f"Circuit: {circuit.size()} gates, depth {circuit.depth()}, {circuit.num_qubits} qubits")

    metrics = ComplexityMetricsClass(circuit)

    # Get individual metric types
    gate_metrics = metrics.get_gate_based_metrics()
    print(f"\nGate-based metrics:")
    print(f"  Gate count: {gate_metrics.gate_count}")
    print(f"  Circuit depth: {gate_metrics.circuit_depth}")
    print(f"  CNOT count: {gate_metrics.cnot_count}")
    print(f"  Two-qubit count: {gate_metrics.two_qubit_count}")

    advanced_metrics = metrics.get_advanced_metrics()
    print(f"\nAdvanced metrics:")
    print(f"  Parallelism factor: {advanced_metrics.parallelism_factor:.3f}")
    print(f"  Circuit efficiency: {advanced_metrics.circuit_efficiency:.3f}")

    # Get complete metrics schema
    complete = metrics.get_metrics()
    print(f"\nComplete metrics type: {type(complete)}")

    # JSON serialization
    print(f"\nJSON serialization available: {hasattr(advanced_metrics, 'model_dump_json')}")


# =============================================================================
# Example 3: Scanner Strategies
# =============================================================================


def scanner_strategies_example():
    """Demonstrate different Scanner strategy patterns."""
    print("\n=== Example 5: Scanner Strategy Patterns ===\n")

    circuit = create_test_circuit()

    # Pattern 1: Pass strategy classes to constructor
    print("Pattern 1: Strategy classes in constructor")
    scanner1 = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    metrics1 = scanner1.calculate_metrics()
    print(f"  Calculated {len(metrics1)} metric types")

    # Pattern 2: Pass strategy instances
    print("\nPattern 2: Strategy instances in constructor")
    qm = QiskitMetrics(circuit)
    cm = ComplexityMetrics(circuit)
    scanner2 = Scanner(circuit=circuit, strategies=[qm, cm])
    metrics2 = scanner2.calculate_metrics()
    print(f"  Calculated {len(metrics2)} metric types")

    # Pattern 3: Add strategies manually
    print("\nPattern 3: Add strategies with add_strategy()")
    scanner3 = Scanner(circuit=circuit)
    scanner3.add_strategy(QiskitMetrics(circuit))
    scanner3.add_strategy(ComplexityMetrics(circuit))
    metrics3 = scanner3.calculate_metrics()
    print(f"  Calculated {len(metrics3)} metric types")

    # Display summary
    print("\nScanner Summary:")
    scanner1.display_summary(metrics1)


def circuit_mismatch_example():
    """Demonstrate circuit mismatch error handling."""
    print("\n=== Example 6: Circuit Mismatch Handling ===\n")

    circuit1 = create_test_circuit()
    circuit2 = QuantumCircuit(2)
    circuit2.x(0)

    # Create strategy with different circuit
    qm_diff = QiskitMetrics(circuit2)

    try:
        Scanner(circuit=circuit1, strategies=[qm_diff])
        print("  ✗ Should have raised error")
    except ValueError as e:
        print(f"  ✓ Correctly caught circuit mismatch: {type(e).__name__}")


# =============================================================================
# Example 4: Conversion Capabilities
# =============================================================================


def conversion_example():
    """Demonstrate conversion between schema and dictionary formats."""
    print("\n=== Example 7: Format Conversions ===\n")

    circuit = create_test_circuit()
    qiskit_metrics = QiskitMetrics(circuit)

    # Get schema metrics
    schema_metrics = qiskit_metrics.get_metrics()

    # Convert to flat dictionary
    flat_dict = schema_metrics.to_flat_dict()
    print(f"Flat dictionary: {len(flat_dict)} keys")

    # Round-trip conversion
    try:
        from qward.metrics.schemas import QiskitMetricsSchema

        reconstructed = QiskitMetricsSchema.from_flat_dict(flat_dict)
        original_depth = schema_metrics.basic_metrics.depth
        reconstructed_depth = reconstructed.basic_metrics.depth

        print(f"Round-trip conversion works: {original_depth == reconstructed_depth}")

    except ImportError:
        print("  Pydantic not available for round-trip demo")


def main():
    """Run all metrics examples."""
    print("🔬 QWARD Metrics Examples")
    print("=" * 50)

    # Schema examples
    schema_based_metrics_example()
    validation_example()
    json_schema_example()

    # Complexity metrics
    complexity_metrics_example()

    # Scanner patterns
    scanner_strategies_example()
    circuit_mismatch_example()

    # Conversions
    conversion_example()

    print("\n" + "=" * 50)
    print("✅ Metrics examples complete!")
    print("\nKey concepts demonstrated:")
    print("  • Schema-based metrics with validation")
    print("  • Complexity metrics calculation")
    print("  • Scanner strategy patterns")
    print("  • Format conversions (schema ↔ dict)")
    print("  • JSON schema generation")


if __name__ == "__main__":
    main()

