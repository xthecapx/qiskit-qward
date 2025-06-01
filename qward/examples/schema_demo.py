"""
Schema-Based Metrics Demo for QWARD.

This demo showcases the schema-based validation functionality inspired by
dataframely's approach to data validation and documentation.

Key features demonstrated:
- Traditional dictionary vs schema-based approaches
- Automatic validation and type checking
- JSON schema generation for documentation
- IDE support and developer experience improvements
"""

from typing import TYPE_CHECKING, Dict, Any
import json

from qiskit import QuantumCircuit
from qward.metrics.qiskit_metrics import QiskitMetrics

if TYPE_CHECKING:
    from qward.metrics.schemas import QiskitMetricsSchema


def create_example_circuit() -> QuantumCircuit:
    """Create a simple quantum circuit for demonstration."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier()
    circuit.measure_all()
    return circuit


def demo_traditional_approach() -> Dict[str, Any]:
    """Demonstrate traditional dictionary-based metrics."""
    print("=" * 60)
    print("Traditional Dictionary-Based Approach")
    print("=" * 60)

    circuit = create_example_circuit()
    qiskit_metrics = QiskitMetrics(circuit)

    # New API - get_metrics() returns schema object
    schema_metrics = qiskit_metrics.get_metrics()

    # Convert to traditional flattened dictionary for backward compatibility
    traditional_metrics = schema_metrics.to_flat_dict()

    print("✅ Traditional metrics (flattened dictionary):")
    print(f"   Type: {type(traditional_metrics)}")
    print(f"   Keys: {list(traditional_metrics.keys())[:5]}...")
    print(f"   Depth: {traditional_metrics['basic_metrics.depth']}")
    print(f"   Qubits: {traditional_metrics['basic_metrics.num_qubits']}")

    print("\n❌ Issues with traditional approach:")
    print("   - No type hints for IDE support")
    print("   - No validation of data constraints")
    print("   - Unclear what fields are available")
    print("   - Easy to make typos in key names")

    # Demonstrate potential issues
    typo_result = traditional_metrics.get("basic_metrics.depht")  # Typo!
    print(f"   - Typo example: {typo_result} (should be None due to typo)")

    return traditional_metrics


def demo_schema_approach() -> "QiskitMetricsSchema":
    """Demonstrate schema-based metrics with validation."""
    print("\n" + "=" * 60)
    print("New Schema-Based Approach")
    print("=" * 60)

    circuit = create_example_circuit()
    qiskit_metrics = QiskitMetrics(circuit)

    try:
        # Schema-based approach - get_metrics() now returns structured, validated object
        schema_metrics = qiskit_metrics.get_metrics()

        print("✓ Schema-based metrics (structured object):")
        print(f"   Type: {type(schema_metrics)}")
        print(f"   Depth: {schema_metrics.basic_metrics.depth}")
        print(f"   Qubits: {schema_metrics.basic_metrics.num_qubits}")
        print(f"   Operations: {schema_metrics.basic_metrics.count_ops}")

        print("\n✅ Benefits of schema approach:")
        print("   - Full type hints and IDE autocomplete")
        print("   - Automatic validation of data types and constraints")
        print("   - Clear documentation of all available fields")
        print("   - Compile-time error detection")

        return schema_metrics

    except ImportError:
        print("❌ Pydantic not available - install pydantic for schema validation")
        return None


def demo_validation_features() -> None:
    """Demonstrate schema validation capabilities."""
    print("\n🔍 Schema validation in action:")

    try:
        from qward.metrics.schemas import BasicMetricsSchema

        print("   - Schema provides automatic validation for:")
        print("   - Depth field: Circuit depth (number of time steps)")
        print("   - Width field: Circuit width (total qubits and classical bits)")
        print("   - Testing validation with invalid data...")

        # Test validation with invalid data
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
        except Exception as e:
            print(f"   ✅ Validation caught invalid data: {e}")

    except ImportError:
        print("   ❌ Pydantic not available for validation demo")


def demo_conversion_capabilities() -> None:
    """Demonstrate conversion between schema and dictionary formats."""
    print("\n🔄 Conversion capabilities:")

    circuit = create_example_circuit()
    qiskit_metrics = QiskitMetrics(circuit)

    try:
        schema_metrics = qiskit_metrics.get_metrics()

        # Convert to flat dictionary (for DataFrame compatibility)
        flat_dict = schema_metrics.to_flat_dict()
        print(f"   - Can convert to flat dict: {len(flat_dict)} keys")
        print(f"   - Sample keys: {list(flat_dict.keys())[:3]}...")

        # Test round-trip conversion
        from qward.metrics.schemas import QiskitMetricsSchema

        reconstructed = QiskitMetricsSchema.from_flat_dict(flat_dict)

        # Compare original and reconstructed
        original_depth = schema_metrics.basic_metrics.depth
        reconstructed_depth = reconstructed.basic_metrics.depth
        round_trip_works = original_depth == reconstructed_depth

        print(f"   - Round-trip conversion works: {round_trip_works}")

    except ImportError:
        print("   ❌ Pydantic not available for conversion demo")


def demo_json_schema_generation() -> None:
    """Demonstrate JSON schema generation for API documentation."""
    print("\n" + "=" * 60)
    print("JSON Schema Generation for Documentation")
    print("=" * 60)

    try:
        from qward.metrics.schemas import BasicMetricsSchema

        # Generate JSON schema
        json_schema = BasicMetricsSchema.model_json_schema()

        print("✅ Generated JSON schema for basic metrics:")
        print(f"   - Title: {json_schema.get('title', 'N/A')}")
        print(f"   - Properties: {list(json_schema.get('properties', {}).keys())}")

        # Show specific field constraints
        properties = json_schema.get("properties", {})
        depth_schema = properties.get("depth", {})
        print(f"   - Depth field minimum: {depth_schema.get('minimum', 'N/A')}")
        print(f"   - Depth field type: {depth_schema.get('type', 'N/A')}")

        print("   - Note: Full schema generation works for schemas without complex objects")
        print("   - Complex objects like CircuitInstruction require custom serialization")

    except ImportError:
        print("❌ Pydantic not available for JSON schema generation")


def main() -> None:
    """Run the complete schema demo."""
    print("QWARD Schema-Based Metrics Demo")
    print("Inspired by dataframely's approach to data validation")
    print("=" * 60)

    # Run all demonstrations
    demo_traditional_approach()
    demo_schema_approach()
    demo_validation_features()
    demo_conversion_capabilities()
    demo_json_schema_generation()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("The schema-based approach provides:")
    print("✅ Type safety with full IDE support")
    print("✅ Automatic data validation")
    print("✅ Clear documentation of data structures")
    print("✅ JSON schema generation for API docs")
    print("✅ Backward compatibility with existing code")
    print("\nThis makes QWARD more robust and user-friendly!")


if __name__ == "__main__":
    main()
