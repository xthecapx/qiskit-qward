"""
Demo showing QWARD's schema-based metrics with validation.

This example demonstrates how to use structured schema objects instead of
plain dictionaries for better type safety, validation, and documentation.
Inspired by dataframely's approach to data validation.
"""

from typing import Any, Dict, TYPE_CHECKING

from qiskit import QuantumCircuit
from qward.metrics import QiskitMetrics

# Import schemas with proper type checking
if TYPE_CHECKING:
    from qward.metrics.schemas import QiskitMetricsSchema, BasicMetricsSchema

try:
    from qward.metrics.schemas import QiskitMetricsSchema, BasicMetricsSchema

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


def create_example_circuit() -> QuantumCircuit:
    """Create a simple example quantum circuit for demonstration."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def demo_traditional_approach() -> None:
    """Demonstrate the traditional dictionary-based approach."""
    print("=" * 60)
    print("Traditional Dictionary-Based Approach")
    print("=" * 60)

    circuit = create_example_circuit()
    qm = QiskitMetrics(circuit)
    metrics_dict = qm.get_metrics()

    print("✅ Traditional metrics (flattened dictionary):")
    print(f"   Type: {type(metrics_dict)}")
    print(f"   Keys: {list(metrics_dict.keys())[:5]}...")
    print(f"   Depth: {metrics_dict['basic_metrics.depth']}")
    print(f"   Qubits: {metrics_dict['basic_metrics.num_qubits']}")

    print("\n❌ Issues with traditional approach:")
    print("   - No type hints for IDE support")
    print("   - No validation of data constraints")
    print("   - Unclear what fields are available")
    print("   - Easy to make typos in key names")

    # Example of potential issues
    wrong_key = metrics_dict.get("basic_metrics.depht")  # typo: 'depht' instead of 'depth'
    print(f"   - Typo example: {wrong_key} (should be None due to typo)")


def demo_schema_approach() -> None:
    """Demonstrate the new schema-based approach."""
    if not SCHEMAS_AVAILABLE:
        print("\n⚠️  Skipping schema demo - Pydantic not available")
        print("   Install pydantic to see structured metrics in action")
        return

    print("\n" + "=" * 60)
    print("New Schema-Based Approach")
    print("=" * 60)

    circuit = create_example_circuit()
    qm = QiskitMetrics(circuit)
    metrics_schema = qm.get_structured_metrics()

    print("✅ Schema-based metrics (structured object):")
    print(f"   Type: {type(metrics_schema)}")
    print(f"   Depth: {metrics_schema.basic_metrics.depth}")
    print(f"   Qubits: {metrics_schema.basic_metrics.num_qubits}")
    print(f"   Operations: {dict(metrics_schema.basic_metrics.count_ops)}")

    print("\n✅ Benefits of schema approach:")
    print("   - Full type hints and IDE autocomplete")
    print("   - Automatic validation of data types and constraints")
    print("   - Clear documentation of all available fields")
    print("   - Compile-time error detection")

    _demo_field_documentation(metrics_schema)
    _demo_validation()
    _demo_conversion_capabilities(metrics_schema)


def _demo_field_documentation(metrics_schema: "QiskitMetricsSchema") -> None:
    """Demonstrate field documentation capabilities."""
    print("\n🔍 Schema validation in action:")

    basic_schema = metrics_schema.basic_metrics
    print(f"   - Depth field: {basic_schema.model_fields['depth'].description}")
    print(f"   - Width field: {basic_schema.model_fields['width'].description}")


def _demo_validation() -> None:
    """Demonstrate automatic validation with invalid data."""
    if not SCHEMAS_AVAILABLE:
        return

    # Import here to ensure it's available when we need it
    from qward.metrics.schemas import BasicMetricsSchema

    try:
        print("   - Testing validation with invalid data...")
        invalid_data: Dict[str, Any] = {
            "depth": -1,  # Invalid: negative depth
            "width": 6,
            "size": 4,
            "num_qubits": 2,
            "num_clbits": 4,
            "num_ancillas": 0,
            "num_parameters": 0,
            "has_calibrations": False,
            "has_layout": False,
            "count_ops": {"h": 1, "cx": 1},
        }
        # This will fail validation due to negative depth
        BasicMetricsSchema(**invalid_data)
        print("   ❌ Validation should have failed!")
    except Exception as e:
        print(f"   ✅ Validation caught invalid data: {e}")


def _demo_conversion_capabilities(metrics_schema: "QiskitMetricsSchema") -> None:
    """Demonstrate conversion capabilities between formats."""
    if not SCHEMAS_AVAILABLE:
        return

    # Import here to ensure it's available when we need it
    from qward.metrics.schemas import QiskitMetricsSchema

    print("\n🔄 Conversion capabilities:")

    # Convert to flat dictionary
    flat_dict = metrics_schema.to_flat_dict()
    print(f"   - Can convert to flat dict: {len(flat_dict)} keys")
    print(f"   - Sample keys: {list(flat_dict.keys())[:3]}...")

    # Show round-trip conversion
    reconstructed = QiskitMetricsSchema.from_flat_dict(flat_dict)
    depth_matches = reconstructed.basic_metrics.depth == metrics_schema.basic_metrics.depth
    print(f"   - Round-trip conversion works: {depth_matches}")


def demo_json_schema_generation() -> None:
    """Demonstrate JSON schema generation for documentation."""
    if not SCHEMAS_AVAILABLE:
        return

    # Import here to ensure it's available when we need it
    from qward.metrics.schemas import BasicMetricsSchema

    print("\n" + "=" * 60)
    print("JSON Schema Generation for Documentation")
    print("=" * 60)

    try:
        schema = BasicMetricsSchema.model_json_schema()

        print("✅ Generated JSON schema for basic metrics:")
        print(f"   - Title: {schema.get('title', 'N/A')}")
        print(f"   - Properties: {list(schema.get('properties', {}).keys())}")

        # Show field constraints
        depth_field = schema.get("properties", {}).get("depth", {})
        print(f"   - Depth field minimum: {depth_field.get('minimum', 'N/A')}")
        print(f"   - Depth field type: {depth_field.get('type', 'N/A')}")

        print("   - Note: Full schema generation works for schemas without complex objects")
        print("   - Complex objects like CircuitInstruction require custom serialization")

    except Exception as e:
        print(f"   ❌ Schema generation failed: {e}")
        print("   - This can happen with complex objects that don't have JSON representations")


def print_summary() -> None:
    """Print the demo summary."""
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


def main() -> None:
    """Run all schema demos."""
    print("QWARD Schema-Based Metrics Demo")
    print("Inspired by dataframely's approach to data validation")

    if not SCHEMAS_AVAILABLE:
        print("\n⚠️  Pydantic schemas not available.")
        print("   Install pydantic to see full structured metrics capabilities:")
        print("   pip install pydantic")
        print("\n   Running traditional approach demo only...\n")

    # Run demos
    demo_traditional_approach()
    demo_schema_approach()
    demo_json_schema_generation()
    print_summary()


if __name__ == "__main__":
    main()
