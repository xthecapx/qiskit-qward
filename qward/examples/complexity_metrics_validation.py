#!/usr/bin/env python3
"""
Complexity Metrics Validation Example

This example demonstrates the refactored complexity metrics API where the main
methods return validated schema objects directly, with optional dict methods
for backward compatibility.
"""

from qiskit import QuantumCircuit
from qward.metrics.complexity_metrics import ComplexityMetrics


def create_test_circuit() -> QuantumCircuit:
    """Create a test quantum circuit with various gate types."""
    qc = QuantumCircuit(4)

    # Add various gate types
    qc.h(0)  # Hadamard
    qc.t(1)  # T gate
    qc.cx(0, 1)  # CNOT
    qc.ry(0.5, 2)  # Parameterized rotation
    qc.cz(1, 2)  # Controlled-Z
    qc.swap(2, 3)  # SWAP
    qc.barrier()  # Barrier
    qc.measure_all()  # Measurements

    return qc


def validate_new_api():
    """Validate the new API where main methods return schemas directly."""
    print("=== New API Validation ===\n")

    # Create test circuit
    qc = create_test_circuit()
    print(f"Test circuit: {qc.size()} gates, {qc.depth()} depth, {qc.num_qubits} qubits\n")

    # Initialize metrics calculator
    metrics = ComplexityMetrics(qc)

    # Test each metric type with new API
    metric_tests = [
        ("Gate-based", "get_gate_based_metrics", "get_gate_based_metrics_dict"),
        ("Entanglement", "get_entanglement_metrics", "get_entanglement_metrics_dict"),
        ("Standardized", "get_standardized_metrics", "get_standardized_metrics_dict"),
        ("Advanced", "get_advanced_metrics", "get_advanced_metrics_dict"),
        ("Derived", "get_derived_metrics", "get_derived_metrics_dict"),
    ]

    all_passed = True

    for name, schema_method, dict_method in metric_tests:
        print(f"Testing {name} metrics...")

        # Get schema object
        schema_obj = getattr(metrics, schema_method)()
        schema_data = schema_obj.model_dump()

        # Get dict data
        dict_data = getattr(metrics, dict_method)()

        # Compare
        matches = schema_data == dict_data
        status = "✓ PASS" if matches else "✗ FAIL"
        print(f"  {status}: Schema vs Dict consistency check")
        print(f"  Schema type: {type(schema_obj)}")
        print(f"  Data: {schema_data}")

        if not matches:
            print(f"  Schema data: {schema_data}")
            print(f"  Dict data:   {dict_data}")
            all_passed = False

        print()

    # Test quantum volume (special case)
    print("Testing Quantum Volume metrics...")
    schema_qv = metrics.get_quantum_volume()
    dict_qv = metrics.estimate_quantum_volume_dict()

    # Compare by converting schema to dict format
    schema_qv_dict = {
        "standard_quantum_volume": schema_qv.standard_quantum_volume,
        "enhanced_quantum_volume": schema_qv.enhanced_quantum_volume,
        "effective_depth": schema_qv.effective_depth,
        "factors": schema_qv.factors.model_dump(),
        "circuit_metrics": schema_qv.circuit_metrics.model_dump(),
    }

    qv_matches = schema_qv_dict == dict_qv
    status = "✓ PASS" if qv_matches else "✗ FAIL"
    print(f"  {status}: Quantum Volume consistency check")
    print(f"  Schema type: {type(schema_qv)}")

    if not qv_matches:
        all_passed = False
        print(f"  Schema data: {schema_qv_dict}")
        print(f"  Dict data:   {dict_qv}")

    print()

    # Test complete metrics schema
    print("Testing Complete metrics schema...")
    complete_metrics = metrics.get_metrics()

    # Verify it has all expected sections
    expected_sections = [
        "gate_based_metrics",
        "entanglement_metrics",
        "standardized_metrics",
        "advanced_metrics",
        "derived_metrics",
        "quantum_volume",
    ]

    has_all_sections = all(hasattr(complete_metrics, section) for section in expected_sections)
    status = "✓ PASS" if has_all_sections else "✗ FAIL"
    print(f"  {status}: Complete schema structure check")
    print(f"  Schema type: {type(complete_metrics)}")

    if not has_all_sections:
        all_passed = False
        missing = [s for s in expected_sections if not hasattr(complete_metrics, s)]
        print(f"  Missing sections: {missing}")

    print()

    return all_passed


def validate_backward_compatibility():
    """Validate that deprecated methods still work with warnings."""
    print("=== Backward Compatibility Validation ===\n")

    qc = create_test_circuit()
    metrics = ComplexityMetrics(qc)

    import warnings

    # Test deprecated structured methods
    deprecated_methods = [
        ("get_structured_gate_based_metrics", "get_gate_based_metrics"),
        ("get_structured_entanglement_metrics", "get_entanglement_metrics"),
        ("get_structured_standardized_metrics", "get_standardized_metrics"),
        ("get_structured_advanced_metrics", "get_advanced_metrics"),
        ("get_structured_derived_metrics", "get_derived_metrics"),
        ("get_structured_quantum_volume", "get_quantum_volume"),
    ]

    all_passed = True

    for deprecated_method, new_method in deprecated_methods:
        print(f"Testing {deprecated_method}...")

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call deprecated method
            deprecated_result = getattr(metrics, deprecated_method)()

            # Call new method
            new_result = getattr(metrics, new_method)()

            # Check if warning was issued
            warning_issued = len(w) > 0 and issubclass(w[0].category, DeprecationWarning)

            # Check if results are identical
            results_match = deprecated_result.model_dump() == new_result.model_dump()

            status = "✓ PASS" if warning_issued and results_match else "✗ FAIL"
            print(f"  {status}: Warning issued: {warning_issued}, Results match: {results_match}")

            if not (warning_issued and results_match):
                all_passed = False
                if not warning_issued:
                    print(f"  ERROR: No deprecation warning issued")
                if not results_match:
                    print(f"  ERROR: Results don't match")

        print()

    # Test deprecated estimate_quantum_volume
    print("Testing estimate_quantum_volume...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        deprecated_qv = metrics.estimate_quantum_volume()
        new_qv = metrics.estimate_quantum_volume_dict()

        warning_issued = len(w) > 0 and issubclass(w[0].category, DeprecationWarning)
        results_match = deprecated_qv == new_qv

        status = "✓ PASS" if warning_issued and results_match else "✗ FAIL"
        print(f"  {status}: Warning issued: {warning_issued}, Results match: {results_match}")

        if not (warning_issued and results_match):
            all_passed = False

    print()
    return all_passed


def demonstrate_new_api_features():
    """Demonstrate the benefits of the new API."""
    print("=== New API Features Demonstration ===\n")

    qc = create_test_circuit()
    metrics = ComplexityMetrics(qc)

    # Get structured metrics directly
    advanced_metrics = metrics.get_advanced_metrics()

    print("New API benefits:")
    print(f"  1. Direct schema objects: {type(advanced_metrics)}")
    print(f"  2. Built-in validation: All values validated on creation")
    print(f"  3. Type safety: IDE autocomplete and type checking")
    print(f"  4. Easy serialization: {advanced_metrics.model_dump_json()}")
    print(f"  5. Field access: parallelism_factor = {advanced_metrics.parallelism_factor}")

    # Show dict access when needed
    print(f"\nDict access when needed:")
    dict_data = metrics.get_advanced_metrics_dict()
    print(f"  Dict format: {dict_data}")

    # Show validation in action
    print(f"\nValidation example:")
    try:
        from qward.metrics.schemas import AdvancedMetricsSchema

        # This should fail
        try:
            AdvancedMetricsSchema(
                parallelism_factor=2.0,
                parallelism_efficiency=1.5,  # Invalid: > 1.0
                circuit_efficiency=0.5,
                quantum_resource_utilization=0.5,
            )
            print("  ✗ Validation failed to catch invalid data")
        except Exception as e:
            print(f"  ✓ Validation caught invalid data: {type(e).__name__}")

    except ImportError:
        print("  Schema validation requires pydantic")


if __name__ == "__main__":
    # Test new API
    new_api_success = validate_new_api()

    # Test backward compatibility
    backward_compat_success = validate_backward_compatibility()

    if new_api_success and backward_compat_success:
        # Demonstrate new features
        demonstrate_new_api_features()

        print("\n=== CONCLUSION ===")
        print("✓ REFACTORING SUCCESSFUL!")
        print("✓ New API: Main methods return validated schema objects")
        print("✓ Dict methods: Available for users who need raw dictionaries")
        print("✓ Backward compatibility: Deprecated methods work with warnings")
        print("✓ All functionality preserved with improved type safety")
    else:
        print("\n=== CONCLUSION ===")
        print("✗ REFACTORING ISSUES DETECTED!")
        if not new_api_success:
            print("✗ New API has consistency issues")
        if not backward_compat_success:
            print("✗ Backward compatibility is broken")
