"""
Demo showing how to use QWARD's CircuitPerformanceMetrics metrics.

This example demonstrates:
1. Traditional approach using CircuitPerformanceMetrics directly
2. Schema-based approach with validation
3. Custom success criteria and fidelity calculation
4. Validation features and error handling
5. JSON schema generation
"""

from typing import Dict, Any, List
import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qward.metrics.circuit_performance import CircuitPerformanceMetrics

# Import schemas for structured data validation
try:
    from qward.metrics.schemas import (
        CircuitPerformanceSchema,
        SuccessMetricsSchema,
        FidelityMetricsSchema,
        StatisticalMetricsSchema,
    )

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


def create_bell_circuit() -> QuantumCircuit:
    """Create a Bell state circuit for testing."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def create_simple_circuit() -> QuantumCircuit:
    """Create a simple circuit that should succeed most of the time."""
    circuit = QuantumCircuit(2, 2)
    circuit.measure_all()  # Should measure |00⟩ most of the time
    return circuit


def run_circuit_simulation(circuit: QuantumCircuit, shots: int = 1024) -> Any:
    """Run a circuit simulation and return the job."""
    simulator = AerSimulator()
    transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit, shots=shots)
    return job


def demo_traditional_approach() -> None:
    """Demonstrate traditional dictionary-based success rate metrics."""
    print("=" * 60)
    print("TRADITIONAL DICTIONARY-BASED APPROACH")
    print("=" * 60)

    # Create and run a simple circuit
    circuit = create_simple_circuit()
    job = run_circuit_simulation(circuit, shots=1024)

    # Calculate success rate metrics
    success_rate = CircuitPerformanceMetrics(circuit, job=job)
    metrics = success_rate.get_metrics()

    print("Single Job Metrics (Dictionary):")
    print(json.dumps(metrics, indent=2, default=str))
    print()

    # Multiple jobs example
    jobs = [
        run_circuit_simulation(circuit, shots=512),
        run_circuit_simulation(circuit, shots=512),
        run_circuit_simulation(circuit, shots=512),
    ]

    success_rate_multi = CircuitPerformanceMetrics(circuit, jobs=jobs)
    multi_metrics = success_rate_multi.get_metrics()

    print("Multiple Jobs Metrics (Dictionary):")
    print(json.dumps(multi_metrics, indent=2, default=str))
    print()


def demo_schema_approach() -> None:
    """
    Demonstrate schema-based metrics with validation.
    """
    print("\n" + "=" * 50)
    print("SCHEMA-BASED APPROACH DEMO")
    print("=" * 50)

    if not SCHEMAS_AVAILABLE:
        print("Pydantic schemas not available. Install pydantic to run this demo.")
        return

    # Create a circuit and run it
    circuit = create_bell_circuit()
    job = run_circuit_simulation(circuit)

    try:
        # Calculate structured metrics for single job
        circuit_performance = CircuitPerformanceMetrics(circuit, job=job)
        structured_metrics = circuit_performance.get_metrics()

        print("✓ Schema-based metrics (structured object):")
        print(f"   Type: {type(structured_metrics)}")
        print(f"   Success Rate: {structured_metrics.success_metrics.success_rate}")
        print(f"   Fidelity: {structured_metrics.fidelity_metrics.fidelity}")
        print(f"   Entropy: {structured_metrics.statistical_metrics.entropy}")

        print("\n✓ Benefits of schema approach:")
        print("   - Full type hints and IDE autocomplete")
        print("   - Automatic validation of data types and constraints")
        print("   - Clear documentation of all available fields")
        print("   - Compile-time error detection")

        # Multiple jobs example
        jobs = [
            run_circuit_simulation(circuit),
            run_circuit_simulation(circuit),
            run_circuit_simulation(circuit),
        ]

        circuit_performance_multi = CircuitPerformanceMetrics(circuit, jobs=jobs)
        aggregate_metrics = circuit_performance_multi.get_metrics()

        print("\n✓ Multiple jobs aggregate metrics:")
        print(f"   Mean Success Rate: {aggregate_metrics.success_metrics.mean_success_rate}")
        print(f"   Mean Fidelity: {aggregate_metrics.fidelity_metrics.mean_fidelity}")
        print(
            f"   Standard Deviation (Success): {aggregate_metrics.success_metrics.std_success_rate}"
        )

    except Exception as e:
        print(f"❌ Error in schema approach: {e}")
        return


def demo_custom_success_criteria() -> None:
    """Demonstrate custom success criteria functionality."""
    print("=" * 60)
    print("CUSTOM SUCCESS CRITERIA")
    print("=" * 60)

    # Create a Bell state circuit
    circuit = create_bell_circuit()
    job = run_circuit_simulation(circuit, shots=1024)

    # Default success criteria (all zeros)
    default_success = CircuitPerformanceMetrics(circuit, job=job)
    default_metrics = default_success.get_metrics()

    print("Default Success Criteria (|00⟩ only):")
    print(f"Success Rate: {default_metrics.success_metrics.success_rate:.3f}")
    print()

    # Custom success criteria (Bell state outcomes: |00⟩ or |11⟩)
    def bell_success_criteria(result: str) -> bool:
        # Remove spaces to get clean bit string
        clean_result = result.replace(" ", "")
        # For Bell states, we expect either all 0s or all 1s
        return clean_result in ["0000", "1111"]  # 00 00 or 11 11 for 2-qubit + 2-classical

    bell_success = CircuitPerformanceMetrics(
        circuit, job=job, success_criteria=bell_success_criteria
    )
    bell_metrics = bell_success.get_metrics()

    print("Custom Success Criteria (|00⟩ or |11⟩):")
    print(f"Success Rate: {bell_metrics.success_metrics.success_rate:.3f}")
    print()


def demo_validation_features() -> None:
    """
    Demonstrate validation features and error handling.
    """
    print("\n" + "=" * 50)
    print("VALIDATION FEATURES DEMO")
    print("=" * 50)

    if not SCHEMAS_AVAILABLE:
        print("Pydantic schemas not available. Install pydantic to run this demo.")
        return

    # Import schemas for validation
    from qward.metrics.schemas import (
        CircuitPerformanceSchema,
        SuccessMetricsSchema,
        FidelityMetricsSchema,
        StatisticalMetricsSchema,
    )

    print("1. Valid single job schema creation:")
    single_job_schema = CircuitPerformanceSchema(
        success_metrics=SuccessMetricsSchema(
            success_rate=0.85,
            error_rate=0.15,
            total_shots=1024,
            successful_shots=870,
        ),
        fidelity_metrics=FidelityMetricsSchema(
            fidelity=0.92,
            method="theoretical_comparison",
            confidence="high",
            has_expected_distribution=True,
        ),
        statistical_metrics=StatisticalMetricsSchema(
            entropy=0.85,
            uniformity=0.75,
            concentration=0.25,
            dominant_outcome_probability=0.45,
            num_unique_outcomes=4,
        ),
    )
    print(
        f"✓ Valid single job schema created: Success Rate = {single_job_schema.success_metrics.success_rate}"
    )

    print("\n2. Valid multiple jobs aggregate schema:")
    aggregate_schema = CircuitPerformanceSchema(
        success_metrics=SuccessMetricsSchema(
            mean_success_rate=0.82,
            std_success_rate=0.05,
            min_success_rate=0.75,
            max_success_rate=0.90,
            total_trials=3072,
            error_rate=0.18,
        ),
        fidelity_metrics=FidelityMetricsSchema(
            mean_fidelity=0.89,
            std_fidelity=0.03,
            min_fidelity=0.85,
            max_fidelity=0.92,
            method="theoretical_comparison",
            confidence="high",
        ),
        statistical_metrics=StatisticalMetricsSchema(
            mean_entropy=0.83,
            mean_uniformity=0.72,
            mean_concentration=0.28,
            mean_dominant_probability=0.47,
            std_entropy=0.02,
            std_uniformity=0.03,
        ),
    )
    print(
        f"✓ Valid aggregate schema created: Mean Success Rate = {aggregate_schema.success_metrics.mean_success_rate}"
    )

    print("\n3. Schema validation error handling:")
    try:
        # This should fail because success_rate > 1.0
        SuccessMetricsSchema(
            success_rate=1.5,  # Invalid: > 1.0
            error_rate=0.15,
            total_shots=1024,
            successful_shots=870,
        )
        print("❌ Validation should have failed!")
    except Exception as e:
        print(f"✓ Validation correctly caught error: {type(e).__name__}")

    print("\n4. Real circuit metrics validation:")
    # Create a circuit and run it
    circuit = create_bell_circuit()
    job = run_circuit_simulation(circuit)

    # Get real metrics and validate them
    circuit_performance = CircuitPerformanceMetrics(circuit, job=job)
    real_metrics = circuit_performance.get_metrics()  # Now returns schema directly

    print("✓ Real metrics validated successfully:")
    print(f"   Success Rate: {real_metrics.success_metrics.success_rate:.3f}")
    print(f"   Fidelity: {real_metrics.fidelity_metrics.fidelity:.3f}")
    print(f"   Method: {real_metrics.fidelity_metrics.method}")
    print(f"   Entropy: {real_metrics.statistical_metrics.entropy:.3f}")


def demo_json_schema_generation() -> None:
    """
    Demonstrate JSON schema generation capabilities.
    """
    print("\n" + "=" * 50)
    print("JSON SCHEMA GENERATION DEMO")
    print("=" * 50)

    if not SCHEMAS_AVAILABLE:
        print("Pydantic schemas not available. Install pydantic to run this demo.")
        return

    # Import schemas for JSON generation
    from qward.metrics.schemas import (
        CircuitPerformanceSchema,
        SuccessMetricsSchema,
        FidelityMetricsSchema,
        StatisticalMetricsSchema,
    )

    job_schema = CircuitPerformanceSchema.model_json_schema()
    print("CircuitPerformance JSON Schema keys:")
    print(list(job_schema.keys()))

    success_schema = SuccessMetricsSchema.model_json_schema()
    print("\nSuccessMetrics JSON Schema properties:")
    print(list(success_schema.get("properties", {}).keys()))


def main() -> None:
    """Run all circuit performance demos."""
    print("🚀 QWARD CircuitPerformance Metrics Demo")
    print("=" * 60)
    print()

    demo_traditional_approach()
    demo_schema_approach()
    demo_custom_success_criteria()
    demo_validation_features()
    demo_json_schema_generation()

    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
