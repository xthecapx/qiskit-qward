"""
Demo showing how to use QWARD's CircuitPerformance metrics.

This example demonstrates:
1. Traditional approach using CircuitPerformance directly
2. Schema-based approach with validation
3. Custom success criteria
4. Validation features and error handling
5. JSON schema generation
"""

from typing import TYPE_CHECKING, Dict, Any, List
import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qward.metrics.circuit_performance import CircuitPerformance

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

if TYPE_CHECKING:
    from qward.metrics.schemas import SuccessRateJobSchema, SuccessRateAggregateSchema


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
    success_rate = CircuitPerformance(circuit, job=job)
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

    success_rate_multi = CircuitPerformance(circuit, jobs=jobs)
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
        circuit_performance = CircuitPerformance(circuit, job=job)
        structured_metrics = circuit_performance.get_structured_metrics()

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

        circuit_performance_multi = CircuitPerformance(circuit, jobs=jobs)
        aggregate_metrics = circuit_performance_multi.get_structured_metrics()

        print(f"\n✓ Multiple jobs aggregate metrics:")
        print(f"   Mean Success Rate: {aggregate_metrics.success_metrics.mean_success_rate}")
        print(f"   Mean Fidelity: {aggregate_metrics.fidelity_metrics.mean_fidelity}")
        print(f"   Individual Jobs: {len(aggregate_metrics.success_metrics.individual_jobs)}")

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
    default_success = CircuitPerformance(circuit, job=job)
    default_metrics = default_success.get_metrics()

    print("Default Success Criteria (|00⟩ only):")
    print(f"Success Rate: {default_metrics['success_metrics']['success_rate']:.3f}")
    print()

    # Custom success criteria (Bell state outcomes: |00⟩ or |11⟩)
    def bell_success_criteria(result: str) -> bool:
        # Remove spaces to get clean bit string
        clean_result = result.replace(" ", "")
        # For Bell states, we expect either all 0s or all 1s
        return clean_result in ["0000", "1111"]  # 00 00 or 11 11 for 2-qubit + 2-classical

    bell_success = CircuitPerformance(circuit, job=job, success_criteria=bell_success_criteria)
    bell_metrics = bell_success.get_metrics()

    print("Custom Success Criteria (|00⟩ or |11⟩):")
    print(f"Success Rate: {bell_metrics['success_metrics']['success_rate']:.3f}")
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
    from qward.metrics.schemas import CircuitPerformanceSchema, SuccessMetricsSchema, FidelityMetricsSchema, StatisticalMetricsSchema

    # Create a circuit and run it
    circuit = create_bell_circuit()
    job = run_circuit_simulation(circuit)

    print("1. Valid schema creation:")
    job_schema = CircuitPerformanceSchema(
        success_metrics=SuccessMetricsSchema(
            job_id="test_job_123",
            success_rate=0.85,
            error_rate=0.15,
            total_shots=1024,
            successful_shots=870,
        ),
        fidelity_metrics=FidelityMetricsSchema(
            job_id="test_job_123",
            fidelity=0.92,
            method="theoretical_comparison",
            confidence="high",
            has_expected_distribution=True,
        ),
        statistical_metrics=StatisticalMetricsSchema(
            job_id="test_job_123",
            entropy=0.85,
            uniformity=0.75,
            concentration=0.25,
            dominant_outcome_probability=0.45,
            num_unique_outcomes=4,
        ),
    )
    print(f"✓ Valid schema created: {job_schema.success_metrics.job_id}")

    print("\n2. Multiple job aggregate schema:")
    aggregate_schema = CircuitPerformanceSchema(
        success_metrics=SuccessMetricsSchema(
            mean_success_rate=0.82,
            std_success_rate=0.05,
            min_success_rate=0.75,
            max_success_rate=0.90,
            total_trials=3072,
            error_rate=0.18,
            individual_jobs=[
                SuccessMetricsSchema(
                    job_id="job_1",
                    success_rate=0.85,
                    error_rate=0.15,
                    total_shots=1024,
                    successful_shots=870,
                ),
                SuccessMetricsSchema(
                    job_id="job_2", 
                    success_rate=0.80,
                    error_rate=0.20,
                    total_shots=1024,
                    successful_shots=819,
                ),
                SuccessMetricsSchema(
                    job_id="job_3",
                    success_rate=0.82,
                    error_rate=0.18,
                    total_shots=1024,
                    successful_shots=840,
                ),
            ],
        ),
        fidelity_metrics=FidelityMetricsSchema(
            mean_fidelity=0.89,
            std_fidelity=0.03,
            min_fidelity=0.85,
            max_fidelity=0.92,
            method="theoretical_comparison",
            confidence="high",
            individual_jobs=[
                FidelityMetricsSchema(
                    job_id="job_1",
                    fidelity=0.92,
                    method="theoretical_comparison",
                    confidence="high",
                    has_expected_distribution=True,
                ),
                FidelityMetricsSchema(
                    job_id="job_2",
                    fidelity=0.85,
                    method="theoretical_comparison", 
                    confidence="high",
                    has_expected_distribution=True,
                ),
                FidelityMetricsSchema(
                    job_id="job_3",
                    fidelity=0.90,
                    method="theoretical_comparison",
                    confidence="high", 
                    has_expected_distribution=True,
                ),
            ],
        ),
        statistical_metrics=StatisticalMetricsSchema(
            mean_entropy=0.83,
            mean_uniformity=0.72,
            mean_concentration=0.28,
            mean_dominant_probability=0.47,
            std_entropy=0.02,
            std_uniformity=0.03,
            individual_jobs=[
                StatisticalMetricsSchema(
                    job_id="job_1",
                    entropy=0.85,
                    uniformity=0.75,
                    concentration=0.25,
                    dominant_outcome_probability=0.45,
                    num_unique_outcomes=4,
                ),
                StatisticalMetricsSchema(
                    job_id="job_2",
                    entropy=0.80,
                    uniformity=0.68,
                    concentration=0.32,
                    dominant_outcome_probability=0.50,
                    num_unique_outcomes=4,
                ),
                StatisticalMetricsSchema(
                    job_id="job_3",
                    entropy=0.84,
                    uniformity=0.73,
                    concentration=0.27,
                    dominant_outcome_probability=0.46,
                    num_unique_outcomes=4,
                ),
            ],
        ),
    )
    print(f"✓ Aggregate schema created with {len(aggregate_schema.success_metrics.individual_jobs)} jobs")


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
    from qward.metrics.schemas import CircuitPerformanceSchema, SuccessMetricsSchema, FidelityMetricsSchema, StatisticalMetricsSchema

    job_schema = CircuitPerformanceSchema.model_json_schema()
    print("CircuitPerformance JSON Schema keys:")
    print(list(job_schema.keys()))

    success_schema = SuccessMetricsSchema.model_json_schema()
    print("\nSuccessMetrics JSON Schema properties:")
    print(list(success_schema.get("properties", {}).keys()))


def main() -> None:
    """Run all success rate demos."""
    print("🚀 QWARD Success Rate Metrics Demo")
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
