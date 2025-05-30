"""
Success Rate Metrics Demo for QWARD.

This demo showcases the success rate metrics functionality, including:
- Traditional dictionary-based metrics
- Schema-based structured metrics with validation
- Single job and multiple job analysis
- Custom success criteria
"""

from typing import TYPE_CHECKING, Dict, Any, List
import json

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qward.metrics.success_rate import SuccessRate

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
    success_rate = SuccessRate(circuit, job=job)
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

    success_rate_multi = SuccessRate(circuit, jobs=jobs)
    multi_metrics = success_rate_multi.get_metrics()

    print("Multiple Jobs Metrics (Dictionary):")
    print(json.dumps(multi_metrics, indent=2, default=str))
    print()


def demo_schema_approach() -> None:
    """Demonstrate schema-based success rate metrics with validation."""
    print("=" * 60)
    print("SCHEMA-BASED APPROACH WITH VALIDATION")
    print("=" * 60)

    try:
        from qward.metrics.schemas import SuccessRateJobSchema, SuccessRateAggregateSchema

        # Create and run a Bell state circuit
        circuit = create_bell_circuit()
        job = run_circuit_simulation(circuit, shots=1024)

        # Calculate structured metrics for single job
        success_rate = SuccessRate(circuit, job=job)
        structured_metrics = success_rate.get_structured_single_job_metrics()

        print("Single Job Structured Metrics:")
        print(f"Type: {type(structured_metrics).__name__}")
        print(f"Job ID: {structured_metrics.job_id}")
        print(f"Success Rate: {structured_metrics.success_rate:.3f}")
        print(f"Error Rate: {structured_metrics.error_rate:.3f}")
        print(f"Fidelity: {structured_metrics.fidelity:.3f}")
        print(f"Total Shots: {structured_metrics.total_shots}")
        print(f"Successful Shots: {structured_metrics.successful_shots}")
        print()

        # Multiple jobs with schema validation
        jobs = [
            run_circuit_simulation(circuit, shots=256),
            run_circuit_simulation(circuit, shots=256),
            run_circuit_simulation(circuit, shots=256),
            run_circuit_simulation(circuit, shots=256),
        ]

        success_rate_multi = SuccessRate(circuit, jobs=jobs)
        aggregate_metrics = success_rate_multi.get_structured_multiple_jobs_metrics()

        print("Multiple Jobs Aggregate Structured Metrics:")
        print(f"Type: {type(aggregate_metrics).__name__}")
        print(f"Mean Success Rate: {aggregate_metrics.mean_success_rate:.3f}")
        print(f"Std Success Rate: {aggregate_metrics.std_success_rate:.3f}")
        print(f"Min Success Rate: {aggregate_metrics.min_success_rate:.3f}")
        print(f"Max Success Rate: {aggregate_metrics.max_success_rate:.3f}")
        print(f"Total Trials: {aggregate_metrics.total_trials}")
        print(f"Average Fidelity: {aggregate_metrics.fidelity:.3f}")
        print(f"Error Rate: {aggregate_metrics.error_rate:.3f}")
        print()

    except ImportError:
        print("❌ Pydantic schemas not available. Install pydantic to use structured metrics.")
        print()


def demo_custom_success_criteria() -> None:
    """Demonstrate custom success criteria functionality."""
    print("=" * 60)
    print("CUSTOM SUCCESS CRITERIA")
    print("=" * 60)

    # Create a Bell state circuit
    circuit = create_bell_circuit()
    job = run_circuit_simulation(circuit, shots=1024)

    # Default success criteria (all zeros)
    default_success = SuccessRate(circuit, job=job)
    default_metrics = default_success.get_metrics()

    print("Default Success Criteria (|00⟩ only):")
    print(f"Success Rate: {default_metrics['success_rate']:.3f}")
    print()

    # Custom success criteria (Bell state outcomes: |00⟩ or |11⟩)
    def bell_success_criteria(result: str) -> bool:
        # Remove spaces to get clean bit string
        clean_result = result.replace(" ", "")
        # For Bell states, we expect either all 0s or all 1s
        return clean_result in ["0000", "1111"]  # 00 00 or 11 11 for 2-qubit + 2-classical

    bell_success = SuccessRate(circuit, job=job, success_criteria=bell_success_criteria)
    bell_metrics = bell_success.get_metrics()

    print("Custom Success Criteria (|00⟩ or |11⟩):")
    print(f"Success Rate: {bell_metrics['success_rate']:.3f}")
    print()


def demo_validation_features() -> None:
    """Demonstrate schema validation features."""
    print("=" * 60)
    print("SCHEMA VALIDATION FEATURES")
    print("=" * 60)

    try:
        from qward.metrics.schemas import SuccessRateJobSchema, SuccessRateAggregateSchema

        # Valid data
        print("✅ Valid Success Rate Job Schema:")
        job_schema = SuccessRateJobSchema(
            job_id="test_job_123",
            success_rate=0.75,
            error_rate=0.25,
            fidelity=0.80,
            total_shots=1024,
            successful_shots=768,
        )
        print(f"Success Rate: {job_schema.success_rate}")
        print(f"Error Rate: {job_schema.error_rate}")
        print("Validation: ✅ Passed")
        print()

        # Test validation - this should work
        print("✅ Valid Aggregate Schema:")
        aggregate_schema = SuccessRateAggregateSchema(
            mean_success_rate=0.72,
            std_success_rate=0.05,
            min_success_rate=0.65,
            max_success_rate=0.80,
            total_trials=3072,
            fidelity=0.75,
            error_rate=0.28,
        )
        print(f"Mean Success Rate: {aggregate_schema.mean_success_rate}")
        print(f"Standard Deviation: {aggregate_schema.std_success_rate}")
        print("Validation: ✅ Passed")
        print()

        # Test validation errors
        print("❌ Testing Validation Errors:")

        try:
            # Invalid: success_rate > 1.0
            SuccessRateJobSchema(
                job_id="test_job_123",
                success_rate=1.5,  # Invalid
                error_rate=0.25,
                fidelity=0.80,
                total_shots=1024,
                successful_shots=768,
            )
        except Exception as e:
            print(f"Success rate > 1.0: {type(e).__name__}")

        try:
            # Invalid: successful_shots > total_shots
            SuccessRateJobSchema(
                job_id="test_job_123",
                success_rate=0.75,
                error_rate=0.25,
                fidelity=0.80,
                total_shots=1024,
                successful_shots=2000,  # Invalid
            )
        except Exception as e:
            print(f"Successful shots > total shots: {type(e).__name__}")

        try:
            # Invalid: min > max in aggregate
            SuccessRateAggregateSchema(
                mean_success_rate=0.72,
                std_success_rate=0.05,
                min_success_rate=0.90,  # Invalid - greater than mean
                max_success_rate=0.80,
                total_trials=3072,
                fidelity=0.75,
                error_rate=0.28,
            )
        except Exception as e:
            print(f"Min > max success rate: {type(e).__name__}")

        print()

    except ImportError:
        print("❌ Pydantic schemas not available. Install pydantic to test validation.")
        print()


def demo_json_schema_generation() -> None:
    """Demonstrate JSON schema generation for API documentation."""
    print("=" * 60)
    print("JSON SCHEMA GENERATION")
    print("=" * 60)

    try:
        from qward.metrics.schemas import SuccessRateJobSchema, SuccessRateAggregateSchema

        print("Success Rate Job Schema (JSON Schema):")
        job_schema = SuccessRateJobSchema.model_json_schema()
        print(json.dumps(job_schema, indent=2))
        print()

        print("Success Rate Aggregate Schema (JSON Schema):")
        aggregate_schema = SuccessRateAggregateSchema.model_json_schema()
        print(json.dumps(aggregate_schema, indent=2))
        print()

    except ImportError:
        print("❌ Pydantic schemas not available. Install pydantic for JSON schema generation.")
        print()


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
