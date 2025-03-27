from qward.validators.teleportation_validator import TeleportationValidator
from qward.experiments.experiments import Experiments
import numpy as np


def main():
    # Create a teleportation validator
    validator = TeleportationValidator(
        payload_size=3,
        gates=["h", "x"],  # Apply Hadamard and X gates to the payload qubit
        use_barriers=True,
    )

    # Create experiments framework
    experiments = Experiments()

    # Run parameter correlation analysis
    param_ranges = {"payload_size": [2, 3, 4], "gates": [["h"], ["x"], ["h", "x"]]}

    print("Running parameter correlation analysis...")
    results = experiments.run_parameter_correlation(validator, param_ranges)
    print("\nResults:")
    print(results)

    # Plot success rates
    print("\nPlotting success rates...")
    experiments.plot_success_rates("payload_size")

    # Analyze error distribution
    print("\nAnalyzing error distribution...")
    error_analysis = experiments.analyze_error_distribution()
    print("\nError Analysis:")
    for key, value in error_analysis.items():
        print(f"{key}: {value}")

    # Export results
    print("\nExporting results...")
    experiments.export_results("teleportation_results.csv")

    # Run fixed parameter test
    print("\nRunning fixed parameter test...")
    fixed_params = {"payload_size": 3, "gates": ["h", "x"]}
    results = experiments.run_fixed_parameter_test(validator, fixed_params)
    print("\nFixed Parameter Test Results:")
    print(results)

    # Run depth analysis
    print("\nRunning depth analysis...")
    results = experiments.run_depth_analysis(
        validator, depth_range=range(1, 6), fixed_params={"payload_size": 3}
    )
    print("\nDepth Analysis Results:")
    print(results)

    # Use Qiskit features directly
    validator.draw()  # Draw the circuit
    validator.depth()  # Get circuit depth
    validator.count_ops()  # Count operations


if __name__ == "__main__":
    main()
