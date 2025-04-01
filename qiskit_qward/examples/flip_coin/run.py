from IPython.display import display
import pandas as pd

from qiskit_qward.examples.flip_coin.scanner import ScanningQuantumFlipCoin


def run_simulation_analysis(
    num_jobs=1000, shots_per_job=1024, show_histogram=True, export_results=True, save_metrics=True
):
    """
    Run simulation analysis on the quantum coin flip circuit.

    Args:
        num_jobs (int): Number of independent jobs to run
        shots_per_job (int): Number of shots per job
        show_histogram (bool): Whether to display histogram
        export_results (bool): Whether to export results to CSV
        save_metrics (bool): Whether to save complexity metrics to CSV

    Returns:
        dict: Comprehensive results including circuit metrics and analysis
    """
    # Create a flip coin scanner
    scanner = ScanningQuantumFlipCoin(use_barriers=True)

    # Display the circuit
    print("Quantum Coin Flip Circuit:")
    circuit_fig = scanner.draw()
    display(circuit_fig)

    # Run multiple jobs with many shots each to gather statistics
    print("\nRunning quantum simulation jobs...")
    print(f"({num_jobs} jobs with {shots_per_job} shots each)")
    results = scanner.run_simulation(
        show_histogram=show_histogram,  # Show histogram of first job
        num_jobs=num_jobs,  # Number of independent jobs
        shots_per_job=shots_per_job,  # Number of coin flips per job
    )

    # Get analysis results
    analysis = results["analysis"] = {}
    for i, analyzer in enumerate(scanner.analyzers):
        analysis[f"analyzer_{i}"] = analyzer.analyze()

    # Display analysis results for tails (1)
    print("\nAnalysis Results for Tails (1):")
    if "analyzer_0" in analysis:
        analyzer_results = analysis["analyzer_0"]
        for key, value in analyzer_results.items():
            if key == "average_counts":
                print(f"\nAverage counts per job:")
                print(f"Average number of heads (0): {value['heads']:.2f}")
                print(f"Average number of tails (1): {value['tails']:.2f}")
            else:
                print(f"{key}: {value:.2%}" if isinstance(value, float) else f"{key}: {value}")

    # Plot success rate distribution for tails
    print("\nPlotting success rate distribution for tails...")
    scanner.plot_analysis(ideal_rate=0.5)  # For a fair coin, we expect 50% tails

    # Export results to CSV
    if export_results and scanner.analyzers:
        print("\nExporting results...")
        scanner.analyzers[0].export_results("flip_coin_results.csv")

    # Display Complexity Metrics
    print("\n----------------------------------------")
    print("Circuit Complexity Analysis")
    print("----------------------------------------")

    # Display complexity metrics using pandas DataFrame for better visualization
    try:

        # Get complexity metrics from the simulation results
        complexity_metrics = results["complexity_metrics"]

        # Display Gate-Based Metrics
        print("\nGate-Based Metrics:")
        gate_metrics_df = pd.DataFrame(
            {
                "Metric": list(complexity_metrics["gate_based_metrics"].keys()),
                "Value": list(complexity_metrics["gate_based_metrics"].values()),
            }
        )
        display(gate_metrics_df)

        # Display Entanglement Metrics
        print("\nEntanglement Metrics:")
        entanglement_df = pd.DataFrame(
            {
                "Metric": list(complexity_metrics["entanglement_metrics"].keys()),
                "Value": list(complexity_metrics["entanglement_metrics"].values()),
            }
        )
        display(entanglement_df)

        # Display Standardized Metrics
        print("\nStandardized Metrics:")
        std_metrics_df = pd.DataFrame(
            {
                "Metric": list(complexity_metrics["standardized_metrics"].keys()),
                "Value": list(complexity_metrics["standardized_metrics"].values()),
            }
        )
        display(std_metrics_df)

        # Display Advanced Metrics
        print("\nAdvanced Circuit Metrics:")
        adv_metrics_df = pd.DataFrame(
            {
                "Metric": list(complexity_metrics["advanced_metrics"].keys()),
                "Value": list(complexity_metrics["advanced_metrics"].values()),
            }
        )
        display(adv_metrics_df)

        # Display Derived Metrics
        print("\nDerived Complexity Metrics:")
        derived_df = pd.DataFrame(
            {
                "Metric": list(complexity_metrics["derived_metrics"].keys()),
                "Value": list(complexity_metrics["derived_metrics"].values()),
            }
        )
        display(derived_df)

        # Display Quantum Volume Estimate
        qv_estimate = results["quantum_volume"]
        print("\nQuantum Volume Estimates:")
        qv_df = pd.DataFrame(
            {
                "Metric": ["Standard Quantum Volume", "Enhanced Quantum Volume"],
                "Value": [
                    qv_estimate["standard_quantum_volume"],
                    qv_estimate["enhanced_quantum_volume"],
                ],
            }
        )
        display(qv_df)

        # Display factors affecting quantum volume
        print("\nFactors affecting Quantum Volume:")
        factors_df = pd.DataFrame(
            {
                "Factor": list(qv_estimate["factors"].keys()),
                "Value": list(qv_estimate["factors"].values()),
            }
        )
        display(factors_df)

    except ImportError:
        print("Pandas not found - displaying raw data:")
        print("\nComplexity Metrics:")
        for category, metrics in complexity_metrics.items():
            if category != "basic_properties":  # Skip the basic properties for brevity
                print(f"\n{category.replace('_', ' ').title()}:")
                for metric, value in metrics.items():
                    print(f"  {metric.replace('_', ' ').title()}: {value}")

        print("\nQuantum Volume Estimate:")
        print(f"  Standard Quantum Volume: {qv_estimate['standard_quantum_volume']}")
        print(f"  Enhanced Quantum Volume: {qv_estimate['enhanced_quantum_volume']}")
        print(f"  Effective Depth: {qv_estimate['effective_depth']}")

        print("\nFactors affecting Quantum Volume:")
        for factor, value in qv_estimate["factors"].items():
            print(f"  {factor.replace('_', ' ').title()}: {value}")

    # Optionally, save the metrics to a CSV file
    if save_metrics:
        try:

            # Create a flattened dictionary for CSV export
            flat_metrics = {}

            # Flatten the complexity metrics
            for category, metrics in complexity_metrics.items():
                if category != "basic_properties":  # Skip the large operation counts
                    for metric, value in metrics.items():
                        flat_metrics[f"{category}_{metric}"] = value

            # Add quantum volume metrics
            flat_metrics["standard_quantum_volume"] = qv_estimate["standard_quantum_volume"]
            flat_metrics["enhanced_quantum_volume"] = qv_estimate["enhanced_quantum_volume"]

            # Add factors
            for factor, value in qv_estimate["factors"].items():
                flat_metrics[f"factor_{factor}"] = value

            # Convert to DataFrame and save
            metrics_df = pd.DataFrame([flat_metrics])
            metrics_df.to_csv("flip_coin_complexity_metrics.csv", index=False)
            print("\nComplexity metrics saved to 'flip_coin_complexity_metrics.csv'")

        except (ImportError, Exception) as e:
            print(f"\nUnable to save metrics to CSV: {str(e)}")

    return results


def run_ibm_analysis(channel=None, token=None, shots=1024, show_histogram=True):
    """
    Run the quantum coin flip circuit on IBM Quantum hardware.

    Args:
        channel (str): IBM Quantum channel (if None, uses environment variables)
        token (str): IBM Quantum token (if None, uses environment variables)
        shots (int): Number of shots to run
        show_histogram (bool): Whether to display histogram

    Returns:
        dict: Results from IBM Quantum hardware run
    """
    print("\nRunning on IBM Quantum Hardware...")
    print(f"(Single job with {shots} shots)")

    # Create a scanner for IBM run
    ibm_validator = ScanningQuantumFlipCoin(use_barriers=True)

    # Display the circuit
    if show_histogram:
        print("Quantum Coin Flip Circuit for IBM Hardware:")
        circuit_fig = ibm_validator.draw()
        display(circuit_fig)

    # Run on IBM with a single job
    ibm_results = ibm_validator.run_on_ibm(channel=channel, token=token)

    if ibm_results["status"] == "completed":
        print(f"\nBackend used: {ibm_results['backend']}")
        print(f"Job ID: {ibm_results['job_id']}")

        print("\nResults from IBM Quantum:")
        counts = ibm_results["counts"]
        total_shots = sum(counts.values())
        heads = counts.get("0", 0)
        tails = counts.get("1", 0)

        print(f"Total shots: {total_shots}")
        print(f"Heads (0): {heads} ({heads/total_shots:.2%})")
        print(f"Tails (1): {tails} ({tails/total_shots:.2%})")

        # Display Complexity Metrics from IBM run
        try:

            print("\n----------------------------------------")
            print("IBM Hardware Circuit Complexity Analysis")
            print("----------------------------------------")

            # Get complexity metrics from the IBM results
            complexity_metrics = ibm_results["complexity_metrics"]
            print(complexity_metrics)

            # Display Quantum Volume Estimate
            qv_estimate = ibm_results["quantum_volume"]
            print("\nQuantum Volume Estimates:")
            qv_df = pd.DataFrame(
                {
                    "Metric": ["Standard Quantum Volume", "Enhanced Quantum Volume"],
                    "Value": [
                        qv_estimate["standard_quantum_volume"],
                        qv_estimate["enhanced_quantum_volume"],
                    ],
                }
            )
            display(qv_df)

        except (ImportError, KeyError, Exception) as e:
            print(f"\nUnable to display IBM complexity metrics: {str(e)}")

    else:
        print(f"\nIBM Quantum execution failed:")
        print(f"Status: {ibm_results['status']}")
        if "error" in ibm_results:
            print(f"Error: {ibm_results['error']}")

    return ibm_results


# Code to run if script is executed directly
if __name__ == "__main__":
    # Run simulation analysis
    sim_results = run_simulation_analysis()

    # Uncomment to run IBM analysis - be careful as this uses real quantum hardware
    # ibm_results = run_ibm_analysis()
