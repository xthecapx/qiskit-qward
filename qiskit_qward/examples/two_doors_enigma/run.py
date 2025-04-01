from qiskit_qward.examples.two_doors_enigma.scanner import ScanningQuantumEnigma

try:
    from IPython.display import display
except ImportError:
    # Define a no-op display function for environments without IPython
    def display(*args, **kwargs):
        pass


def run():
    # Create and run the quantum enigma scanner
    scanner = ScanningQuantumEnigma()

    # Run simulation with histogram
    results = scanner.run_simulation(show_histogram=True, num_jobs=1000, shots_per_job=1000)

    # Print the results
    print("\nQuantum Enigma Results:")
    print("------------------------")
    for state, count in results["results_metrics"]["counts"].items():
        # state format: [lie_qubit, q1_guardian, q0_guardian]
        lie_qubit = state[0]  # Which guardian is lying (0=q0, 1=q1)
        q1_guardian = state[1]  # q1 guardian's answer (0=right door, 1=left door)
        q0_guardian = state[2]  # q0 guardian's answer (0=right door, 1=left door)

        # Determine where the treasure is
        # If both guardians point to the same door, that's the door NOT to open
        # So the treasure is behind the opposite door
        if q0_guardian == q1_guardian:
            # If they both point to right door (0), treasure is behind left door (1)
            # If they both point to left door (1), treasure is behind right door (0)
            treasure_door = "1" if q0_guardian == "0" else "0"
        else:
            treasure_door = "unknown"  # This shouldn't happen in our circuit

        print(f"State {state}: {count} times")
        print(f"  Guardian lying: {'q1' if lie_qubit == '1' else 'q0'}")
        print(f"  q1 guardian points to: {'Right' if q1_guardian == '0' else 'Left'} door")
        print(f"  q0 guardian points to: {'Right' if q0_guardian == '0' else 'Left'} door")
        print(f"  Treasure is behind: {'Right' if treasure_door == '0' else 'Left'} door")
        print("  ---")

    # Print circuit metrics
    print("\nCircuit Metrics:")
    print("---------------")
    for metric, value in results["circuit_metrics"].items():
        print(f"{metric}: {value}")

    # Print success analysis
    print("\nSuccess Analysis:")
    print("----------------")
    analysis_results = scanner.run_analysis()
    analysis = analysis_results["analyzer_0"]  # Get the first analyzer's results
    print(f"Mean success rate: {analysis['mean_success_rate']:.2%}")
    print(f"Standard deviation: {analysis['std_success_rate']:.2%}")
    print(f"Min success rate: {analysis['min_success_rate']:.2%}")
    print(f"Max success rate: {analysis['max_success_rate']:.2%}")
    print(f"Total trials: {analysis['total_trials']}")

    # Plot analysis results
    print("\nGenerating analysis plots...")
    scanner.plot_analysis(ideal_rate=1.0)

    # Display complexity metrics
    print("\n----------------------------------------")
    print("Circuit Complexity Analysis")
    print("----------------------------------------")

    try:
        import pandas as pd

        # Get complexity metrics from the results
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

        # Save metrics to CSV
        flat_metrics = {}
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
        metrics_df.to_csv("enigma_complexity_metrics.csv", index=False)
        print("\nComplexity metrics saved to 'enigma_complexity_metrics.csv'")

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
    except Exception as e:
        print(f"Error displaying complexity metrics: {str(e)}")

    return scanner


if __name__ == "__main__":
    scanner = run()
    print("\nCircuit:")
    scanner.draw()  # This will use the matplotlib output
