from qiskit_qward.examples.two_doors_enigma.scanner import ScanningQuantumEnigma


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

    return scanner


if __name__ == "__main__":
    scanner = run()
    print("\nCircuit:")
    scanner.draw()  # This will use the matplotlib output
