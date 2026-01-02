"""
Example usage for QuantumSpecificMetricsVisualizer.

This script computes QuantumSpecificMetrics using the Scanner (if available)
or expects a metrics_data dict with a DataFrame under "QuantumSpecificMetrics",
then renders the compact dashboard and individual plots.
"""

from qward.visualization import Visualizer, PlotConfig
from qward.scanner import Scanner


def main():
    # Build a scanner as in other examples (replace with your circuit loading as needed)
    # For demonstration purposes, this just assumes Scanner is configured to compute QuantumSpecificMetrics.
    scanner = Scanner()

    viz = Visualizer(
        scanner=scanner, config=PlotConfig(style="quantum"), output_dir="qward/examples/img"
    )

    # Print available metrics and plots
    viz.print_available_metrics()

    # Create dashboard for quantum-specific metrics only (if available)
    dashboards = viz.create_dashboard(save=True, show=False)
    if "QuantumSpecificMetrics" in dashboards:
        print("Saved QuantumSpecificMetrics dashboard to output directory")

    # Generate individual plots if you prefer
    try:
        viz.generate_plot(
            metric_name="QuantumSpecificMetrics",
            plot_name="all_metrics_bar",
            save=True,
            show=False,
        )
        viz.generate_plot(
            metric_name="QuantumSpecificMetrics",
            plot_name="quantum_radar",
            save=True,
            show=False,
        )
    except Exception as e:
        print(f"Warning: could not generate individual quantum specific plots: {e}")


if __name__ == "__main__":
    main()
