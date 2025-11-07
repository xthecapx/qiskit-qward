"""Quick example for BehavioralMetricsVisualizer usage."""

from qiskit import QuantumCircuit
from qward import Scanner
from qward.metrics import BehavioralMetrics
from qward.visualization import Visualizer
from qward.visualization.constants import Metrics, Plots


def main():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.measure_all()

    scanner = Scanner(circuit=qc)
    scanner.add_strategy(BehavioralMetrics(qc))

    metrics = scanner.calculate_metrics()

    viz = Visualizer(scanner=scanner, output_dir="qward/examples/img")

    # Single plot
    viz.generate_plot(
        metric_name=Metrics.BEHAVIORAL,
        plot_name=Plots.Behavioral.SUMMARY,
        save=True,
        show=False,
    )

    # Dashboard
    dashboards = viz.create_dashboard(save=True, show=False)
    _ = dashboards


if __name__ == "__main__":
    main()
