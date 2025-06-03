"""Tests for the new QWARD visualization API with type-safe constants."""

import unittest
import tempfile
import os
from unittest.mock import patch
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import (
    Visualizer,
    QiskitVisualizer,
    ComplexityVisualizer,
    CircuitPerformanceVisualizer,
    PlotConfig,
    PlotMetadata,
    PlotType,
)
from qward.visualization.constants import Metrics, Plots


class TestNewVisualizationAPI(unittest.TestCase):
    """Tests for the new visualization API with constants and metadata."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test circuit
        self.circuit = QuantumCircuit(2, 2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

        # Create simulator and job
        self.simulator = AerSimulator()
        self.job = self.simulator.run(self.circuit, shots=100)

        # Success criteria
        self.success_criteria = lambda result: result in ["00 00", "11 00"]

        # Create scanner with all metrics
        self.scanner = Scanner(circuit=self.circuit, job=self.job)
        self.scanner.add_strategy(QiskitMetrics(self.circuit))
        self.scanner.add_strategy(ComplexityMetrics(self.circuit))
        self.scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.circuit, job=self.job, success_criteria=self.success_criteria
            )
        )

        # Calculate metrics
        self.metrics_dict = self.scanner.calculate_metrics()

    def test_constants_import_and_access(self):
        """Test that constants can be imported and accessed correctly."""
        # Test Metrics constants
        self.assertEqual(Metrics.QISKIT, "QiskitMetrics")
        self.assertEqual(Metrics.COMPLEXITY, "ComplexityMetrics")
        self.assertEqual(Metrics.CIRCUIT_PERFORMANCE, "CircuitPerformance")

        # Test Plots constants
        self.assertEqual(Plots.Qiskit.CIRCUIT_STRUCTURE, "circuit_structure")
        self.assertEqual(Plots.Qiskit.GATE_DISTRIBUTION, "gate_distribution")
        self.assertEqual(Plots.Complexity.COMPLEXITY_RADAR, "complexity_radar")
        self.assertEqual(
            Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON, "success_error_comparison"
        )

    def test_visualizer_get_available_plots(self):
        """Test get_available_plots method."""
        visualizer = Visualizer(scanner=self.scanner)

        # Test getting all available plots
        all_plots = visualizer.get_available_plots()
        self.assertIsInstance(all_plots, dict)
        self.assertIn(Metrics.QISKIT, all_plots)
        self.assertIn(Metrics.COMPLEXITY, all_plots)

        # Test getting plots for specific metric
        qiskit_plots = visualizer.get_available_plots(Metrics.QISKIT)
        self.assertIsInstance(qiskit_plots, dict)
        self.assertIn(Metrics.QISKIT, qiskit_plots)
        self.assertIsInstance(qiskit_plots[Metrics.QISKIT], list)
        self.assertGreater(len(qiskit_plots[Metrics.QISKIT]), 0)

    def test_visualizer_get_plot_metadata(self):
        """Test get_plot_metadata method."""
        visualizer = Visualizer(scanner=self.scanner)

        # Test getting metadata for QiskitMetrics plot
        metadata = visualizer.get_plot_metadata(Metrics.QISKIT, Plots.Qiskit.CIRCUIT_STRUCTURE)
        self.assertIsInstance(metadata, PlotMetadata)
        self.assertEqual(metadata.name, Plots.Qiskit.CIRCUIT_STRUCTURE)
        self.assertIsNotNone(metadata.description)
        self.assertIsInstance(metadata.plot_type, PlotType)
        self.assertIsInstance(metadata.dependencies, list)

        # Test getting metadata for ComplexityMetrics plot
        metadata = visualizer.get_plot_metadata(
            Metrics.COMPLEXITY, Plots.Complexity.COMPLEXITY_RADAR
        )
        self.assertIsInstance(metadata, PlotMetadata)
        self.assertEqual(metadata.name, Plots.Complexity.COMPLEXITY_RADAR)
        self.assertIsNotNone(metadata.description)

    def test_visualizer_generate_single_plot(self):
        """Test generate_plot method for single plot generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=self.scanner, output_dir=temp_dir)

            # Test generating single plot
            fig = visualizer.generate_plot(
                metric_name=Metrics.QISKIT,
                plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
                save=False,
                show=False,
            )

            self.assertIsNotNone(fig)
            plt.close(fig)  # Clean up

    def test_visualizer_generate_plots_with_selections(self):
        """Test generate_plots method with specific selections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=self.scanner, output_dir=temp_dir)

            # Test generating selected plots
            selections = {
                Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION],
                Metrics.COMPLEXITY: [Plots.Complexity.GATE_BASED_METRICS],
            }

            results = visualizer.generate_plots(selections=selections, save=False, show=False)

            self.assertIsInstance(results, dict)
            self.assertIn(Metrics.QISKIT, results)
            self.assertIn(Metrics.COMPLEXITY, results)

            # Check QiskitMetrics plots
            qiskit_plots = results[Metrics.QISKIT]
            self.assertIsInstance(qiskit_plots, dict)
            self.assertIn(Plots.Qiskit.CIRCUIT_STRUCTURE, qiskit_plots)
            self.assertIn(Plots.Qiskit.GATE_DISTRIBUTION, qiskit_plots)

            # Check ComplexityMetrics plots
            complexity_plots = results[Metrics.COMPLEXITY]
            self.assertIsInstance(complexity_plots, dict)
            self.assertIn(Plots.Complexity.GATE_BASED_METRICS, complexity_plots)

            # Clean up figures
            for metric_plots in results.values():
                for fig in metric_plots.values():
                    plt.close(fig)

    def test_visualizer_generate_all_plots_for_metric(self):
        """Test generate_plots method with None (all plots) for specific metric."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=self.scanner, output_dir=temp_dir)

            # Test generating all plots for QiskitMetrics
            results = visualizer.generate_plots(
                selections={Metrics.QISKIT: None}, save=False, show=False  # None = all plots
            )

            self.assertIsInstance(results, dict)
            self.assertIn(Metrics.QISKIT, results)

            qiskit_plots = results[Metrics.QISKIT]
            self.assertIsInstance(qiskit_plots, dict)
            self.assertGreater(len(qiskit_plots), 0)

            # Verify all expected QiskitMetrics plots are present
            expected_plots = [
                Plots.Qiskit.CIRCUIT_STRUCTURE,
                Plots.Qiskit.GATE_DISTRIBUTION,
                Plots.Qiskit.INSTRUCTION_METRICS,
                Plots.Qiskit.CIRCUIT_SUMMARY,
            ]

            for plot_name in expected_plots:
                self.assertIn(plot_name, qiskit_plots)

            # Clean up figures
            for fig in qiskit_plots.values():
                plt.close(fig)

    def test_individual_visualizer_new_api(self):
        """Test new API methods on individual visualizers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test QiskitVisualizer
            qiskit_viz = QiskitVisualizer(
                metrics_dict={Metrics.QISKIT: self.metrics_dict[Metrics.QISKIT]},
                output_dir=temp_dir,
            )

            # Test class methods
            available_plots = QiskitVisualizer.get_available_plots()
            self.assertIsInstance(available_plots, list)
            self.assertGreater(len(available_plots), 0)

            metadata = QiskitVisualizer.get_plot_metadata(Plots.Qiskit.CIRCUIT_STRUCTURE)
            self.assertIsInstance(metadata, PlotMetadata)

            # Test instance methods
            fig = qiskit_viz.generate_plot(Plots.Qiskit.CIRCUIT_STRUCTURE, save=False, show=False)
            self.assertIsNotNone(fig)
            plt.close(fig)

            # Test generate_plots with list - returns dict mapping plot names to figures
            plots = qiskit_viz.generate_plots(
                [Plots.Qiskit.CIRCUIT_STRUCTURE, Plots.Qiskit.GATE_DISTRIBUTION],
                save=False,
                show=False,
            )
            self.assertIsInstance(plots, dict)
            self.assertEqual(len(plots), 2)
            self.assertIn(Plots.Qiskit.CIRCUIT_STRUCTURE, plots)
            self.assertIn(Plots.Qiskit.GATE_DISTRIBUTION, plots)
            for fig in plots.values():
                plt.close(fig)

            # Test generate_all_plots - returns dict mapping plot names to figures
            all_plots = qiskit_viz.generate_all_plots(save=False, show=False)
            self.assertIsInstance(all_plots, dict)
            self.assertGreater(len(all_plots), 0)
            for fig in all_plots.values():
                plt.close(fig)

    def test_memory_efficient_defaults(self):
        """Test that default parameters are memory efficient."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=self.scanner, output_dir=temp_dir)

            # Count initial files
            initial_files = len(os.listdir(temp_dir))

            # Generate plots with default parameters (should not save)
            visualizer.generate_plots(
                selections={Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE]}
            )  # Default: save=False, show=False

            # Check that no files were created
            final_files = len(os.listdir(temp_dir))
            self.assertEqual(initial_files, final_files)

            # Generate plots with explicit save=True
            visualizer.generate_plots(
                selections={Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE]}, save=True
            )

            # Check that files were created
            final_files_with_save = len(os.listdir(temp_dir))
            self.assertGreater(final_files_with_save, initial_files)

    def test_error_handling_invalid_metric(self):
        """Test error handling for invalid metric names."""
        visualizer = Visualizer(scanner=self.scanner)

        with self.assertRaises(ValueError):
            visualizer.generate_plot(metric_name="InvalidMetric", plot_name="invalid_plot")

    def test_error_handling_invalid_plot(self):
        """Test error handling for invalid plot names."""
        visualizer = Visualizer(scanner=self.scanner)

        with self.assertRaises(ValueError):
            visualizer.generate_plot(metric_name=Metrics.QISKIT, plot_name="invalid_plot")

    def test_error_handling_mismatched_metric_plot(self):
        """Test error handling for mismatched metric and plot combinations."""
        visualizer = Visualizer(scanner=self.scanner)

        # Try to use ComplexityMetrics plot with QiskitMetrics
        with self.assertRaises(ValueError):
            visualizer.generate_plot(
                metric_name=Metrics.QISKIT, plot_name=Plots.Complexity.COMPLEXITY_RADAR
            )

    def test_custom_plot_config(self):
        """Test visualization with custom PlotConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PlotConfig(
                figsize=(10, 8),
                style="quantum",
                color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1"],
                save_format="svg",
                alpha=0.8,
            )

            visualizer = Visualizer(scanner=self.scanner, config=config, output_dir=temp_dir)

            # Generate plot with custom config
            fig = visualizer.generate_plot(
                metric_name=Metrics.QISKIT,
                plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
                save=False,
                show=False,
            )

            self.assertIsNotNone(fig)
            # Check that figure size matches config
            self.assertEqual(fig.get_size_inches().tolist(), [10, 8])
            plt.close(fig)

    def test_dashboard_creation_unchanged(self):
        """Test that dashboard creation still works as before."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=self.scanner, output_dir=temp_dir)

            dashboards = visualizer.create_dashboard(save=False, show=False)

            self.assertIsInstance(dashboards, dict)
            self.assertGreater(len(dashboards), 0)

            # Clean up figures
            for fig in dashboards.values():
                plt.close(fig)

    def test_plot_registry_consistency(self):
        """Test that plot registries are consistent across visualizers."""
        # Test QiskitVisualizer registry
        qiskit_plots = QiskitVisualizer.get_available_plots()
        self.assertIn(Plots.Qiskit.CIRCUIT_STRUCTURE, qiskit_plots)
        self.assertIn(Plots.Qiskit.GATE_DISTRIBUTION, qiskit_plots)

        # Test ComplexityVisualizer registry
        complexity_plots = ComplexityVisualizer.get_available_plots()
        self.assertIn(Plots.Complexity.COMPLEXITY_RADAR, complexity_plots)
        self.assertIn(Plots.Complexity.GATE_BASED_METRICS, complexity_plots)

        # Test CircuitPerformanceVisualizer registry
        perf_plots = CircuitPerformanceVisualizer.get_available_plots()
        self.assertIn(Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON, perf_plots)
        self.assertIn(Plots.CircuitPerformance.FIDELITY_COMPARISON, perf_plots)

    def test_metadata_completeness(self):
        """Test that all plots have complete metadata."""
        visualizers = [QiskitVisualizer, ComplexityVisualizer, CircuitPerformanceVisualizer]

        for visualizer_class in visualizers:
            available_plots = visualizer_class.get_available_plots()

            for plot_name in available_plots:
                metadata = visualizer_class.get_plot_metadata(plot_name)

                # Check required metadata fields
                self.assertIsNotNone(metadata.name)
                self.assertIsNotNone(metadata.method_name)
                self.assertIsNotNone(metadata.description)
                self.assertIsInstance(metadata.plot_type, PlotType)
                self.assertIsNotNone(metadata.filename)
                self.assertIsInstance(metadata.dependencies, list)
                self.assertIsNotNone(metadata.category)

    def test_backward_compatibility(self):
        """Test that old methods still work for backward compatibility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test individual visualizers still have old methods
            qiskit_viz = QiskitVisualizer(
                metrics_dict={Metrics.QISKIT: self.metrics_dict[Metrics.QISKIT]},
                output_dir=temp_dir,
            )

            # Test that old individual plot methods still exist
            self.assertTrue(hasattr(qiskit_viz, "plot_circuit_structure"))
            self.assertTrue(hasattr(qiskit_viz, "plot_gate_distribution"))

            # Test dashboard creation still works
            dashboard = qiskit_viz.create_dashboard(save=False, show=False)
            self.assertIsNotNone(dashboard)
            plt.close(dashboard)

    @patch("matplotlib.pyplot.show")
    def test_show_parameter_functionality(self, mock_show):
        """Test that show parameter controls matplotlib.pyplot.show calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=self.scanner, output_dir=temp_dir)

            # Generate plot with show=False (default)
            fig = visualizer.generate_plot(
                metric_name=Metrics.QISKIT,
                plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
                save=False,
                show=False,
            )
            plt.close(fig)

            # show() should not have been called
            mock_show.assert_not_called()

            # Generate plot with show=True
            fig = visualizer.generate_plot(
                metric_name=Metrics.QISKIT,
                plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
                save=False,
                show=True,
            )
            plt.close(fig)

            # show() should have been called
            mock_show.assert_called()


if __name__ == "__main__":
    unittest.main()
