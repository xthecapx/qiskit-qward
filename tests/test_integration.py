"""Integration tests for qward library."""

import unittest
import tempfile
import os
from qiskit import QuantumCircuit
from qiskit.circuit.library.basis_change import QFTGate
from qiskit_aer import AerSimulator
import pandas as pd

from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import (
    Visualizer,
    QiskitVisualizer,
    ComplexityVisualizer,
    CircuitPerformanceVisualizer,
)
from qward.visualization.constants import Metrics, Plots


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete QWARD workflow."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test circuits
        self.bell_circuit = QuantumCircuit(2, 2)
        self.bell_circuit.h(0)
        self.bell_circuit.cx(0, 1)
        self.bell_circuit.measure_all()

        self.ghz_circuit = QuantumCircuit(3, 3)
        self.ghz_circuit.h(0)
        self.ghz_circuit.cx(0, 1)
        self.ghz_circuit.cx(0, 2)
        self.ghz_circuit.measure_all()

        # Create simulator and jobs
        self.simulator = AerSimulator()
        self.bell_job = self.simulator.run(self.bell_circuit, shots=1000)
        self.ghz_job = self.simulator.run(self.ghz_circuit, shots=1000)

        # Success criteria - updated to handle correct format with spaces
        self.bell_success = lambda result: result in ["00 00", "11 00"]
        self.ghz_success = lambda result: result in ["000 000", "111 000"]

    def test_complete_workflow_bell_state(self):
        """Test complete workflow with Bell state circuit."""
        # 1. Create Scanner with all metrics
        scanner = Scanner(circuit=self.bell_circuit, job=self.bell_job)
        scanner.add_strategy(QiskitMetrics(self.bell_circuit))
        scanner.add_strategy(ComplexityMetrics(self.bell_circuit))
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.bell_circuit, job=self.bell_job, success_criteria=self.bell_success
            )
        )

        # 2. Calculate metrics
        results = scanner.calculate_metrics()

        # 3. Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn("QiskitMetrics", results)
        self.assertIn("ComplexityMetrics", results)
        self.assertIn("CircuitPerformance.individual_jobs", results)

        # 4. Verify DataFrame structure
        for df in results.values():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df.columns), 0)
            self.assertEqual(len(df), 1)  # Should have one row

        # 5. Verify specific metrics make sense for Bell state
        qiskit_df = results["QiskitMetrics"]
        self.assertEqual(qiskit_df["basic_metrics.num_qubits"].iloc[0], 2)
        self.assertEqual(qiskit_df["basic_metrics.depth"].iloc[0], 3)  # H, CX, Measure

        complexity_df = results["ComplexityMetrics"]
        self.assertEqual(complexity_df["gate_based_metrics.cnot_count"].iloc[0], 1)
        self.assertGreater(complexity_df["advanced_metrics.parallelism_factor"].iloc[0], 0)

        performance_df = results["CircuitPerformance.individual_jobs"]
        success_rate = performance_df["success_metrics.success_rate"].iloc[0]
        self.assertGreaterEqual(success_rate, 0.8)  # Bell state should have high success rate

    def test_complete_workflow_ghz_state(self):
        """Test complete workflow with GHZ state circuit."""
        # 1. Create Scanner with all metrics
        scanner = Scanner(circuit=self.ghz_circuit, job=self.ghz_job)
        scanner.add_strategy(QiskitMetrics(self.ghz_circuit))
        scanner.add_strategy(ComplexityMetrics(self.ghz_circuit))
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.ghz_circuit, job=self.ghz_job, success_criteria=self.ghz_success
            )
        )

        # 2. Calculate metrics
        results = scanner.calculate_metrics()

        # 3. Verify results structure
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)

        # 4. Verify specific metrics make sense for GHZ state
        qiskit_df = results["QiskitMetrics"]
        self.assertEqual(qiskit_df["basic_metrics.num_qubits"].iloc[0], 3)
        self.assertGreater(qiskit_df["basic_metrics.depth"].iloc[0], 2)

        complexity_df = results["ComplexityMetrics"]
        self.assertEqual(complexity_df["gate_based_metrics.cnot_count"].iloc[0], 2)  # Two CX gates

    def test_schema_based_workflow(self):
        """Test workflow using schema-based API directly."""
        # 1. Create metric calculators
        qiskit_metrics = QiskitMetrics(self.bell_circuit)
        complexity_metrics = ComplexityMetrics(self.bell_circuit)
        performance_metrics = CircuitPerformanceMetrics(
            circuit=self.bell_circuit, job=self.bell_job, success_criteria=self.bell_success
        )

        # 2. Get schema objects
        qiskit_schema = qiskit_metrics.get_metrics()
        complexity_schema = complexity_metrics.get_metrics()
        performance_schema = performance_metrics.get_metrics()

        # 3. Verify schema types
        from qward.metrics.schemas import (
            QiskitMetricsSchema,
            ComplexityMetricsSchema,
            CircuitPerformanceSchema,
        )

        self.assertIsInstance(qiskit_schema, QiskitMetricsSchema)
        self.assertIsInstance(complexity_schema, ComplexityMetricsSchema)
        self.assertIsInstance(performance_schema, CircuitPerformanceSchema)

        # 4. Verify type-safe access
        self.assertEqual(qiskit_schema.basic_metrics.num_qubits, 2)
        self.assertGreater(complexity_schema.advanced_metrics.parallelism_factor, 0)
        self.assertGreaterEqual(performance_schema.success_metrics.success_rate, 0.0)
        self.assertLessEqual(performance_schema.success_metrics.success_rate, 1.0)

        # 5. Verify cross-field validation
        expected_error_rate = 1.0 - performance_schema.success_metrics.success_rate
        self.assertAlmostEqual(
            performance_schema.success_metrics.error_rate, expected_error_rate, places=5
        )

    def test_multiple_circuits_comparison(self):
        """Test comparing metrics across multiple circuits."""
        # Create QFT circuit
        qft_circuit = QuantumCircuit(3)
        qft_circuit.append(QFTGate(3), range(3))

        # Create a simple variational circuit
        variational_circuit = QuantumCircuit(2)
        variational_circuit.ry(0.5, 0)
        variational_circuit.ry(0.3, 1)
        variational_circuit.cx(0, 1)
        variational_circuit.ry(0.7, 0)
        variational_circuit.ry(0.2, 1)

        circuits = {
            "Bell": self.bell_circuit,
            "GHZ": self.ghz_circuit,
            "QFT": qft_circuit,
            "Variational": variational_circuit,
        }

        results = {}

        for name, circuit in circuits.items():
            # Calculate QiskitMetrics and ComplexityMetrics for each
            qiskit_metrics = QiskitMetrics(circuit)
            complexity_metrics = ComplexityMetrics(circuit)

            qiskit_schema = qiskit_metrics.get_metrics()
            complexity_schema = complexity_metrics.get_metrics()

            results[name] = {
                "num_qubits": qiskit_schema.basic_metrics.num_qubits,
                "depth": qiskit_schema.basic_metrics.depth,
                "gate_count": complexity_schema.gate_based_metrics.gate_count,
                "weighted_complexity": complexity_schema.derived_metrics.weighted_complexity,
            }

        # Verify results make sense
        self.assertEqual(results["Bell"]["num_qubits"], 2)
        self.assertEqual(results["GHZ"]["num_qubits"], 3)
        self.assertEqual(results["QFT"]["num_qubits"], 3)
        self.assertEqual(results["Variational"]["num_qubits"], 2)

        # Different circuits have different complexities - just verify they're reasonable
        # QFT might be optimized differently, so just check that all have positive values
        for name, metrics in results.items():
            self.assertGreater(metrics["depth"], 0)
            self.assertGreater(metrics["gate_count"], 0)
            self.assertGreater(metrics["weighted_complexity"], 0)

    def test_visualization_integration(self):
        """Test integration with visualization system."""
        # 1. Calculate metrics
        scanner = Scanner(circuit=self.bell_circuit, job=self.bell_job)
        scanner.add_strategy(QiskitMetrics(self.bell_circuit))
        scanner.add_strategy(ComplexityMetrics(self.bell_circuit))
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.bell_circuit, job=self.bell_job, success_criteria=self.bell_success
            )
        )

        results = scanner.calculate_metrics()

        # 2. Test individual visualizers with new API
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test QiskitVisualizer with new API
            qiskit_viz = QiskitVisualizer(
                metrics_dict={"QiskitMetrics": results["QiskitMetrics"]}, output_dir=temp_dir
            )
            qiskit_plots = qiskit_viz.generate_all_plots(save=True, show=False)
            self.assertIsInstance(qiskit_plots, dict)
            self.assertGreater(len(qiskit_plots), 0)

            # Test ComplexityVisualizer with new API
            complexity_viz = ComplexityVisualizer(
                metrics_dict={"ComplexityMetrics": results["ComplexityMetrics"]},
                output_dir=temp_dir,
            )
            complexity_plots = complexity_viz.generate_all_plots(save=True, show=False)
            self.assertIsInstance(complexity_plots, dict)
            self.assertGreater(len(complexity_plots), 0)

            # Test CircuitPerformanceVisualizer with new API
            perf_data = {k: v for k, v in results.items() if k.startswith("CircuitPerformance")}
            perf_viz = CircuitPerformanceVisualizer(metrics_dict=perf_data, output_dir=temp_dir)
            perf_plots = perf_viz.generate_all_plots(save=True, show=False)
            self.assertIsInstance(perf_plots, dict)
            self.assertGreater(len(perf_plots), 0)

            # Verify files were created
            files_created = os.listdir(temp_dir)
            self.assertGreater(len(files_created), 0)

    def test_unified_visualizer_integration(self):
        """Test integration with unified Visualizer using new API."""
        # 1. Calculate metrics
        scanner = Scanner(circuit=self.bell_circuit, job=self.bell_job)
        scanner.add_strategy(QiskitMetrics(self.bell_circuit))
        scanner.add_strategy(ComplexityMetrics(self.bell_circuit))
        scanner.add_strategy(
            CircuitPerformanceMetrics(
                circuit=self.bell_circuit, job=self.bell_job, success_criteria=self.bell_success
            )
        )

        # 2. Test unified visualizer with new API
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = Visualizer(scanner=scanner, output_dir=temp_dir)

            # Test dashboard creation (unchanged)
            dashboards = visualizer.create_dashboard(save=True, show=False)
            self.assertIsInstance(dashboards, dict)
            self.assertGreater(len(dashboards), 0)

            # Test new API: generate_plots with type-safe constants
            selected_plots = visualizer.generate_plots(
                selections={
                    Metrics.QISKIT: [
                        Plots.Qiskit.CIRCUIT_STRUCTURE,
                        Plots.Qiskit.GATE_DISTRIBUTION,
                    ],
                    Metrics.COMPLEXITY: [Plots.Complexity.COMPLEXITY_RADAR],
                },
                save=True,
                show=False,
            )

            self.assertIsInstance(selected_plots, dict)
            self.assertIn(Metrics.QISKIT, selected_plots)
            self.assertIn(Metrics.COMPLEXITY, selected_plots)
            self.assertGreater(len(selected_plots[Metrics.QISKIT]), 0)
            self.assertGreater(len(selected_plots[Metrics.COMPLEXITY]), 0)

            # Test new API: generate all plots for specific metric
            all_qiskit_plots = visualizer.generate_plots(
                selections={Metrics.QISKIT: None}, save=True, show=False  # None = all plots
            )

            self.assertIsInstance(all_qiskit_plots, dict)
            self.assertIn(Metrics.QISKIT, all_qiskit_plots)
            self.assertGreater(len(all_qiskit_plots[Metrics.QISKIT]), 0)

            # Test new API: single plot generation
            single_plot = visualizer.generate_plot(
                metric_name=Metrics.QISKIT,
                plot_name=Plots.Qiskit.CIRCUIT_STRUCTURE,
                save=True,
                show=False,
            )
            self.assertIsNotNone(single_plot)

            # Test available metrics and plots
            available_metrics = visualizer.get_available_metrics()
            self.assertIn(Metrics.QISKIT, available_metrics)
            self.assertIn(Metrics.COMPLEXITY, available_metrics)

            # Test plot metadata
            available_plots = visualizer.get_available_plots()
            self.assertIn(Metrics.QISKIT, available_plots)

            for plot_name in available_plots[Metrics.QISKIT]:
                metadata = visualizer.get_plot_metadata(Metrics.QISKIT, plot_name)
                self.assertIsNotNone(metadata.description)
                self.assertIsNotNone(metadata.plot_type)

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow."""
        # Test with invalid circuit - should handle gracefully
        scanner = Scanner(circuit=None)
        # Don't add strategies that require a circuit

        results = scanner.calculate_metrics()
        # Should handle gracefully and return empty results
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 0)

        # Test with circuit but no job for CircuitPerformanceMetrics
        scanner = Scanner(circuit=self.bell_circuit)
        # Create CircuitPerformanceMetrics without job - it won't be ready
        perf_metrics = CircuitPerformanceMetrics(circuit=self.bell_circuit)
        # Only add it if it's ready (it won't be without a job)
        if perf_metrics.is_ready():
            scanner.add_strategy(perf_metrics)

        results = scanner.calculate_metrics()
        # Should handle gracefully - CircuitPerformanceMetrics won't be added if not ready
        self.assertIsInstance(results, dict)

    def test_large_circuit_performance(self):
        """Test performance with larger circuits."""
        # Create a larger circuit
        large_circuit = QuantumCircuit(8, 8)
        for i in range(8):
            large_circuit.h(i)
        for i in range(7):
            large_circuit.cx(i, i + 1)
        large_circuit.measure_all()

        # Test that it can handle larger circuits
        scanner = Scanner(circuit=large_circuit)
        scanner.add_strategy(QiskitMetrics(large_circuit))
        scanner.add_strategy(ComplexityMetrics(large_circuit))

        results = scanner.calculate_metrics()

        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn("QiskitMetrics", results)
        self.assertIn("ComplexityMetrics", results)

        qiskit_df = results["QiskitMetrics"]
        self.assertEqual(qiskit_df["basic_metrics.num_qubits"].iloc[0], 8)

        complexity_df = results["ComplexityMetrics"]
        # Updated expectation - actual gate count includes barriers and measures
        actual_gate_count = complexity_df["gate_based_metrics.gate_count"].iloc[0]
        self.assertGreaterEqual(actual_gate_count, 15)  # At least 8 H + 7 CX

    def test_constructor_strategies_integration(self):
        """Test Scanner constructor with strategies."""
        # Test with strategy classes
        scanner = Scanner(circuit=self.bell_circuit, strategies=[QiskitMetrics, ComplexityMetrics])

        results = scanner.calculate_metrics()

        self.assertIsInstance(results, dict)
        self.assertIn("QiskitMetrics", results)
        self.assertIn("ComplexityMetrics", results)

        # Test with strategy instances
        qiskit_metrics = QiskitMetrics(self.bell_circuit)
        complexity_metrics = ComplexityMetrics(self.bell_circuit)

        scanner = Scanner(
            circuit=self.bell_circuit, strategies=[qiskit_metrics, complexity_metrics]
        )

        results = scanner.calculate_metrics()

        self.assertIsInstance(results, dict)
        self.assertIn("QiskitMetrics", results)
        self.assertIn("ComplexityMetrics", results)

    def test_flat_dict_conversion_integration(self):
        """Test flat dictionary conversion integration."""
        # Get schema objects
        qiskit_metrics = QiskitMetrics(self.bell_circuit)
        complexity_metrics = ComplexityMetrics(self.bell_circuit)

        qiskit_schema = qiskit_metrics.get_metrics()
        complexity_schema = complexity_metrics.get_metrics()

        # Convert to flat dictionaries
        qiskit_flat = qiskit_schema.to_flat_dict()
        complexity_flat = complexity_schema.to_flat_dict()

        # Verify structure
        self.assertIsInstance(qiskit_flat, dict)
        self.assertIsInstance(complexity_flat, dict)

        # Verify expected keys
        self.assertIn("basic_metrics.depth", qiskit_flat)
        self.assertIn("basic_metrics.num_qubits", qiskit_flat)
        self.assertIn("gate_based_metrics.gate_count", complexity_flat)
        self.assertIn("advanced_metrics.parallelism_factor", complexity_flat)

        # Verify values match schema access
        self.assertEqual(qiskit_flat["basic_metrics.depth"], qiskit_schema.basic_metrics.depth)
        self.assertEqual(
            complexity_flat["gate_based_metrics.gate_count"],
            complexity_schema.gate_based_metrics.gate_count,
        )


if __name__ == "__main__":
    unittest.main()
