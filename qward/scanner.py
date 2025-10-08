"""
Scanner class for QWARD.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob

from qward.metrics.base_metric import MetricCalculator


class Scanner:
    """
    Class for analyzing quantum circuits using the Strategy pattern.

    This class provides methods for analyzing quantum circuits using various metric strategies.
    """

    def __init__(
        self,
        circuit: Optional[QuantumCircuit] = None,
        *,
        job: Optional[Union[AerJob, QiskitJob]] = None,
        strategies: Optional[list] = None,
    ):
        """
        Initialize a Scanner object.

        Args:
            circuit: The quantum circuit to analyze
            job: The job that executed the circuit
            strategies: Optional list of metric strategy classes or instances.
                       If a class is provided, it will be instantiated with the circuit.
                       If an instance is provided, its circuit must match the Scanner's circuit.
        """
        self._circuit = circuit
        self._job = job
        self._strategies: List[MetricCalculator] = []

        if strategies is not None:
            for strategy in strategies:
                self._add_strategy_to_list(strategy)

    def _add_strategy_to_list(self, strategy):
        """Helper method to add a strategy to the list with validation."""
        # If strategy is a class (not instance), instantiate with circuit
        if isinstance(strategy, type):
            self._strategies.append(strategy(self._circuit))
            return

        # Strategy is an instance - validate circuit compatibility
        strategy_circuit = getattr(strategy, "circuit", None) or getattr(strategy, "_circuit", None)

        if strategy_circuit is not None and strategy_circuit is not self._circuit:
            raise ValueError(
                f"Strategy instance {strategy.__class__.__name__} was initialized with a different circuit than the Scanner."
            )

        self._strategies.append(strategy)

    @property
    def circuit(self) -> Optional[QuantumCircuit]:
        """
        Get the quantum circuit.

        Returns:
            Optional[QuantumCircuit]: The quantum circuit
        """
        return self._circuit

    @property
    def job(self) -> Optional[Union[AerJob, QiskitJob]]:
        """
        Get the job that executed the circuit.

        Returns:
            Optional[Union[AerJob, QiskitJob]]: The job that executed the circuit
        """
        return self._job

    @property
    def strategies(self) -> List[MetricCalculator]:
        """
        Get the metric strategies.

        Returns:
            List[MetricCalculator]: The metric strategies
        """
        return self._strategies

    def add_strategy(self, strategy: MetricCalculator) -> None:
        """
        Add a metric strategy to the scanner.

        Args:
            strategy: The metric strategy to add
        """
        self._strategies.append(strategy)

    def calculate_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate metrics using all registered strategies.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each metric type.
            For CircuitPerformance metrics, returns two DataFrames:
            - "CircuitPerformance.individual_jobs": DataFrame containing metrics for each job
            - "CircuitPerformance.aggregate": DataFrame containing aggregate metrics across all jobs
        """
        metric_dataframes: Dict[str, pd.DataFrame] = {}

        for strategy in self.strategies:
            metric_results = strategy.get_metrics()
            metric_name = strategy.__class__.__name__

            if metric_name == "CircuitPerformanceMetrics":
                self._process_circuit_performance_metrics(
                    strategy, metric_results, metric_dataframes
                )
            else:
                self._process_standard_metrics(metric_name, metric_results, metric_dataframes)

        return metric_dataframes

    def _process_circuit_performance_metrics(self, strategy, metric_results, metric_dataframes):
        """Process CircuitPerformanceMetrics using schema-based API with proper column separation."""
        display_name = "CircuitPerformance"

        # Check if we have multiple jobs
        jobs_count = len(getattr(strategy, "_jobs", []))

        if hasattr(metric_results, "to_flat_dict"):
            if jobs_count > 1:
                # For multiple jobs, create separate DataFrames with only relevant columns

                # Create aggregate DataFrame with only aggregate-relevant fields
                aggregate_data = self._extract_aggregate_fields(metric_results)
                aggregate_df = pd.DataFrame([aggregate_data])
                metric_dataframes[f"{display_name}.aggregate"] = aggregate_df

                # Create individual jobs DataFrame with only individual-relevant fields
                individual_jobs_data = []
                for job in getattr(strategy, "_jobs", []):
                    # Create a temporary single-job strategy to get individual metrics
                    temp_strategy = strategy.__class__(
                        strategy.circuit,
                        job=job,
                        success_criteria=strategy.success_criteria,
                        expected_distribution=strategy.expected_distribution,
                    )
                    job_metrics = temp_strategy.get_metrics()
                    individual_data = self._extract_individual_fields(job_metrics)
                    individual_jobs_data.append(individual_data)

                if individual_jobs_data:
                    individual_jobs_df = pd.DataFrame(individual_jobs_data)
                    metric_dataframes[f"{display_name}.individual_jobs"] = individual_jobs_df
                else:
                    # Fallback: create empty DataFrame with individual columns
                    individual_data = self._extract_individual_fields(metric_results)
                    metric_dataframes[f"{display_name}.individual_jobs"] = pd.DataFrame(
                        [individual_data]
                    )
            else:
                # Single job case - only create individual_jobs DataFrame with individual fields
                individual_data = self._extract_individual_fields(metric_results)
                single_job_df = pd.DataFrame([individual_data])
                metric_dataframes[f"{display_name}.individual_jobs"] = single_job_df
        else:
            # Fallback for any remaining edge cases
            flattened_metrics = self._flatten_metrics(metric_results)
            single_job_df = pd.DataFrame([flattened_metrics])
            metric_dataframes[f"{display_name}.individual_jobs"] = single_job_df

    def _extract_individual_fields(self, metric_results):
        """Extract only fields relevant for individual job metrics."""
        if hasattr(metric_results, "to_flat_dict"):
            all_fields = metric_results.to_flat_dict()
        else:
            all_fields = self._flatten_metrics(metric_results)

        # Define individual job relevant fields (exclude aggregate fields)
        individual_fields = {}
        for key, value in all_fields.items():
            # Include fields that are NOT aggregate-specific
            if not any(
                aggregate_prefix in key
                for aggregate_prefix in ["mean_", "std_", "min_", "max_", "total_trials"]
            ):
                individual_fields[key] = value

        return individual_fields

    def _extract_aggregate_fields(self, metric_results):
        """Extract only fields relevant for aggregate metrics."""
        if hasattr(metric_results, "to_flat_dict"):
            all_fields = metric_results.to_flat_dict()
        else:
            all_fields = self._flatten_metrics(metric_results)

        # Define aggregate relevant fields (exclude individual-specific fields)
        aggregate_fields = {}
        for key, value in all_fields.items():
            # Include aggregate fields OR common fields like error_rate, method, confidence
            if any(
                aggregate_prefix in key
                for aggregate_prefix in ["mean_", "std_", "min_", "max_", "total_trials"]
            ) or any(
                common_field in key for common_field in ["error_rate", "method", "confidence"]
            ):
                aggregate_fields[key] = value

        return aggregate_fields

    def _process_standard_metrics(self, metric_name, metric_results, metric_dataframes):
        """Process standard metrics using schema-based API."""
        # All metrics should return schema objects with to_flat_dict()
        if hasattr(metric_results, "to_flat_dict"):
            flattened_metrics = metric_results.to_flat_dict()
        else:
            # Fallback for any remaining edge cases
            flattened_metrics = self._flatten_metrics(metric_results)

        df = pd.DataFrame([flattened_metrics])
        metric_dataframes[metric_name] = df

    def _flatten_metrics(self, metric_results):
        """Flatten metric results into a single-level dictionary."""
        flattened_metrics = {}

        if isinstance(metric_results, dict):
            for key, value in metric_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_metrics[f"{key}.{subkey}"] = subvalue
                else:
                    flattened_metrics[key] = value
        elif hasattr(metric_results, "model_dump"):
            # Schema object without to_flat_dict - try model_dump
            metric_dict = metric_results.model_dump()
            for key, value in metric_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_metrics[f"{key}.{subkey}"] = subvalue
                else:
                    flattened_metrics[key] = value
        else:
            # Fallback - treat as single value
            flattened_metrics = {"value": metric_results}

        return flattened_metrics

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        """
        Set the quantum circuit.

        Args:
            circuit: The quantum circuit
        """
        self._circuit = circuit

    def set_job(self, job: Union[AerJob, QiskitJob]) -> None:
        """
        Set the job that executed the circuit.

        Args:
            job: The job that executed the circuit
        """
        self._job = job

    def display_summary(self, metrics_dict: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Display a summary of the calculated metrics.

        Args:
            metrics_dict: Optional metrics dictionary. If None, will calculate metrics first.
        """
        if metrics_dict is None:
            metrics_dict = self.calculate_metrics()

        print("\n" + "=" * 60)
        print("🎯 QWARD ANALYSIS SUMMARY")
        print("=" * 60)

        self._display_circuit_performance_summary(metrics_dict)
        self._display_aggregate_performance_summary(metrics_dict)
        self._display_other_metrics_summary(metrics_dict)

        print("=" * 60)

    def _display_circuit_performance_summary(self, metrics_dict: Dict[str, pd.DataFrame]) -> None:
        """Display circuit performance analysis summary."""
        if "CircuitPerformance.individual_jobs" not in metrics_dict:
            return

        individual_df = metrics_dict["CircuitPerformance.individual_jobs"]
        print("\n📋 Circuit Performance Analysis:")
        print(f"   Jobs analyzed: {len(individual_df)}")

        # Show individual job results
        for i, row in individual_df.iterrows():
            success_rate = row.get("success_metrics.success_rate", 0)
            fidelity = row.get("fidelity_metrics.fidelity", 0)
            total_shots = row.get("success_metrics.total_shots", 0)
            print(
                f"   Job {i+1}: {success_rate:.1%} success, {fidelity:.3f} fidelity ({total_shots} shots)"
            )

    def _display_aggregate_performance_summary(self, metrics_dict: Dict[str, pd.DataFrame]) -> None:
        """Display aggregate performance summary."""
        if "CircuitPerformance.aggregate" not in metrics_dict:
            return

        aggregate_df = metrics_dict["CircuitPerformance.aggregate"]
        if aggregate_df.empty:
            return

        row = aggregate_df.iloc[0]
        mean_success = row.get("success_metrics.mean_success_rate", 0)
        std_success = row.get("success_metrics.std_success_rate", 0)
        total_trials = row.get("success_metrics.total_trials", 0)

        print("\n📊 Overall Performance:")
        print(f"   Average success rate: {mean_success:.1%} ± {std_success:.1%}")
        print(f"   Total measurements: {total_trials}")

        # Performance analysis
        self._display_performance_analysis(mean_success)

    def _display_performance_analysis(self, mean_success: float) -> None:
        """Display performance analysis based on success rate."""
        if mean_success >= 0.95:
            print("   ✅ Excellent performance")
        elif mean_success >= 0.80:
            print("   ⚠️  Good performance with some noise impact")
        elif mean_success >= 0.50:
            print("   ⚠️  Moderate performance - significant noise detected")
        else:
            print("   ❌ Poor performance - high noise or errors")

    def _display_other_metrics_summary(self, metrics_dict: Dict[str, pd.DataFrame]) -> None:
        """Display summary of other metrics."""
        other_metrics = [k for k in metrics_dict.keys() if not k.startswith("CircuitPerformance")]
        if not other_metrics:
            return

        print("\n📈 Additional Metrics:")
        for metric_name in other_metrics:
            df = metrics_dict[metric_name]
            print(f"   {metric_name}: {df.shape[1]} metrics calculated")

            if not df.empty:
                self._display_metric_preview(df)

    def _display_metric_preview(self, df: pd.DataFrame) -> None:
        """Display a preview of the first 5 metrics."""
        first_row = df.iloc[0]
        columns_to_show = list(df.columns)[:5]  # First 5 columns

        print("     Preview (first 5):")
        for col in columns_to_show:
            value = first_row[col]
            formatted_value = self._format_metric_value(value)
            print(f"       {col}: {formatted_value}")

        # Show "..." if there are more columns
        if len(df.columns) > 5:
            remaining = len(df.columns) - 5
            print(f"       ... and {remaining} more metrics")

    def _format_metric_value(self, value) -> str:
        """Format a metric value for display."""
        if isinstance(value, float):
            if 0 < value < 1:
                return f"{value:.3f}"
            else:
                return f"{value:.1f}"
        else:
            return str(value)
