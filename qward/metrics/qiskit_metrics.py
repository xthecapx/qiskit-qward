"""
Qiskit metrics implementation for QWARD.
"""

from typing import Any, Dict

from qward.metrics.base_metric import Metric
from qward.metrics.types import MetricsType, MetricsId
from qward.utils.flatten import flatten_dict


class QiskitMetrics(Metric):
    """
    Class for extracting metrics from QuantumCircuit objects.

    This class provides methods for analyzing quantum circuits and extracting
    various metrics that are directly available from the QuantumCircuit class.
    """

    def _get_metric_type(self) -> MetricsType:
        """
        Get the type of this metric.

        Returns:
            MetricsType: The type of this metric
        """
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """
        Get the ID of this metric.

        Returns:
            MetricsId: The ID of this metric
        """
        return MetricsId.QISKIT

    def is_ready(self) -> bool:
        """
        Check if the metric is ready to be calculated.

        Returns:
            bool: True if the metric is ready to be calculated, False otherwise
        """
        return self.circuit is not None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the metrics, flattening nested dictionaries for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Dictionary containing the metrics (flattened)
        """
        metrics = {
            "basic_metrics": self.get_basic_metrics(),
            "instruction_metrics": self.get_instruction_metrics(),
            "scheduling_metrics": self.get_scheduling_metrics(),
        }
        # Flatten nested dicts for count_ops and instructions
        to_flatten = {}
        if "count_ops" in metrics["basic_metrics"]:
            count_ops = metrics["basic_metrics"].pop("count_ops")
            to_flatten.update({"basic_metrics.count_ops": count_ops})
        if "instructions" in metrics["instruction_metrics"]:
            instructions = metrics["instruction_metrics"].pop("instructions")
            to_flatten.update({"instruction_metrics.instructions": instructions})
        # Flatten and merge
        flat_metrics = flatten_dict(metrics)
        flat_metrics.update(flatten_dict(to_flatten))
        return flat_metrics

    def get_basic_metrics(self) -> Dict[str, Any]:
        """
        Get basic metrics about the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing basic metrics
        """
        circuit = self.circuit

        return {
            "depth": circuit.depth(),
            "width": circuit.width(),
            "size": circuit.size(),
            "count_ops": circuit.count_ops(),
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "num_ancillas": circuit.num_ancillas,
            "num_parameters": circuit.num_parameters,
            "has_calibrations": bool(circuit.calibrations),
            "has_layout": bool(circuit.layout),
        }

    def get_instruction_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the instructions in the circuit.

        Returns:
            Dict[str, Any]: Dictionary containing instruction metrics
        """
        circuit = self.circuit

        instructions = {}
        for name in circuit.count_ops().keys():
            instructions[name] = circuit.get_instructions(name)

        return {
            "instructions": instructions,
            "num_connected_components": circuit.num_connected_components(),
            "num_nonlocal_gates": circuit.num_nonlocal_gates(),
            "num_tensor_factors": circuit.num_tensor_factors(),
            "num_unitary_factors": circuit.num_unitary_factors(),
        }

    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the scheduling of the circuit.
        If the circuit is not scheduled, returns a dictionary with is_scheduled=False.

        Returns:
            Dict[str, Any]: Dictionary containing scheduling metrics
        """
        circuit = self.circuit
        metrics = {"is_scheduled": False}

        # Check if circuit is scheduled by checking if op_start_times is not None
        if hasattr(circuit, "op_start_times") and circuit.op_start_times is not None:
            metrics.update(
                {
                    "is_scheduled": True,
                    "layout": circuit.layout,
                    "op_start_times": circuit.op_start_times,
                    "qubit_duration": circuit.qubit_duration,
                    "qubit_start_time": circuit.qubit_start_time,
                    "qubit_stop_time": circuit.qubit_stop_time,
                }
            )

        return metrics
