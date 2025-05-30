"""
Qiskit metrics implementation for QWARD.

This module provides the QiskitMetrics class for extracting various metrics
from QuantumCircuit objects, supporting both traditional dictionary output
and structured schema-based output with validation.
"""

from typing import Any, Dict

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId
from qward.utils.flatten import flatten_dict

# Import schemas for structured data validation
try:
    from qward.metrics.schemas import (
        QiskitMetricsSchema,
        BasicMetricsSchema,
        InstructionMetricsSchema,
        SchedulingMetricsSchema,
    )

    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False


class QiskitMetrics(MetricCalculator):
    """
    Extract metrics from QuantumCircuit objects.

    This class analyzes quantum circuits and extracts various metrics that are
    directly available from the QuantumCircuit class. It supports both traditional
    dictionary-based output and structured schema-based output with validation.

    Attributes:
        circuit: The quantum circuit to analyze (inherited from MetricCalculator)
    """

    def _get_metric_type(self) -> MetricsType:
        """Get the type of this metric."""
        return MetricsType.PRE_RUNTIME

    def _get_metric_id(self) -> MetricsId:
        """Get the ID of this metric."""
        return MetricsId.QISKIT

    def is_ready(self) -> bool:
        """Check if the metric is ready to be calculated."""
        return self.circuit is not None

    def _ensure_schemas_available(self) -> None:
        """Ensure Pydantic schemas are available, raise ImportError if not."""
        if not SCHEMAS_AVAILABLE:
            raise ImportError(
                "Pydantic schemas are not available. Install pydantic to use structured metrics."
            )

    # =============================================================================
    # Main API Methods
    # =============================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics as a flattened dictionary for DataFrame compatibility.

        Returns:
            Dict[str, Any]: Flattened dictionary containing all metrics
        """
        basic_metrics = self.get_basic_metrics()
        instruction_metrics = self.get_instruction_metrics()
        scheduling_metrics = self.get_scheduling_metrics()

        metrics = {
            "basic_metrics": basic_metrics,
            "instruction_metrics": instruction_metrics,
            "scheduling_metrics": scheduling_metrics,
        }

        # Flatten nested dictionaries for DataFrame compatibility
        to_flatten: Dict[str, Any] = {}
        if "count_ops" in basic_metrics:
            count_ops = basic_metrics.pop("count_ops")
            to_flatten["basic_metrics.count_ops"] = count_ops
        if "instructions" in instruction_metrics:
            instructions = instruction_metrics.pop("instructions")
            to_flatten["instruction_metrics.instructions"] = instructions

        # Flatten and merge
        flat_metrics = flatten_dict(metrics)
        flat_metrics.update(flatten_dict(to_flatten))
        return flat_metrics

    def get_structured_metrics(self) -> QiskitMetricsSchema:
        """
        Get metrics as a structured, validated schema object.

        Returns:
            QiskitMetricsSchema: Complete validated metrics schema

        Raises:
            ImportError: If Pydantic schemas are not available
            ValidationError: If metrics data doesn't match schema constraints
        """
        self._ensure_schemas_available()

        return QiskitMetricsSchema(
            basic_metrics=self.get_structured_basic_metrics(),
            instruction_metrics=self.get_structured_instruction_metrics(),
            scheduling_metrics=self.get_structured_scheduling_metrics(),
        )

    # =============================================================================
    # Basic Metrics
    # =============================================================================

    def get_basic_metrics(self) -> Dict[str, Any]:
        """
        Get basic circuit metrics.

        Returns:
            Dict[str, Any]: Basic metrics including depth, width, size, counts, etc.
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

    def get_structured_basic_metrics(self) -> BasicMetricsSchema:
        """
        Get basic metrics as a validated schema object.

        Returns:
            BasicMetricsSchema: Validated basic metrics
        """
        self._ensure_schemas_available()
        return BasicMetricsSchema(**self.get_basic_metrics())

    # =============================================================================
    # Instruction Metrics
    # =============================================================================

    def get_instruction_metrics(self) -> Dict[str, Any]:
        """
        Get instruction-related circuit metrics.

        Returns:
            Dict[str, Any]: Instruction metrics including connectivity and factors
        """
        circuit = self.circuit

        # Group instructions by operation name
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

    def get_structured_instruction_metrics(self) -> InstructionMetricsSchema:
        """
        Get instruction metrics as a validated schema object.

        Returns:
            InstructionMetricsSchema: Validated instruction metrics
        """
        self._ensure_schemas_available()
        return InstructionMetricsSchema(**self.get_instruction_metrics())

    # =============================================================================
    # Scheduling Metrics
    # =============================================================================

    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """
        Get scheduling-related circuit metrics.

        Returns:
            Dict[str, Any]: Scheduling metrics (empty if circuit not scheduled)
        """
        circuit = self.circuit
        metrics: Dict[str, Any] = {"is_scheduled": False}

        # Check if circuit has scheduling information
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

    def get_structured_scheduling_metrics(self) -> SchedulingMetricsSchema:
        """
        Get scheduling metrics as a validated schema object.

        Returns:
            SchedulingMetricsSchema: Validated scheduling metrics
        """
        self._ensure_schemas_available()
        return SchedulingMetricsSchema(**self.get_scheduling_metrics())
