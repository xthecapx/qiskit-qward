"""
Qiskit metrics implementation for QWARD.

This module provides the QiskitMetrics class for extracting various metrics
from QuantumCircuit objects using structured schema-based output with validation.
"""

from typing import Any, Dict

from qward.metrics.base_metric import MetricCalculator
from qward.metrics.types import MetricsType, MetricsId

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
    directly available from the QuantumCircuit class using structured schema-based
    output with validation.

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

    def get_metrics(self) -> QiskitMetricsSchema:
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
            basic_metrics=self.get_basic_metrics(),
            instruction_metrics=self.get_instruction_metrics(),
            scheduling_metrics=self.get_scheduling_metrics(),
        )

    # =============================================================================
    # Primary Structured API Methods
    # =============================================================================

    def get_basic_metrics(self) -> BasicMetricsSchema:
        """
        Get basic circuit metrics as a validated schema object.

        Returns:
            BasicMetricsSchema: Validated basic metrics including depth, width, size, counts, etc.
        """
        self._ensure_schemas_available()

        circuit = self.circuit

        # Check if calibrations attribute exists (removed in Qiskit 2.0)
        has_calibrations = (
            hasattr(circuit, "calibrations") and bool(circuit.calibrations)
            if hasattr(circuit, "calibrations")
            else False
        )

        # Check if layout attribute exists
        has_layout = hasattr(circuit, "layout") and bool(circuit.layout)

        basic_data = {
            "depth": circuit.depth(),
            "width": circuit.width(),
            "size": circuit.size(),
            "count_ops": circuit.count_ops(),
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "num_ancillas": circuit.num_ancillas,
            "num_parameters": circuit.num_parameters,
            "has_calibrations": has_calibrations,
            "has_layout": has_layout,
        }

        return BasicMetricsSchema(**basic_data)

    def get_instruction_metrics(self) -> InstructionMetricsSchema:
        """
        Get instruction-related circuit metrics as a validated schema object.

        Returns:
            InstructionMetricsSchema: Validated instruction metrics including connectivity and factors
        """
        self._ensure_schemas_available()

        circuit = self.circuit

        # Group instructions by operation name
        instructions = {}
        for name in circuit.count_ops().keys():
            instructions[name] = circuit.get_instructions(name)

        instruction_data = {
            "instructions": instructions,
            "num_connected_components": circuit.num_connected_components(),
            "num_nonlocal_gates": circuit.num_nonlocal_gates(),
            "num_tensor_factors": circuit.num_tensor_factors(),
            "num_unitary_factors": circuit.num_unitary_factors(),
        }

        return InstructionMetricsSchema(**instruction_data)

    def get_scheduling_metrics(self) -> SchedulingMetricsSchema:
        """
        Get scheduling-related circuit metrics as a validated schema object.

        Returns:
            SchedulingMetricsSchema: Validated scheduling metrics (empty if circuit not scheduled)
        """
        self._ensure_schemas_available()

        circuit = self.circuit
        metrics: Dict[str, Any] = {"is_scheduled": False}

        # Check if circuit has scheduling information
        if hasattr(circuit, "op_start_times") and circuit.op_start_times is not None:
            # Use the new estimate_duration() method instead of deprecated properties
            try:
                duration = circuit.estimate_duration()
                metrics.update(
                    {
                        "is_scheduled": True,
                        "layout": circuit.layout,
                        "op_start_times": circuit.op_start_times,
                        "qubit_duration": duration,
                        "qubit_start_time": 0,  # Start time is typically 0 for scheduled circuits
                        "qubit_stop_time": duration,  # Stop time equals duration
                    }
                )
            except Exception:
                # Fallback if estimate_duration() is not available or fails
                metrics.update(
                    {
                        "is_scheduled": True,
                        "layout": circuit.layout,
                        "op_start_times": circuit.op_start_times,
                        "qubit_duration": None,
                        "qubit_start_time": None,
                        "qubit_stop_time": None,
                    }
                )

        return SchedulingMetricsSchema(**metrics)
