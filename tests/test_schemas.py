"""Tests for qward schema validation system."""

import unittest
import json
from pydantic import ValidationError
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Instruction

from qward.metrics.schemas import (
    QiskitMetricsSchema,
    ComplexityMetricsSchema,
    CircuitPerformanceSchema,
    BasicMetricsSchema,
    GateBasedMetricsSchema,
    SuccessMetricsSchema,
    FidelityMetricsSchema,
    StatisticalMetricsSchema,
)


class TestSchemaValidation(unittest.TestCase):
    """Tests for schema validation system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample CircuitInstruction objects for testing
        self.sample_circuit = QuantumCircuit(2)
        self.sample_circuit.h(0)
        self.sample_circuit.cx(0, 1)

        # Extract instructions for use in tests
        self.h_instruction = [
            inst for inst in self.sample_circuit.data if inst.operation.name == "h"
        ]
        self.cx_instruction = [
            inst for inst in self.sample_circuit.data if inst.operation.name == "cx"
        ]

    def test_basic_metrics_schema_valid(self):
        """Test BasicMetricsSchema with valid data."""
        valid_data = {
            "depth": 5,
            "width": 3,
            "size": 10,
            "num_qubits": 3,
            "num_clbits": 2,
            "num_ancillas": 0,
            "num_parameters": 0,
            "has_calibrations": False,
            "has_layout": False,
            "count_ops": {"h": 2, "cx": 3},
        }

        schema = BasicMetricsSchema(**valid_data)

        self.assertEqual(schema.depth, 5)
        self.assertEqual(schema.width, 3)
        self.assertEqual(schema.size, 10)
        self.assertEqual(schema.num_qubits, 3)
        self.assertEqual(schema.num_clbits, 2)

    def test_basic_metrics_schema_invalid(self):
        """Test BasicMetricsSchema with invalid data."""
        invalid_data = {
            "depth": -1,  # Invalid: negative depth
            "width": 3,
            "size": 10,
            "num_qubits": 3,
            "num_clbits": 2,
            "num_ancillas": 0,
            "num_parameters": 0,
            "has_calibrations": False,
            "has_layout": False,
            "count_ops": {"h": 2, "cx": 3},
        }

        with self.assertRaises(ValidationError):
            BasicMetricsSchema(**invalid_data)

    def test_gate_based_metrics_schema_valid(self):
        """Test GateBasedMetricsSchema with valid data."""
        valid_data = {
            "gate_count": 10,
            "circuit_depth": 5,
            "cnot_count": 3,
            "t_count": 2,
            "two_qubit_count": 4,
            "multi_qubit_ratio": 0.4,
        }

        schema = GateBasedMetricsSchema(**valid_data)

        self.assertEqual(schema.gate_count, 10)
        self.assertEqual(schema.circuit_depth, 5)
        self.assertEqual(schema.cnot_count, 3)
        self.assertEqual(schema.t_count, 2)
        self.assertEqual(schema.two_qubit_count, 4)
        self.assertEqual(schema.multi_qubit_ratio, 0.4)

    def test_gate_based_metrics_schema_ratio_validation(self):
        """Test GateBasedMetricsSchema ratio validation."""
        # Test invalid ratio > 1.0
        invalid_data = {
            "gate_count": 10,
            "circuit_depth": 5,
            "cnot_count": 3,
            "t_count": 2,
            "two_qubit_count": 4,
            "multi_qubit_ratio": 1.5,  # Invalid: > 1.0
        }

        with self.assertRaises(ValidationError):
            GateBasedMetricsSchema(**invalid_data)

    def test_success_metrics_schema_valid(self):
        """Test SuccessMetricsSchema with valid data."""
        valid_data = {
            "success_rate": 0.85,
            "error_rate": 0.15,
            "total_shots": 1000,
            "successful_shots": 850,
        }

        schema = SuccessMetricsSchema(**valid_data)

        self.assertEqual(schema.success_rate, 0.85)
        self.assertEqual(schema.error_rate, 0.15)
        self.assertEqual(schema.total_shots, 1000)
        self.assertEqual(schema.successful_shots, 850)

    def test_success_metrics_schema_error_rate_validation(self):
        """Test SuccessMetricsSchema error rate validation."""
        # Test valid error_rate = 1 - success_rate
        valid_data = {
            "success_rate": 0.85,
            "error_rate": 0.15,  # Valid: 1 - 0.85 = 0.15
            "total_shots": 1000,
            "successful_shots": 850,
        }

        schema = SuccessMetricsSchema(**valid_data)
        self.assertEqual(schema.error_rate, 0.15)

    def test_success_metrics_schema_shots_validation(self):
        """Test SuccessMetricsSchema shots validation."""
        # Test invalid successful_shots > total_shots
        invalid_data = {
            "success_rate": 0.85,
            "error_rate": 0.15,
            "total_shots": 1000,
            "successful_shots": 1100,  # Invalid: > total_shots
        }

        with self.assertRaises(ValidationError):
            SuccessMetricsSchema(**invalid_data)

    def test_fidelity_metrics_schema_valid(self):
        """Test FidelityMetricsSchema with valid data."""
        valid_data = {
            "fidelity": 0.92,
            "has_expected_distribution": True,
            "method": "theoretical_comparison",
            "confidence": "high",
        }

        schema = FidelityMetricsSchema(**valid_data)

        self.assertEqual(schema.fidelity, 0.92)
        self.assertEqual(schema.has_expected_distribution, True)
        self.assertEqual(schema.method, "theoretical_comparison")
        self.assertEqual(schema.confidence, "high")

    def test_fidelity_metrics_schema_invalid_range(self):
        """Test FidelityMetricsSchema with invalid fidelity range."""
        invalid_data = {
            "fidelity": 1.5,  # Invalid: > 1.0
            "has_expected_distribution": True,
            "method": "theoretical_comparison",
            "confidence": "high",
        }

        with self.assertRaises(ValidationError):
            FidelityMetricsSchema(**invalid_data)

    def test_statistical_metrics_schema_valid(self):
        """Test StatisticalMetricsSchema with valid data."""
        valid_data = {
            "entropy": 1.8,
            "uniformity": 0.75,
            "concentration": 0.25,
            "dominant_outcome_probability": 0.6,
            "num_unique_outcomes": 4,
        }

        schema = StatisticalMetricsSchema(**valid_data)

        self.assertEqual(schema.entropy, 1.8)
        self.assertEqual(schema.uniformity, 0.75)
        self.assertEqual(schema.concentration, 0.25)
        self.assertEqual(schema.dominant_outcome_probability, 0.6)
        self.assertEqual(schema.num_unique_outcomes, 4)

    def test_qiskit_metrics_schema_complete(self):
        """Test complete QiskitMetricsSchema."""
        valid_data = {
            "basic_metrics": {
                "depth": 5,
                "width": 3,
                "size": 10,
                "num_qubits": 3,
                "num_clbits": 2,
                "num_ancillas": 0,
                "num_parameters": 0,
                "has_calibrations": False,
                "has_layout": False,
                "count_ops": {"h": 2, "cx": 3},
            },
            "instruction_metrics": {
                "num_connected_components": 1,
                "num_nonlocal_gates": 4,
                "num_tensor_factors": 1,
                "num_unitary_factors": 1,
                "instructions": {"h": self.h_instruction, "cx": self.cx_instruction},
            },
            "scheduling_metrics": {
                "is_scheduled": False,
                "layout": None,
                "op_start_times": None,
                "qubit_duration": None,
                "qubit_start_time": None,
                "qubit_stop_time": None,
            },
        }

        schema = QiskitMetricsSchema(**valid_data)

        self.assertEqual(schema.basic_metrics.depth, 5)
        self.assertEqual(schema.instruction_metrics.num_nonlocal_gates, 4)
        self.assertEqual(schema.scheduling_metrics.is_scheduled, False)

    def test_complexity_metrics_schema_complete(self):
        """Test complete ComplexityMetricsSchema."""
        valid_data = {
            "gate_based_metrics": {
                "gate_count": 10,
                "circuit_depth": 5,
                "cnot_count": 3,
                "t_count": 2,
                "two_qubit_count": 4,
                "multi_qubit_ratio": 0.4,
            },
            "entanglement_metrics": {"entangling_gate_density": 0.6, "entangling_width": 2},
            "standardized_metrics": {
                "circuit_volume": 50,
                "gate_density": 2.0,
                "clifford_ratio": 0.8,
                "non_clifford_ratio": 0.2,
            },
            "advanced_metrics": {
                "parallelism_factor": 0.7,
                "circuit_efficiency": 0.85,
                "parallelism_efficiency": 0.6,
                "quantum_resource_utilization": 0.9,
            },
            "derived_metrics": {
                "square_ratio": 0.8,
                "weighted_complexity": 15,
                "normalized_weighted_complexity": 5.5,
            },
        }

        schema = ComplexityMetricsSchema(**valid_data)

        self.assertEqual(schema.gate_based_metrics.gate_count, 10)
        self.assertEqual(schema.entanglement_metrics.entangling_gate_density, 0.6)
        self.assertEqual(schema.standardized_metrics.circuit_volume, 50)

    def test_circuit_performance_schema_complete(self):
        """Test complete CircuitPerformanceSchema."""
        valid_data = {
            "success_metrics": {
                "success_rate": 0.85,
                "error_rate": 0.15,
                "total_shots": 1000,
                "successful_shots": 850,
            },
            "fidelity_metrics": {
                "fidelity": 0.92,
                "has_expected_distribution": True,
                "method": "theoretical_comparison",
                "confidence": "high",
            },
            "statistical_metrics": {
                "entropy": 1.8,
                "uniformity": 0.75,
                "concentration": 0.25,
                "dominant_outcome_probability": 0.6,
                "num_unique_outcomes": 4,
            },
        }

        schema = CircuitPerformanceSchema(**valid_data)

        self.assertEqual(schema.success_metrics.success_rate, 0.85)
        self.assertEqual(schema.fidelity_metrics.fidelity, 0.92)
        self.assertEqual(schema.statistical_metrics.entropy, 1.8)

    def test_to_flat_dict_qiskit_metrics(self):
        """Test to_flat_dict for QiskitMetricsSchema."""
        valid_data = {
            "basic_metrics": {
                "depth": 5,
                "width": 3,
                "size": 10,
                "num_qubits": 3,
                "num_clbits": 2,
                "num_ancillas": 0,
                "num_parameters": 0,
                "has_calibrations": False,
                "has_layout": False,
                "count_ops": {"h": 2, "cx": 3},
            },
            "instruction_metrics": {
                "num_connected_components": 1,
                "num_nonlocal_gates": 4,
                "num_tensor_factors": 1,
                "num_unitary_factors": 1,
                "instructions": {"h": self.h_instruction, "cx": self.cx_instruction},
            },
            "scheduling_metrics": {
                "is_scheduled": False,
                "layout": None,
                "op_start_times": None,
                "qubit_duration": None,
                "qubit_start_time": None,
                "qubit_stop_time": None,
            },
        }

        schema = QiskitMetricsSchema(**valid_data)
        flat_dict = schema.to_flat_dict()

        self.assertIsInstance(flat_dict, dict)
        self.assertIn("basic_metrics.depth", flat_dict)
        self.assertIn("instruction_metrics.num_nonlocal_gates", flat_dict)
        self.assertIn("scheduling_metrics.is_scheduled", flat_dict)

    def test_to_flat_dict_complexity_metrics(self):
        """Test to_flat_dict for ComplexityMetricsSchema."""
        valid_data = {
            "gate_based_metrics": {
                "gate_count": 10,
                "circuit_depth": 5,
                "cnot_count": 3,
                "t_count": 2,
                "two_qubit_count": 4,
                "multi_qubit_ratio": 0.4,
            },
            "entanglement_metrics": {"entangling_gate_density": 0.6, "entangling_width": 2},
            "standardized_metrics": {
                "circuit_volume": 50,
                "gate_density": 2.0,
                "clifford_ratio": 0.8,
                "non_clifford_ratio": 0.2,
            },
            "advanced_metrics": {
                "parallelism_factor": 0.7,
                "circuit_efficiency": 0.85,
                "parallelism_efficiency": 0.6,
                "quantum_resource_utilization": 0.9,
            },
            "derived_metrics": {
                "square_ratio": 0.8,
                "weighted_complexity": 15,
                "normalized_weighted_complexity": 5.5,
            },
        }

        schema = ComplexityMetricsSchema(**valid_data)
        flat_dict = schema.to_flat_dict()

        self.assertIsInstance(flat_dict, dict)
        self.assertIn("gate_based_metrics.gate_count", flat_dict)
        self.assertIn("entanglement_metrics.entangling_gate_density", flat_dict)

    def test_edge_case_zero_values(self):
        """Test edge cases with zero values."""
        valid_data = {
            "depth": 0,
            "width": 0,
            "size": 0,
            "num_qubits": 0,
            "num_clbits": 0,
            "num_ancillas": 0,
            "num_parameters": 0,
            "has_calibrations": False,
            "has_layout": False,
            "count_ops": {},
        }

        schema = BasicMetricsSchema(**valid_data)

        self.assertEqual(schema.depth, 0)
        self.assertEqual(schema.size, 0)

    def test_edge_case_boundary_ratios(self):
        """Test edge cases with boundary ratio values."""
        valid_data = {
            "gate_count": 10,
            "circuit_depth": 5,
            "cnot_count": 3,
            "t_count": 2,
            "two_qubit_count": 4,
            "multi_qubit_ratio": 0.0,  # Boundary value
        }

        schema = GateBasedMetricsSchema(**valid_data)

        self.assertEqual(schema.multi_qubit_ratio, 0.0)

        # Test upper boundary
        valid_data["multi_qubit_ratio"] = 1.0
        schema = GateBasedMetricsSchema(**valid_data)
        self.assertEqual(schema.multi_qubit_ratio, 1.0)

    def test_type_coercion(self):
        """Test type coercion in schemas."""
        # Test that integers are accepted for float fields
        valid_data = {
            "fidelity": 1,  # Integer instead of float
            "has_expected_distribution": True,
            "method": "theoretical_comparison",
            "confidence": "high",
        }

        schema = FidelityMetricsSchema(**valid_data)

        self.assertEqual(schema.fidelity, 1.0)  # Should be coerced to float
        self.assertIsInstance(schema.fidelity, float)


if __name__ == "__main__":
    unittest.main()
