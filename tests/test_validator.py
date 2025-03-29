"""Tests for qiskit_qward validators."""

from unittest import TestCase

from qiskit_qward.validators.base_validator import BaseValidator


class TestBaseValidator(TestCase):
    """Tests base validator."""

    def test_base_validator_init(self):
        """Tests base validator initialization."""
        validator = BaseValidator(
            num_qubits=2, num_clbits=2, use_barriers=True, name="test_validator"
        )

        self.assertEqual(validator.num_qubits, 2)
        self.assertEqual(validator.num_clbits, 2)
        self.assertTrue(validator.use_barriers)
        self.assertEqual(validator.name, "test_validator")
