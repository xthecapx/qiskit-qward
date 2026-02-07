"""Integration tests for matrix product verification algorithms."""

import unittest

from qward.algorithms import matrix_product_verification_tests as mpv_tests


class TestMatrixProductVerificationAlgorithm(unittest.TestCase):
    """Wire matrix product verification tests into pytest/unittest."""

    def test_classical_reference_cases(self):
        results = mpv_tests.run_all_tests(verbose=False)
        self.assertEqual(results["failed"], 0)

    def test_base_class_structure(self):
        self.assertTrue(mpv_tests.test_base_class_structure())

    def test_non_square_matrices(self):
        self.assertTrue(mpv_tests.test_non_square_matrices())

    def test_quantum_freivalds_circuit(self):
        self.assertTrue(mpv_tests.test_quantum_freivalds_circuit())

    def test_quantum_freivalds_simulation(self):
        self.assertTrue(mpv_tests.test_quantum_freivalds_simulation())

    def test_quantum_vs_classical_comparison(self):
        self.assertTrue(mpv_tests.test_quantum_vs_classical_comparison())

    def test_bit_ordering_reference(self):
        self.assertTrue(mpv_tests.test_bit_ordering_reference())

    def test_bit_order_error_state_mapping(self):
        self.assertTrue(mpv_tests.test_bit_order_error_state_mapping())
