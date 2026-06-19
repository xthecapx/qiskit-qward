"""Tests for qward.scan module (functional API and CLI)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from qiskit import QuantumCircuit
from qiskit.qpy import dump as qpy_dump
from qiskit_aer import AerSimulator

from qward.scan import scan_post, scan_pre
from qward.scan._ibm import trim_idle_qubits


class TestScanPre(unittest.TestCase):
    """Test scan_pre function."""

    def setUp(self):
        self.circuit = QuantumCircuit(3)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)
        self.circuit.measure_all()

    def test_returns_dict_of_dataframes(self):
        results = scan_pre(self.circuit)
        self.assertIsInstance(results, dict)
        for _key, val in results.items():
            self.assertIsInstance(val, pd.DataFrame)

    def test_contains_expected_metrics(self):
        results = scan_pre(self.circuit)
        self.assertIn("QiskitMetrics", results)
        self.assertIn("ComplexityMetrics", results)
        self.assertIn("ElementMetrics", results)
        self.assertIn("StructuralMetrics", results)
        self.assertIn("BehavioralMetrics", results)

    def test_includes_quantum_specific_small_circuit(self):
        results = scan_pre(self.circuit)
        self.assertIn("QuantumSpecificMetrics", results)

    def test_excludes_quantum_specific_when_disabled(self):
        results = scan_pre(self.circuit, include_quantum_specific=False)
        self.assertNotIn("QuantumSpecificMetrics", results)

    def test_excludes_quantum_specific_large_circuit(self):
        big = QuantumCircuit(25)
        big.h(0)
        big.measure_all()
        results = scan_pre(big, max_qubits_for_unitary=20)
        self.assertNotIn("QuantumSpecificMetrics", results)


class TestScanPost(unittest.TestCase):
    """Test scan_post function."""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def test_with_target_state(self):
        counts = {"00": 900, "11": 100}
        results = scan_post(self.circuit, counts, target_state="00")

        self.assertIn("FidelityMetrics", results)
        df = results["FidelityMetrics"]
        self.assertIn("dsr", df.columns)
        self.assertIn("hellinger_fidelity", df.columns)
        self.assertIn("success_rate", df.columns)

    def test_with_expected_outcomes_only(self):
        counts = {"000": 800, "111": 200}
        results = scan_post(self.circuit, counts, expected_outcomes=["000"])

        df = results["FidelityMetrics"]
        self.assertIn("dsr", df.columns)
        self.assertNotIn("hellinger_fidelity", df.columns)

    def test_with_target_histogram_only(self):
        counts = {"00": 500, "11": 500}
        ideal = {"00": 0.5, "11": 0.5}
        results = scan_post(self.circuit, counts, target_histogram=ideal)

        df = results["FidelityMetrics"]
        self.assertIn("hellinger_fidelity", df.columns)
        self.assertNotIn("dsr", df.columns)

    def test_values_in_range(self):
        counts = {"00": 600, "01": 200, "10": 100, "11": 100}
        results = scan_post(self.circuit, counts, target_state="00")

        df = results["FidelityMetrics"]
        row = df.iloc[0]
        self.assertGreaterEqual(row["dsr"], 0.0)
        self.assertLessEqual(row["dsr"], 1.0)
        self.assertGreaterEqual(row["success_rate"], 0.0)
        self.assertLessEqual(row["success_rate"], 1.0)
        self.assertGreaterEqual(row["hellinger_fidelity"], 0.0)
        self.assertLessEqual(row["hellinger_fidelity"], 1.0)

    def test_perfect_result(self):
        counts = {"111": 1000}
        results = scan_post(self.circuit, counts, target_state="111")
        df = results["FidelityMetrics"]
        row = df.iloc[0]

        self.assertAlmostEqual(row["dsr"], 1.0)
        self.assertAlmostEqual(row["success_rate"], 1.0)
        self.assertAlmostEqual(row["hellinger_fidelity"], 1.0)
        self.assertAlmostEqual(row["tvd"], 0.0)


class TestTrimIdleQubits(unittest.TestCase):
    """Test trim_idle_qubits utility."""

    def test_trim_removes_idle(self):
        qc = QuantumCircuit(5, 2)
        qc.h(1)
        qc.cx(1, 3)
        qc.measure(1, 0)
        qc.measure(3, 1)

        trimmed = trim_idle_qubits(qc)
        self.assertEqual(trimmed.num_qubits, 2)

    def test_no_idle_unchanged(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        trimmed = trim_idle_qubits(qc)
        self.assertEqual(trimmed.num_qubits, 2)

    def test_empty_circuit_raises(self):
        qc = QuantumCircuit(3)
        with self.assertRaises(ValueError):
            trim_idle_qubits(qc)


class TestScanCLI(unittest.TestCase):
    """Test CLI interface."""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "qward.scan", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("qward.scan", result.stdout)

    def test_pre_subcommand(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        with tempfile.NamedTemporaryFile(suffix=".qpy", delete=False) as f:
            qpy_dump(qc, f)
            qpy_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "qward.scan", "pre", "--circuit", qpy_path],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("QiskitMetrics", result.stdout)
            self.assertIn("ComplexityMetrics", result.stdout)
        finally:
            Path(qpy_path).unlink()

    def test_counts_subcommand_json_output(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        with tempfile.NamedTemporaryFile(suffix=".qpy", delete=False) as f:
            qpy_dump(qc, f)
            qpy_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            out_path = out.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qward.scan",
                    "counts",
                    "--circuit",
                    qpy_path,
                    "--counts",
                    '{"00": 900, "11": 100}',
                    "--target-state",
                    "00",
                    "-o",
                    out_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            self.assertIn("FidelityMetrics", data)
        finally:
            Path(qpy_path).unlink()
            Path(out_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
