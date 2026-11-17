"""Tests for qward.qaqs: frozen-model inference, footprint parity and prediction warnings.

The fixtures in ``tests/data/qaqs`` come from the thesis dataset: the original scikit-learn /
XGBoost predictions on stored feature rows, and 20 real validation circuits (logical and
compiled) with their stored footprints, pre-registered predictions and billed cost.
"""

import base64
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeKingston

from qward.qaqs import _inference, available_models, model_card, predict, predict_cost
from qward.qaqs.footprint import (
    compiled_footprint,
    ensure_barrier,
    logical_footprint,
    quantum_specific,
)
from qward.qaqs.models import DATA_DIR, load_model

DATA = Path(__file__).resolve().parent / "data" / "qaqs"
GOLDEN = json.loads((DATA / "golden_models.json").read_text(encoding="utf-8"))["models"]
FIXTURES = json.loads((DATA / "validation_fixtures.json").read_text(encoding="utf-8"))["fixtures"]

QUALITY_MODELS = [m for m in GOLDEN if not m.startswith("cost")]


def _circuit(blob: str) -> QuantumCircuit:
    return qpy.load(io.BytesIO(base64.b64decode(blob)))[0]


def _matrix(rows):
    return np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)


class TestInferenceParity(unittest.TestCase):
    """NumPy inference must reproduce the original scikit-learn / XGBoost predictions."""

    def test_every_quality_model_matches_the_original(self):
        for model_id in QUALITY_MODELS:
            with self.subTest(model=model_id):
                gold = GOLDEN[model_id]
                model = load_model(model_id)
                self.assertEqual(model["features"], gold["features"])
                got = _inference.predict_quality(model, _matrix(gold["rows"]))
                np.testing.assert_allclose(got, gold["raw"], atol=1e-5, rtol=0)

    def test_cost_model_matches_the_original(self):
        gold = GOLDEN["cost/billed_seconds"]
        model = load_model("cost/billed_seconds")
        matrix = _matrix(gold["rows"])
        columns = {name: matrix[:, i] for i, name in enumerate(gold["features"])}
        got = _inference.predict_cost(model, columns)
        np.testing.assert_allclose(got["billed_seconds"], gold["raw"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(got["physical"], gold["physical"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            got["circuit_seconds"], gold["circuitSeconds"], atol=1e-9, rtol=0
        )

    def test_missing_values_use_training_medians(self):
        model = load_model("ibm_kingston/success")
        row = np.full((1, len(model["features"])), np.nan)
        filled = np.array([model["preprocessing"]["imputer"]])
        np.testing.assert_allclose(
            _inference.predict_quality(model, row), _inference.predict_quality(model, filled)
        )


class TestFootprintParity(unittest.TestCase):
    """The footprint code reproduces the values stored with 20 real validation circuits."""

    def test_footprint_matches_stored_training_columns(self):
        for fixture in FIXTURES:
            with self.subTest(chip=fixture["chip"], config=fixture["config"]):
                mine = {
                    **logical_footprint(_circuit(fixture["logicalQpy"])),
                    **compiled_footprint(_circuit(fixture["isaQpy"])),
                }
                for column, stored in fixture["storedFeatures"].items():
                    if stored is None or column.startswith("a.quantum_specific"):
                        continue
                    self.assertIn(column, mine)
                    self.assertAlmostEqual(float(mine[column]), float(stored), places=9, msg=column)

    def test_quantum_specific_metrics_match_training_values(self):
        try:
            import torch  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError:
            self.skipTest("PyTorch is not installed")
        checked = 0
        for fixture in FIXTURES:
            circuit = _circuit(fixture["logicalQpy"])
            stored = {
                k: v
                for k, v in fixture["storedFeatures"].items()
                if k.startswith("a.quantum_specific") and v is not None
            }
            if circuit.num_qubits > 4 or not stored:
                continue
            with self.subTest(config=fixture["config"]):
                mine = quantum_specific(circuit)
                self.assertEqual(mine["a.quantum_specific_present"], 1.0)
                for column, value in stored.items():
                    # The metric is computed in float32 (PyTorch complex64), whose spacing at
                    # values near 7 is 4.8e-7, and different CPU builds of PyTorch differ by a
                    # few float32 steps. Compare to float32 precision, not to 6 decimals.
                    self.assertAlmostEqual(mine[column], value, delta=max(1e-5, 1e-5 * abs(value)))
                checked += 1
        self.assertGreater(checked, 0)

    def test_quantum_specific_is_absent_for_wide_or_dynamic_circuits(self):
        wide = QuantumCircuit(9, 9)
        wide.h(range(9))
        wide.measure(range(9), range(9))
        self.assertEqual(quantum_specific(wide), {"a.quantum_specific_present": 0.0})


class TestPredictionsFromStoredFootprints(unittest.TestCase):
    """Footprint -> prediction equals the pre-registered prediction, including cost."""

    def test_quality_equals_preregistered_predictions(self):
        for fixture in FIXTURES:
            footprint = dict(fixture["storedFeatures"])
            footprint["a.quantum_specific_present"] = (
                1.0 if footprint.get("a.quantum_specific.magic") is not None else 0.0
            )
            for target, expected in fixture["expectedQuality"].items():
                with self.subTest(chip=fixture["chip"], config=fixture["config"], target=target):
                    model = load_model(f"{fixture['chip']}/{target}")
                    from qward.qaqs.footprint import (
                        as_row,
                    )  # pylint: disable=import-outside-toplevel

                    value = _inference.predict_quality(model, as_row(footprint, model["features"]))[
                        0
                    ]
                    self.assertAlmostEqual(float(np.clip(value, 0, 1)), expected, places=5)

    def test_cost_equals_frozen_model_on_stored_inputs(self):
        checked = 0
        for fixture in FIXTURES:
            if fixture["expectedCostSeconds"] is None:
                continue
            footprint = compiled_footprint(_circuit(fixture["isaQpy"]))
            footprint["duration_s"] = fixture["storedDuration"]
            footprint["dynamic"] = fixture["dynamic"]
            result = predict_cost(
                footprint,
                twirling=fixture["twirling"],
                dynamical_decoupling=fixture["dynamicalDecoupling"],
            )
            with self.subTest(chip=fixture["chip"], config=fixture["config"]):
                self.assertAlmostEqual(result.seconds, fixture["expectedCostSeconds"], places=6)
                self.assertEqual(result.billed_seconds, int(np.ceil(result.seconds - 1e-9)))
                self.assertAlmostEqual(result.resource_units, result.seconds / 60.0, places=12)
            checked += 1
        self.assertGreater(checked, 10)


class TestPredictApi(unittest.TestCase):
    """The public ``predict`` function on a small circuit compiled for a fake chip."""

    @classmethod
    def setUpClass(cls):
        cls.circuit = QuantumCircuit(3, 3)
        cls.circuit.h(0)
        cls.circuit.cx(0, 1)
        cls.circuit.cx(1, 2)
        cls.circuit.measure(range(3), range(3))
        cls.result = predict(cls.circuit, FakeFez(), quantum_specific_metrics=False)

    def test_returns_footprint_quality_and_cost(self):
        r = self.result
        self.assertEqual(len(r.quality), 8)  # 2 chips x 4 targets
        self.assertIn("b.two_qubit_gates", r.footprint)
        self.assertEqual(r.footprint["b.two_qubit_gates"], 2)
        for q in r.quality.values():
            self.assertTrue(0.0 <= q.value <= 1.0)
        self.assertIsNotNone(r.cost)
        self.assertTrue(1.9 < r.cost.seconds < 2.1)
        self.assertEqual(r.cost.billed_seconds, 2)

    def test_each_prediction_carries_its_validation_score(self):
        success = self.result.quality["ibm_fez/success"]
        self.assertIsNotNone(success.validation_r2)
        self.assertGreater(success.validation_r2, 0.5)
        self.assertEqual(success.verdict, "pass")
        self.assertEqual(len(self.result.quality_table()), 8)

    def test_dsr_is_flagged_as_unreliable(self):
        dsr = self.result.quality["ibm_kingston/dsr"]
        self.assertTrue(any("DSR" in w for w in dsr.warnings))
        self.assertFalse(dsr.reliable)

    def test_fake_backend_and_missing_quantum_specific_are_reported(self):
        self.assertTrue(any("fake backend" in w for w in self.result.warnings))
        fez_dsr = self.result.quality["ibm_fez/dsr"]
        self.assertTrue(any("quantum-specific" in w for w in fez_dsr.warnings))

    def test_cost_warns_outside_validated_shots(self):
        base = predict_cost(self.result.footprint, shots=1024)
        more = predict_cost(self.result.footprint, shots=4096)
        self.assertEqual(base.warnings, ())
        self.assertTrue(any("not validated" in w for w in more.warnings))
        self.assertGreater(more.seconds, base.seconds)
        self.assertAlmostEqual(
            more.seconds - base.seconds,
            3 * (base.seconds - 1.702),
            delta=0.05,
        )

    def test_chip_and_target_selection_and_validation(self):
        r = predict(
            self.circuit, FakeKingston(), chips=("ibm_kingston",), targets=("success",), cost=False
        )
        self.assertEqual(list(r.quality), ["ibm_kingston/success"])
        self.assertIsNone(r.cost)
        with self.assertRaises(ValueError):
            predict(self.circuit, FakeFez(), chips=("ibm_marrakesh",))

    def test_large_circuits_are_warned_about(self):
        big = QuantumCircuit(20, 20)
        big.h(0)
        for i in range(19):
            big.cx(i, i + 1)
        for _ in range(60):
            for i in range(19):
                big.cx(i, i + 1)
        big.measure(range(20), range(20))
        r = predict(big, FakeFez(), chips=("ibm_fez",), targets=("success",), cost=False)
        self.assertTrue(
            any(
                "more than the largest in training" in w
                for w in r.quality["ibm_fez/success"].warnings
            )
        )


class TestBarrierHandling(unittest.TestCase):
    """Training circuits always had a barrier before measuring; the footprint is sensitive to it."""

    @staticmethod
    def _ghz(barrier: bool) -> QuantumCircuit:
        qc = QuantumCircuit(5, 5)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)
        if barrier:
            qc.barrier()
        qc.measure(range(5), range(5))
        return qc

    def test_barrier_is_added_before_final_measurements_when_missing(self):
        fixed, added = ensure_barrier(self._ghz(False))
        self.assertTrue(added)
        names = [i.operation.name for i in fixed.data]
        self.assertEqual(names[-6], "barrier")
        self.assertEqual(names[-5:], ["measure"] * 5)
        self.assertEqual(fixed.num_qubits, 5)

    def test_existing_barriers_and_gate_free_circuits_are_left_alone(self):
        qc = self._ghz(True)
        same, added = ensure_barrier(qc)
        self.assertFalse(added)
        self.assertIs(same, qc)
        only_measure = QuantumCircuit(2, 2)
        only_measure.measure(range(2), range(2))
        self.assertEqual(ensure_barrier(only_measure)[1], True)
        no_measure = QuantumCircuit(2)
        no_measure.h(0)
        self.assertEqual(ensure_barrier(no_measure)[1], False)

    def test_predict_matches_the_barrier_version_and_says_so(self):
        backend = FakeKingston()
        kw = {
            "chips": ("ibm_kingston",),
            "targets": ("success",),
            "cost": False,
            "quantum_specific_metrics": False,
        }
        plain = predict(self._ghz(False), backend, **kw)
        barred = predict(self._ghz(True), backend, **kw)
        self.assertAlmostEqual(
            plain.quality["ibm_kingston/success"].value,
            barred.quality["ibm_kingston/success"].value,
            places=9,
        )
        self.assertTrue(any("no barrier found" in w for w in plain.warnings))
        self.assertFalse(any("no barrier found" in w for w in barred.warnings))
        as_given = predict(self._ghz(False), backend, add_barrier=False, **kw)
        self.assertFalse(any("no barrier found" in w for w in as_given.warnings))
        self.assertTrue(
            any("outside the range" in w for w in as_given.quality["ibm_kingston/success"].warnings)
        )

    def test_circuit_that_compiles_to_nothing_is_refused_with_an_explanation(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(0)
        qc.measure(range(2), range(2))
        with self.assertRaisesRegex(ValueError, "no gates"):
            predict(qc, FakeFez(), cost=False)


class TestPackaging(unittest.TestCase):
    """Models are plain JSON, load without heavy libraries, and are all present."""

    def test_models_are_bundled_as_json_only(self):
        files = sorted(p.name for p in DATA_DIR.iterdir())
        self.assertTrue(files and all(name.endswith(".json") for name in files))
        self.assertEqual(len(available_models()), 9)

    def test_model_cards_describe_validation(self):
        card = model_card("ibm_kingston/success")
        self.assertEqual(card.chip, "ibm_kingston")
        self.assertGreater(card.trained_on_jobs, 1000)
        self.assertIn("b.two_qubit_gates", card.training_range)
        self.assertIsNotNone(card.unseen_chip_new_circuits)

    def test_import_does_not_load_training_libraries(self):
        """Inference needs NumPy only: no scikit-learn, XGBoost, LightGBM or joblib (pickle)."""
        code = (
            "import sys, qward.qaqs;"
            "print([m for m in ('sklearn', 'xgboost', 'lightgbm', 'joblib') if m in sys.modules])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
