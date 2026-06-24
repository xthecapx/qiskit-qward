"""Tests for EstimatorMetrics class.

Uses circuits from the Treasure Door and Four Hair Colours quantum games
with StatevectorEstimator (no credentials needed).
"""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qward.metrics.estimator_metrics import EstimatorMetrics
from qward.metrics.types import MetricsId, MetricsType
from qward.scan import scan_post
from qward.schemas.estimator_schema import EstimatorSchema


def build_treasure_door_circuit() -> QuantumCircuit:
    """Optimized treasure door: Bell pair on q0,q1 + H on q2."""
    qc = QuantumCircuit(3, name="treasure_door")
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    return qc


def build_hair_colours_circuit(nb_players: int) -> QuantumCircuit:
    """Parity-strategy circuit for the hair colours enigma."""
    nb_qubits = 2 * nb_players
    circuit = QuantumCircuit(nb_qubits, name=f"hair_colours_{nb_players}")

    for hair_qubit in range(nb_players):
        circuit.h(hair_qubit)

    start_qubit = 1
    for answer_qubit in range(nb_players, nb_qubits - start_qubit):
        circuit.barrier()
        for visible_hair_qubit in range(start_qubit, nb_players):
            circuit.cx(visible_hair_qubit, answer_qubit)
        circuit.barrier()
        for later_answer_qubit in range(answer_qubit + 1, nb_qubits):
            circuit.cx(answer_qubit, later_answer_qubit)
        start_qubit += 1

    return circuit


def pauli_label(num_qubits: int, operations: dict) -> str:
    """Build Qiskit Pauli label (high-to-low qubit order)."""
    return "".join(operations.get(qubit, "I") for qubit in reversed(range(num_qubits)))


class TestEstimatorMetricsInit(unittest.TestCase):
    """Test EstimatorMetrics initialization."""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.evs = np.array([0.95])

    def test_init_with_expectation_values(self):
        em = EstimatorMetrics(self.circuit, expectation_values=self.evs)
        self.assertTrue(em.is_ready())

    def test_init_with_job(self):
        estimator = StatevectorEstimator()
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        job = estimator.run([(self.circuit, [obs])])
        em = EstimatorMetrics(self.circuit, job=job)
        self.assertTrue(em.is_ready())

    def test_init_no_data(self):
        em = EstimatorMetrics(self.circuit)
        self.assertFalse(em.is_ready())

    def test_init_both_raises(self):
        estimator = StatevectorEstimator()
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        job = estimator.run([(self.circuit, [obs])])
        with self.assertRaises(ValueError):
            EstimatorMetrics(self.circuit, job=job, expectation_values=self.evs)

    def test_metric_type(self):
        em = EstimatorMetrics(self.circuit, expectation_values=self.evs)
        self.assertEqual(em.metric_type, MetricsType.POST_RUNTIME)

    def test_metric_id(self):
        em = EstimatorMetrics(self.circuit, expectation_values=self.evs)
        self.assertEqual(em.id, MetricsId.ESTIMATOR)

    def test_not_ready_returns_empty_schema(self):
        em = EstimatorMetrics(self.circuit)
        schema = em.get_metrics()
        self.assertIsInstance(schema, EstimatorSchema)
        self.assertIsNone(schema.num_observables)


class TestEstimatorMetricsFromValues(unittest.TestCase):
    """Test EstimatorMetrics with raw numpy arrays (credential-free)."""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)

    def test_single_observable_perfect(self):
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=np.array([1.0]),
            ideal_expectation_values=np.array([1.0]),
        )
        schema = em.get_metrics()
        self.assertEqual(schema.num_observables, 1)
        self.assertAlmostEqual(schema.expectation_values[0], 1.0)
        self.assertAlmostEqual(schema.success_probabilities[0], 1.0)
        self.assertAlmostEqual(schema.mean_observable_fidelity, 1.0)
        self.assertAlmostEqual(schema.mean_relative_error, 0.0)

    def test_single_observable_noisy(self):
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=np.array([0.8]),
            ideal_expectation_values=np.array([1.0]),
        )
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.expectation_values[0], 0.8)
        self.assertAlmostEqual(schema.success_probabilities[0], 0.9)
        self.assertAlmostEqual(schema.observable_fidelities[0], 0.9)
        self.assertAlmostEqual(schema.relative_errors[0], 0.2)
        self.assertAlmostEqual(schema.depolarization_factor, 0.8)

    def test_multiple_observables(self):
        evs = np.array([0.9, -0.1, 0.5])
        em = EstimatorMetrics(self.circuit, expectation_values=evs)
        schema = em.get_metrics()
        self.assertEqual(schema.num_observables, 3)
        self.assertEqual(len(schema.expectation_values), 3)
        self.assertEqual(len(schema.success_probabilities), 3)
        self.assertAlmostEqual(schema.mean_expectation_value, np.mean(evs))

    def test_success_probability_computation(self):
        evs = np.array([1.0, 0.0, -1.0])
        em = EstimatorMetrics(self.circuit, expectation_values=evs)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.success_probabilities[0], 1.0)
        self.assertAlmostEqual(schema.success_probabilities[1], 0.5)
        self.assertAlmostEqual(schema.success_probabilities[2], 0.0)

    def test_observable_fidelity_with_ideal(self):
        evs = np.array([0.9, -0.1])
        ideal = np.array([1.0, 0.0])
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=evs,
            ideal_expectation_values=ideal,
        )
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.observable_fidelities[0], 0.95)
        self.assertAlmostEqual(schema.observable_fidelities[1], 0.95)

    def test_relative_error_calculation(self):
        evs = np.array([0.8, 0.5])
        ideal = np.array([1.0, 1.0])
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=evs,
            ideal_expectation_values=ideal,
        )
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.relative_errors[0], 0.2)
        self.assertAlmostEqual(schema.relative_errors[1], 0.5)

    def test_snr_calculation(self):
        evs = np.array([0.9, 0.5])
        stds = np.array([0.1, 0.25])
        em = EstimatorMetrics(self.circuit, expectation_values=evs, standard_deviations=stds)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.signal_to_noise_ratios[0], 9.0)
        self.assertAlmostEqual(schema.signal_to_noise_ratios[1], 2.0)
        self.assertAlmostEqual(schema.mean_snr, 5.5)

    def test_depolarization_factor(self):
        ideal = np.array([1.0, -1.0, 0.5])
        evs = np.array([0.7, -0.7, 0.35])
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=evs,
            ideal_expectation_values=ideal,
        )
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.depolarization_factor, 0.7)

    def test_without_ideal_values(self):
        em = EstimatorMetrics(self.circuit, expectation_values=np.array([0.9]))
        schema = em.get_metrics()
        self.assertIsNone(schema.observable_fidelities)
        self.assertIsNone(schema.relative_errors)
        self.assertIsNone(schema.depolarization_factor)

    def test_without_stds(self):
        em = EstimatorMetrics(self.circuit, expectation_values=np.array([0.9]))
        schema = em.get_metrics()
        self.assertIsNone(schema.signal_to_noise_ratios)
        self.assertIsNone(schema.mean_snr)

    def test_to_flat_dict(self):
        evs = np.array([0.9, 0.5])
        stds = np.array([0.1, 0.2])
        em = EstimatorMetrics(self.circuit, expectation_values=evs, standard_deviations=stds)
        schema = em.get_metrics()
        flat = schema.to_flat_dict()
        self.assertIn("evs_0", flat)
        self.assertIn("evs_1", flat)
        self.assertIn("stds_0", flat)
        self.assertIn("stds_1", flat)
        self.assertIn("success_prob_0", flat)
        self.assertIn("mean_success_probability", flat)
        self.assertNotIn("observable_fidelities", flat)

    def test_observable_labels(self):
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=np.array([1.0, 0.0]),
            observable_labels=["ZZ", "ZI"],
        )
        schema = em.get_metrics()
        self.assertEqual(schema.observable_labels, ["ZZ", "ZI"])


class TestEstimatorMetricsFromJob(unittest.TestCase):
    """Test EstimatorMetrics with StatevectorEstimator jobs."""

    def setUp(self):
        self.estimator = StatevectorEstimator()

    def test_treasure_door_izz(self):
        """IZZ observable on treasure door: perfect q0==q1 correlation."""
        qc = build_treasure_door_circuit()
        obs = SparsePauliOp.from_list([("IZZ", 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()

        self.assertAlmostEqual(schema.expectation_values[0], 1.0, places=10)
        self.assertAlmostEqual(schema.success_probabilities[0], 1.0, places=10)

    def test_treasure_door_multi_observable(self):
        """Multiple observables on treasure door circuit."""
        qc = build_treasure_door_circuit()
        observables = [
            SparsePauliOp.from_list([("IZZ", 1.0)]),
            SparsePauliOp.from_list([("XII", 1.0)]),
            SparsePauliOp.from_list([("IIZ", 1.0)]),
            SparsePauliOp.from_list([("IZI", 1.0)]),
            SparsePauliOp.from_list([("ZII", 1.0)]),
        ]
        job = self.estimator.run([(qc, observables)])

        em = EstimatorMetrics(
            qc,
            job=job,
            ideal_expectation_values=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
            observable_labels=["IZZ", "XII", "IIZ", "IZI", "ZII"],
        )
        schema = em.get_metrics()

        self.assertEqual(schema.num_observables, 5)
        self.assertAlmostEqual(schema.expectation_values[0], 1.0, places=10)
        self.assertAlmostEqual(schema.expectation_values[1], 1.0, places=10)
        self.assertAlmostEqual(schema.expectation_values[2], 0.0, places=10)
        self.assertAlmostEqual(schema.expectation_values[3], 0.0, places=10)
        self.assertAlmostEqual(schema.expectation_values[4], 0.0, places=10)
        self.assertAlmostEqual(schema.mean_observable_fidelity, 1.0, places=10)

    def test_hair_colours_4_players_per_player(self):
        """Per-player observables: players 1-3 always correct, player 0 = 50%."""
        nb_players = 4
        qc = build_hair_colours_circuit(nb_players)
        num_qubits = 2 * nb_players

        observables = []
        for player in range(nb_players):
            label = pauli_label(num_qubits, {player: "Z", nb_players + player: "Z"})
            observables.append(SparsePauliOp.from_list([(label, 1.0)]))

        job = self.estimator.run([(qc, observables)])
        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()

        self.assertEqual(schema.num_observables, 4)
        # Player 0: random (expectation 0.0, success prob 0.5)
        self.assertAlmostEqual(schema.expectation_values[0], 0.0, places=10)
        self.assertAlmostEqual(schema.success_probabilities[0], 0.5, places=10)
        # Players 1-3: always correct (expectation 1.0)
        for i in range(1, 4):
            self.assertAlmostEqual(schema.expectation_values[i], 1.0, places=10)
            self.assertAlmostEqual(schema.success_probabilities[i], 1.0, places=10)

    def test_hair_colours_4_players_all_correct(self):
        """Global parity observable: all correct = 50%."""
        nb_players = 4
        qc = build_hair_colours_circuit(nb_players)
        num_qubits = 2 * nb_players

        label = pauli_label(num_qubits, {q: "Z" for q in range(nb_players)})
        obs = SparsePauliOp.from_list([(label, 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()

        self.assertAlmostEqual(schema.expectation_values[0], 0.0, places=10)
        self.assertAlmostEqual(schema.success_probabilities[0], 0.5, places=10)

    def test_hair_colours_6_players(self):
        """6-player version: same pattern scales."""
        nb_players = 6
        qc = build_hair_colours_circuit(nb_players)
        num_qubits = 2 * nb_players

        observables = []
        for player in range(nb_players):
            label = pauli_label(num_qubits, {player: "Z", nb_players + player: "Z"})
            observables.append(SparsePauliOp.from_list([(label, 1.0)]))

        job = self.estimator.run([(qc, observables)])
        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()

        # Player 0: random
        self.assertAlmostEqual(schema.expectation_values[0], 0.0, places=10)
        # Players 1-5: always correct
        for i in range(1, 6):
            self.assertAlmostEqual(schema.expectation_values[i], 1.0, places=10)

    def test_bell_state_zz(self):
        """Bell state with ZZ: perfect correlation."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.expectation_values[0], 1.0, places=10)

    def test_bell_state_zi(self):
        """Bell state with ZI: individual qubit is maximally mixed."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = SparsePauliOp.from_list([("ZI", 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.expectation_values[0], 0.0, places=10)

    def test_single_qubit_z(self):
        """|0> state with Z: expectation = +1."""
        qc = QuantumCircuit(1)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.expectation_values[0], 1.0, places=10)

    def test_single_qubit_x_state_z(self):
        """|+> state with Z: expectation = 0."""
        qc = QuantumCircuit(1)
        qc.h(0)
        obs = SparsePauliOp.from_list([("Z", 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.expectation_values[0], 0.0, places=10)

    def test_with_ideal_values_from_job(self):
        """Ideal StatevectorEstimator matches ideal perfectly."""
        qc = build_treasure_door_circuit()
        obs = SparsePauliOp.from_list([("IZZ", 1.0)])
        job = self.estimator.run([(qc, [obs])])

        em = EstimatorMetrics(qc, job=job, ideal_expectation_values=np.array([1.0]))
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.mean_observable_fidelity, 1.0)
        self.assertAlmostEqual(schema.mean_relative_error, 0.0)


class TestEstimatorMetricsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.cx(0, 1)

    def test_zero_expectation_value(self):
        """evs=0 means maximally mixed (success prob 0.5)."""
        em = EstimatorMetrics(self.circuit, expectation_values=np.array([0.0]))
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.success_probabilities[0], 0.5)
        self.assertAlmostEqual(schema.mean_expectation_value, 0.0)

    def test_negative_expectation_value(self):
        """evs=-1 means anti-correlated (success prob 0)."""
        em = EstimatorMetrics(self.circuit, expectation_values=np.array([-1.0]))
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.success_probabilities[0], 0.0)

    def test_scalar_evs_handling(self):
        """Single float converted to 1-element array."""
        em = EstimatorMetrics(self.circuit, expectation_values=np.float64(0.75))
        schema = em.get_metrics()
        self.assertEqual(schema.num_observables, 1)
        self.assertAlmostEqual(schema.expectation_values[0], 0.75)

    def test_depolarization_with_zero_ideal(self):
        """Depolarization skips observables with ideal=0."""
        evs = np.array([0.7, 0.0])
        ideal = np.array([1.0, 0.0])
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=evs,
            ideal_expectation_values=ideal,
        )
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.depolarization_factor, 0.7)

    def test_snr_with_zero_std(self):
        """SNR is inf when std=0 (StatevectorEstimator case)."""
        evs = np.array([0.9])
        stds = np.array([0.0])
        em = EstimatorMetrics(self.circuit, expectation_values=evs, standard_deviations=stds)
        schema = em.get_metrics()
        self.assertAlmostEqual(schema.signal_to_noise_ratios[0], 0.0)

    def test_all_zero_ideal_no_depolarization(self):
        """All ideal values zero: depolarization is None."""
        evs = np.array([0.1, -0.1])
        ideal = np.array([0.0, 0.0])
        em = EstimatorMetrics(
            self.circuit,
            expectation_values=evs,
            ideal_expectation_values=ideal,
        )
        schema = em.get_metrics()
        self.assertIsNone(schema.depolarization_factor)


class TestScanPost(unittest.TestCase):
    """Test scan_post with both primitive types."""

    def test_scan_post_estimator_basic(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        result = scan_post(
            circuit,
            expectation_values=np.array([1.0, 0.0]),
            ideal_expectation_values=np.array([1.0, 0.0]),
            observable_labels=["ZZ", "ZI"],
        )

        self.assertIn("FidelityMetrics", result)
        df = result["FidelityMetrics"]
        self.assertIn("evs_0", df.columns)
        self.assertIn("evs_1", df.columns)
        self.assertIn("mean_observable_fidelity", df.columns)

    def test_scan_post_estimator_with_stds(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        result = scan_post(
            circuit,
            expectation_values=np.array([0.9]),
            standard_deviations=np.array([0.05]),
        )

        df = result["FidelityMetrics"]
        self.assertIn("snr_0", df.columns)
        self.assertIn("mean_snr", df.columns)

    def test_scan_post_with_counts(self):
        """scan_post Sampler path (backward compat)."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()

        result = scan_post(
            circuit,
            {"00": 500, "11": 500},
            target_state="00",
        )

        self.assertIn("FidelityMetrics", result)
        df = result["FidelityMetrics"]
        self.assertIn("dsr", df.columns)

    def test_scanner_integration_with_estimator_strategy(self):
        """EstimatorMetrics works within Scanner pipeline directly."""
        from qward.scanner import Scanner

        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.h(2)

        em = EstimatorMetrics(
            circuit,
            expectation_values=np.array([1.0, 1.0, 0.0]),
            observable_labels=["IZZ", "XII", "IIZ"],
        )
        scanner = Scanner(circuit=circuit)
        scanner.add_strategy(em)
        result = scanner.calculate_metrics()

        self.assertIn("EstimatorMetrics", result)
        df = result["EstimatorMetrics"]
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["evs_0"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
