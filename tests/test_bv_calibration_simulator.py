"""Local simulator validation; no IBM connection or QPU execution."""

import json
from unittest.mock import Mock

import pytest
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import QuantumError
from qiskit.quantum_info import average_gate_fidelity

from qward.examples.papers.bv import bv_calibration_ibm
from qward.examples.papers.bv.bv_calibration_configs import CONFIGS, build_circuit
from qward.examples.papers.bv.bv_calibration_simulator import (
    BVCalibrationSimulatorExperiment,
    PROFILES,
    simulator_method,
    synthetic_backend,
)
from qward.metrics.circuit_error_prediction import record_hash


def _config(family="bv_padded", angle=0):
    return next(
        c
        for c in CONFIGS.values()
        if c.num_qubits == 2
        and c.layout_label == "low_predicted_error"
        and c.family == family
        and c.forward_layers == 0
        and c.angle == angle
        and c.pattern == "ALT"
    )


def test_synthetic_calibration_matches_requested_average_errors():
    backend = synthetic_backend(PROFILES["NOMINAL"])
    assert backend.target["cx"][(0, 1)].error == 0.01
    assert backend.target["cx"][(5, 6)].error == 0.02
    assert backend.target["measure"][(0,)].error == 0.02
    assert backend.target["rz"][(0,)].error == 0
    noise = NoiseModel.from_backend(backend, thermal_relaxation=False)
    assert not noise.is_ideal()
    for gate, operands, infidelity in (("cx", [0, 1], 0.01), ("sx", [0], 0.001)):
        error = next(
            e
            for e in noise.to_dict()["errors"]
            if e["type"] == "qerror"
            and e["operations"] == [gate]
            and [tuple(q) for q in e["gate_qubits"]] == [tuple(operands)]
        )
        # Constant-size gate channels, not a circuit-state simulation.
        actual = 1 - average_gate_fidelity(QuantumError.from_dict(error).to_quantumchannel())
        assert actual == pytest.approx(infidelity)


@pytest.mark.parametrize("profile", ["IDEAL", "NOMINAL"])
def test_local_run_uses_frozen_prediction_and_never_connects_to_ibm(tmp_path, monkeypatch, profile):
    import qiskit_ibm_runtime

    network = Mock(side_effect=AssertionError("No IBM service in simulator execution"))
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", network)
    monkeypatch.setattr(bv_calibration_ibm, "calculate_qward_metrics", lambda circuit: {})
    experiment = BVCalibrationSimulatorExperiment(shots=512, output_dir=tmp_path)
    result = experiment.run(_config().config_id, profile=profile, num_runs=2, seed=5)
    saved = json.loads(open(result["record_path"], encoding="utf-8").read())
    assert saved["status"] == "completed"
    assert saved["execution_type"] == "NOISY_SIMULATOR"
    assert saved["simulator_method"] == "stabilizer"
    archived = json.loads(open(saved["simulation"]["noise_model_path"], encoding="utf-8").read())
    assert record_hash(archived) == saved["simulation"]["noise_model_hash"]
    assert len(saved["individual_results"]) == 2
    assert all(
        r["shots"] == 512 and r["backend_type"] == "simulator" for r in saved["individual_results"]
    )
    if profile == "IDEAL":
        assert all(
            r["histogram_assessment"]["success_rate"] == 1 for r in saved["individual_results"]
        )
    else:
        assert all(r["prediction"]["esp_score"] < 1 for r in saved["individual_results"])
    network.assert_not_called()


def test_non_clifford_control_selects_mps_and_executes_locally(tmp_path, monkeypatch):
    import math

    monkeypatch.setattr(bv_calibration_ibm, "calculate_qward_metrics", lambda circuit: {})
    config = _config("bv_target_deviation", math.pi / 6)
    backend = synthetic_backend(PROFILES["IDEAL"])
    isa = generate_preset_pass_manager(backend=backend, optimization_level=0).run(
        build_circuit(config)
    )
    assert simulator_method(isa) == "matrix_product_state"
    experiment = BVCalibrationSimulatorExperiment(shots=4096, output_dir=tmp_path)
    result = experiment.run(config.config_id, profile="IDEAL", num_runs=1, seed=7)
    saved = json.loads(open(result["record_path"], encoding="utf-8").read())
    observed = saved["individual_results"][0]["histogram_assessment"]["success_rate"]
    assert observed == pytest.approx(math.cos(math.pi / 12) ** 2, abs=0.02)
    assert result["method"] == "matrix_product_state"


def test_wrong_archived_noise_model_is_rejected_before_simulation(tmp_path):
    path = tmp_path / "wrong_model.json"
    path.write_text(json.dumps({"errors": []}))
    experiment = BVCalibrationSimulatorExperiment(shots=10, output_dir=tmp_path / "raw")
    with pytest.raises(ValueError, match="Archived noise model differs"):
        experiment.run(_config().config_id, profile="NOMINAL", noise_model_path=path)
    assert not (tmp_path / "raw").exists()


def test_scaling_scope_adds_wider_primary_bv_without_changing_ibm_matrix():
    from qward.examples.papers.bv.bv_calibration_configs import SCALING_CONFIGS
    from qward.examples.papers.bv.bv_calibration_simulator import select_configs

    assert len(CONFIGS) == 100
    assert len(select_configs("smoke")) == 25
    scaling = select_configs("scaling")
    assert len(scaling) == 48 + len(SCALING_CONFIGS) == 96
    assert {c.num_qubits for c in SCALING_CONFIGS.values()} == {6, 8}
    backend = synthetic_backend(PROFILES["NOMINAL"], line_length=9)
    assert backend.num_qubits == 18
    assert backend.target["cx"][(9, 10)].error == 0.02
    assert backend.target["cx"][(7, 8)].error == 0.01
