"""Tests for the BV-derived signal-plus-background proof experiment."""

import math

import pytest

from qward.examples.papers.bv import bv_signal_background_ibm as campaign_module
from qward.examples.papers.bv.bv_signal_background import (
    CAMPAIGN_TOTAL_QUBITS,
    PROOF_TOTAL_QUBITS,
    build_signal_background_circuit,
    distribution_fidelities,
    dsr_profile,
    heavy_hex_transpiled_complexity,
    ideal_distribution,
    make_spec,
    noisy_counts,
    probabilities_to_counts,
)
from qward.examples.papers.bv.bv_signal_background_configs import get_config
from qward.examples.papers.bv.bv_signal_background_ibm import (
    BVSignalBackgroundIBMExperiment,
)


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


@pytest.mark.parametrize("total_qubits", PROOF_TOTAL_QUBITS)
def test_targets_are_analytic_and_distinct(total_qubits):
    spec = make_spec(total_qubits)
    first, second = spec.targets

    assert len(first) == len(second) == spec.num_measured_qubits
    assert first[:2] == "00"
    assert second[:2] == "01"
    assert all(first[i] != first[i + 1] for i in range(2, len(first) - 1))
    assert _hamming(first[2:], second[2:]) == 2


@pytest.mark.parametrize("total_qubits", PROOF_TOTAL_QUBITS)
def test_circuit_shape_and_complexity(total_qubits):
    spec = make_spec(total_qubits)
    circuit = build_signal_background_circuit(spec)

    assert circuit.num_qubits == total_qubits
    assert circuit.num_clbits == spec.num_measured_qubits
    assert circuit.depth() <= 8 * total_qubits
    assert {"ry", "cry", "crz"} & set(circuit.count_ops())


@pytest.mark.parametrize("total_qubits", PROOF_TOTAL_QUBITS)
def test_exact_distribution_has_dominant_targets_and_broad_background(total_qubits):
    spec = make_spec(total_qubits)
    probabilities = ideal_distribution(spec)
    target_probability = sum(probabilities.get(target, 0.0) for target in spec.targets)
    top_two = {
        outcome
        for outcome, _probability in sorted(
            probabilities.items(), key=lambda item: item[1], reverse=True
        )[:2]
    }
    background_support = set(probabilities) - set(spec.targets)

    assert math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(target_probability, spec.target_mass, abs_tol=1e-9)
    assert top_two == set(spec.targets)
    assert len(background_support) > 2


@pytest.mark.parametrize("total_qubits", PROOF_TOTAL_QUBITS)
def test_dsr_uses_targets_without_ideal_distribution(total_qubits):
    spec = make_spec(total_qubits)
    first, second = spec.targets
    background = "1" + "0" * (spec.num_measured_qubits - 1)
    counts = {first: 400, second: 400, background: 200}
    profile = dsr_profile(counts, spec)

    assert profile["success_rate"] == pytest.approx(0.8)
    assert profile["chance_baseline"] == pytest.approx(
        len(spec.targets) / (2**spec.num_measured_qubits),
        abs=1e-6,
    )
    assert 0.0 <= profile["dsr_michelson"] <= 1.0


@pytest.mark.parametrize("total_qubits", PROOF_TOTAL_QUBITS)
def test_full_fidelities_do_not_collapse_to_success_rate(total_qubits):
    spec = make_spec(total_qubits)
    probabilities = ideal_distribution(spec)
    counts = probabilities_to_counts(probabilities, shots=16_384)
    profile = dsr_profile(counts, spec)
    fidelities = distribution_fidelities(counts, probabilities)

    assert fidelities["hellinger_fidelity"] > 0.99
    assert fidelities["tvd_fidelity"] > 0.99
    assert abs(fidelities["hellinger_fidelity"] - profile["success_rate"]) > 0.05
    assert abs(fidelities["tvd_fidelity"] - profile["success_rate"]) > 0.05


def test_equal_success_can_have_different_full_fidelity():
    spec = make_spec(6)
    first, second = spec.targets
    background_a = "10" + "0" * spec.num_data_qubits
    background_b = "11" + "0" * spec.num_data_qubits
    reference = {first: 0.4, second: 0.4, background_a: 0.2}
    observed_a = {first: 400, second: 400, background_a: 200}
    observed_b = {first: 400, second: 400, background_b: 200}

    assert (
        dsr_profile(observed_a, spec)["success_rate"]
        == dsr_profile(observed_b, spec)["success_rate"]
    )
    fidelity_a = distribution_fidelities(observed_a, reference)
    fidelity_b = distribution_fidelities(observed_b, reference)
    assert fidelity_a["hellinger_fidelity"] != fidelity_b["hellinger_fidelity"]
    assert fidelity_a["tvd_fidelity"] != fidelity_b["tvd_fidelity"]


def test_noisy_proof_and_routing_complexity_are_available():
    spec = make_spec(6)
    counts = noisy_counts(spec, shots=2048)
    complexity = heavy_hex_transpiled_complexity(spec)

    assert sum(counts.values()) == 2048
    assert complexity["depth"] > build_signal_background_circuit(spec).depth()
    assert complexity["two_qubit_gates"] > 0
    assert complexity["three_qubit_gates"] == 0


@pytest.mark.parametrize("total_qubits", CAMPAIGN_TOTAL_QUBITS)
def test_ibm_campaign_config_has_two_known_targets(total_qubits):
    config = get_config(f"BVSB{total_qubits}")
    circuit = BVSignalBackgroundIBMExperiment().create_circuit(config)

    assert config.target_mass == pytest.approx(0.9)
    assert len(config.expected_outcomes) == 2
    assert circuit.num_qubits == total_qubits
    assert circuit.num_clbits == total_qubits - 1
    assert "measure" in circuit.count_ops()
    assert "if_else" in circuit.count_ops()
    assert "ccx" not in circuit.count_ops()


def test_ibm_campaign_defaults_to_ten_jobs_at_level_three():
    experiment = BVSignalBackgroundIBMExperiment()
    parser = experiment.create_argument_parser()
    parsed = parser.parse_args(["--config", "BVSB28"])
    auto_backend = parser.parse_args(["--select-backend-only"])

    assert parsed.runs == 10
    assert parsed.opt_levels == [3]
    assert parsed.shots == 1024
    assert auto_backend.select_backend_only is True


@pytest.mark.parametrize(
    ("arguments", "expected_min_qubits"),
    [
        (["--select-backend-only"], 29),
        (["--select-backend-only", "--config", "BVSB27"], 27),
    ],
)
def test_ibm_campaign_selects_least_busy_dynamic_backend(
    monkeypatch,
    arguments,
    expected_min_qubits,
):
    captured = {}

    class FakeStatus:
        pending_jobs = 3

    class FakeBackend:
        name = "ibm_test"

        @staticmethod
        def status():
            return FakeStatus()

    class FakeService:
        def __init__(self, **kwargs):
            captured["service_kwargs"] = kwargs

        @staticmethod
        def least_busy(**kwargs):
            captured["least_busy_kwargs"] = kwargs
            return FakeBackend()

    monkeypatch.setattr(campaign_module, "QiskitRuntimeService", FakeService)
    monkeypatch.setattr(
        campaign_module,
        "resolve_ibm_credentials",
        lambda *_args: {"channel": None, "token": None, "instance": None},
    )
    result = BVSignalBackgroundIBMExperiment().run_cli(arguments)

    assert result["backend_name"] == "ibm_test"
    assert captured["least_busy_kwargs"] == {
        "operational": True,
        "simulator": False,
        "dynamic_circuits": True,
        "min_num_qubits": expected_min_qubits,
    }


def test_ibm_campaign_records_signal_diagnostics():
    experiment = BVSignalBackgroundIBMExperiment()
    config = get_config("BVSB28")
    first, second = config.expected_outcomes
    background = "1" + "0" * (config.spec.num_measured_qubits - 1)
    result = experiment.evaluate_result(
        {first: 450, second: 450, background: 100},
        config,
        total_shots=1000,
    )

    assert result["success_rate"] == pytest.approx(0.9)
    assert result["signal_detected"] is True
    assert result["strongest_target_count"] == 450
    assert result["strongest_competitor_count"] == 100
    assert result["hellinger_fidelity"] is None
    assert result["tvd_fidelity"] is None


@pytest.mark.parametrize("bad_total", [0, 5, 7, 12])
def test_rejects_unsupported_proof_size(bad_total):
    with pytest.raises(ValueError):
        make_spec(bad_total)


@pytest.mark.parametrize("bad_mass", [0.0, 1.0, -0.1, 1.1])
def test_rejects_invalid_target_mass(bad_mass):
    with pytest.raises(ValueError):
        make_spec(6, target_mass=bad_mass)
