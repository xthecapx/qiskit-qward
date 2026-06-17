#!/usr/bin/env python3
"""
Phase Estimation AWS Braket Execution Script

Usage:
    uv run python qward/examples/papers/phase_estimation/pe_aws.py --config PE-T4
    uv run python qward/examples/papers/phase_estimation/pe_aws.py --list
"""

from pathlib import Path
from typing import Any, Dict, List, Callable, Optional

from qiskit import QuantumCircuit

from qward.algorithms import PhaseEstimation, PhaseEstimationCircuitGenerator
from qward.examples.papers.aws_experiment_base import AWSExperimentBase
from qward.examples.papers.phase_estimation.pe_configs import (
    get_config,
    PEExperimentConfig,
    CONFIGS_BY_ID,
)

RIGETTI_OPTIMIZATION_LEVEL = 3


class PEAWSExperiment(AWSExperimentBase[PEExperimentConfig]):
    """Phase Estimation experiment runner for AWS Braket."""

    def __init__(self, shots: int = 1024, timeout: int = 600):
        super().__init__(
            shots=shots,
            timeout=timeout,
            output_subdir=str(Path(__file__).resolve().parent / "data" / "qpu" / "aws"),
        )

    @property
    def algorithm_name(self) -> str:
        return "PHASE-ESTIMATION"

    def run(
        self,
        config_id: str,
        device_id: str = "Ankaa-3",
        region: str = "us-west-1",
        save_results: bool = True,
        wait_for_results: bool = True,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        optimization_level: Optional[int] = RIGETTI_OPTIMIZATION_LEVEL,
    ) -> Dict[str, Any]:
        return super().run(
            config_id=config_id,
            device_id=device_id,
            region=region,
            save_results=save_results,
            wait_for_results=wait_for_results,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            optimization_level=optimization_level,
        )

    def get_config(self, config_id: str) -> PEExperimentConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: PEExperimentConfig) -> QuantumCircuit:
        gen = PhaseEstimationCircuitGenerator(
            test_case=config.test_case,
            num_counting_qubits=config.num_counting_qubits,
            use_barriers=True,
        )
        return gen.circuit

    def create_success_criteria(self, config: PEExperimentConfig) -> Callable[[str], bool]:
        expected = config.expected_outcome
        n = config.num_counting_qubits
        search_space = config.search_space

        def is_success(result: str) -> bool:
            clean = result.replace(" ", "").strip()
            if clean == expected:
                return True
            measured = int(clean, 2)
            expected_val = int(expected, 2)
            diff = min(
                abs(measured - expected_val),
                search_space - abs(measured - expected_val),
            )
            return diff <= 1

        return is_success

    def get_expected_outcomes(self, config: PEExperimentConfig) -> List[str]:
        return [config.expected_outcome]

    def get_random_chance(self, config: PEExperimentConfig) -> float:
        return config.classical_random_prob

    def get_config_description(self, config: PEExperimentConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: PEExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        expected_outcome = config.expected_outcome
        p_exact = counts.get(expected_outcome, 0) / total_shots

        measured_phases = {}
        for outcome, count in counts.items():
            clean = outcome.replace(" ", "").strip()
            val = int(clean, 2)
            phase = val / config.search_space
            measured_phases[phase] = measured_phases.get(phase, 0) + count

        top_phase = max(measured_phases, key=measured_phases.get) if measured_phases else 0
        phase_error = abs(top_phase - config.expected_phase)
        phase_error = min(phase_error, 1.0 - phase_error)

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        return {
            "success_rate": s_rate,
            "success_count": s_count,
            "p_exact_outcome": p_exact,
            "expected_outcome": expected_outcome,
            "expected_phase": config.expected_phase,
            "measured_phase": top_phase,
            "phase_error": phase_error,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
        }


if __name__ == "__main__":
    experiment = PEAWSExperiment()
    experiment.run_cli()
