#!/usr/bin/env python3
"""
QFT AWS Braket execution script.

This script runs Quantum Fourier Transform on AWS Braket hardware using the
AWSExperimentBase framework. It reuses Region 1 prioritized configurations
identified through simulator analysis.

Usage:
    python qft_aws.py                        # Run default config (SR2)
    python qft_aws.py --config SR4           # Run specific config
    python qft_aws.py --list                 # List available configs

Example:
    >>> from qward.examples.papers.qft.qft_aws import QFTAWSExperiment
    >>> experiment = QFTAWSExperiment()
    >>> result = experiment.run("SR2", device_id="Ankaa-3")
    >>> print(f"DSR: {result['batch_summary']['mean_dsr_michelson']:.4f}")
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit

from qward.algorithms import AWSJobResult, QFTCircuitGenerator
from qward.examples.papers.aws_experiment_base import AWSExperimentBase
from qward.examples.papers.qft.qft_configs import CONFIGS_BY_ID, QFTExperimentConfig, get_config
from qward.examples.papers.qft.qft_ibm import REGION1_PRIORITY


class QFTAWSExperiment(AWSExperimentBase[QFTExperimentConfig]):
    """QFT algorithm experiment runner for AWS Braket."""

    @property
    def algorithm_name(self) -> str:
        return "QFT"

    def get_config(self, config_id: str) -> QFTExperimentConfig:
        """Get QFT experiment configuration."""
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        return list(CONFIGS_BY_ID.keys())

    def create_circuit(self, config: QFTExperimentConfig) -> QuantumCircuit:
        """Create QFT circuit for the configuration."""
        if config.test_mode == "roundtrip":
            qft_gen = QFTCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="roundtrip",
                input_state=config.input_state,
                use_barriers=True,
            )
        else:
            qft_gen = QFTCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="period_detection",
                period=config.period,
                use_barriers=True,
            )

        self._current_generator = qft_gen
        return qft_gen.circuit

    def create_success_criteria(self, config: QFTExperimentConfig) -> Callable[[str], bool]:
        """Create success criteria for QFT."""
        if hasattr(self, "_current_generator") and self._current_generator is not None:
            return self._current_generator.success_criteria

        if config.test_mode == "roundtrip":
            expected_state = config.input_state

            def roundtrip_success(result: str) -> bool:
                clean_result = result.replace(" ", "").strip()
                return clean_result == expected_state

            return roundtrip_success

        search_space = config.search_space
        period = config.period
        expected_peaks = set()
        step = search_space // period
        for index in range(period):
            peak_value = index * step
            peak_str = format(peak_value, f"0{config.num_qubits}b")
            expected_peaks.add(peak_str)

        def period_success(result: str) -> bool:
            clean_result = result.replace(" ", "").strip()
            return clean_result in expected_peaks

        return period_success

    def get_expected_outcomes(self, config: QFTExperimentConfig) -> List[str]:
        """Get expected outcomes used for DSR calculations."""
        if config.test_mode == "roundtrip":
            return [config.input_state]

        search_space = config.search_space
        period = config.period
        if period is None or period <= 0:
            return []

        step = search_space // period
        peaks = []
        for index in range(period):
            peak_value = index * step
            peaks.append(format(peak_value, f"0{config.num_qubits}b"))
        return peaks

    def get_random_chance(self, config: QFTExperimentConfig) -> float:
        """Get classical random chance for the configuration."""
        if config.test_mode == "roundtrip":
            return 1.0 / config.search_space

        num_peaks = config.search_space // config.period
        return num_peaks / config.search_space

    def get_config_description(self, config: QFTExperimentConfig) -> Dict[str, Any]:
        """Get configuration description for saving."""
        desc: Dict[str, Any] = {
            "config_id": config.config_id,
            "num_qubits": config.num_qubits,
            "test_mode": config.test_mode,
            "search_space": config.search_space,
            "theoretical_gate_count": config.theoretical_gate_count,
            "description": config.description,
        }

        if config.test_mode == "roundtrip":
            desc["input_state"] = config.input_state
        else:
            desc["period"] = config.period
            desc["expected_num_peaks"] = config.expected_num_peaks

        return desc

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: QFTExperimentConfig,
        total_shots: int,
        aws_result: Optional[AWSJobResult] = None,
    ) -> Dict[str, Any]:
        """Evaluate QFT result with algorithm-specific metrics."""
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        result: Dict[str, Any] = {
            "success_rate": s_rate,
            "success_count": s_count,
            "test_mode": config.test_mode,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
            "threshold_95": s_rate >= 0.95,
            "threshold_99": s_rate >= 0.99,
        }

        if config.test_mode == "roundtrip":
            result["input_state"] = config.input_state
        else:
            result["period"] = config.period
            result["expected_num_peaks"] = config.expected_num_peaks

        if aws_result is not None:
            result.update(
                {
                    "dsr_michelson": aws_result.dsr_michelson,
                    "dsr_ratio": aws_result.dsr_ratio,
                    "dsr_log_ratio": aws_result.dsr_log_ratio,
                    "dsr_normalized_margin": aws_result.dsr_normalized_margin,
                    "peak_mismatch": aws_result.peak_mismatch,
                }
            )

        return result

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return REGION1_PRIORITY

    def get_output_dir(self) -> Path:
        """Get output directory for QFT AWS results."""
        return Path(__file__).parent / "data" / "qpu" / "aws"


# =============================================================================
# Convenience Functions
# =============================================================================


def run_qft_on_aws(
    config_id: str = "SR2",
    device_id: str = "Ankaa-3",
    region: str = "us-west-1",
    shots: int = 1024,
    timeout: int = 600,
    save_results: bool = True,
    wait_for_results: bool = True,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run QFT on AWS Braket hardware."""
    experiment = QFTAWSExperiment(shots=shots, timeout=timeout)
    return experiment.run(
        config_id=config_id,
        device_id=device_id,
        region=region,
        save_results=save_results,
        wait_for_results=wait_for_results,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def list_configs() -> None:
    """List available QFT configurations."""
    experiment = QFTAWSExperiment()
    experiment.list_configs()


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run QFT AWS experiment from command line."""
    experiment = QFTAWSExperiment()
    experiment.run_cli()


if __name__ == "__main__":
    main()
