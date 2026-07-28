#!/usr/bin/env python3
"""AWS Braket runner for the BV-derived signal-plus-background campaign.

Unlike the IBM runner (bv_signal_background_ibm.py), this uses the
*coherently*-controlled circuit (build_signal_background_circuit) instead of
the dynamic-circuit variant (mid-circuit measurement + if_test). AWS
Braket / Rigetti does not support conditional branching on a mid-circuit
measurement, so the coherent version is the only portable option. Both
variants produce the same ideal measurement distribution (see
bv_signal_background.py), so DSR/statistics are directly comparable.

Usage:
    uv run python qward/examples/papers/bv/bv_signal_background_aws.py --config BVSB27
    uv run python qward/examples/papers/bv/bv_signal_background_aws.py --list
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit

from qward.algorithms import AWSJobResult
from qward.examples.papers.aws_experiment_base import AWSExperimentBase
from qward.examples.papers.bv.bv_signal_background import (
    build_signal_background_circuit,
    dsr_profile,
)
from qward.examples.papers.bv.bv_signal_background_configs import (
    CONFIGS,
    BVSignalBackgroundConfig,
    get_config,
)

RIGETTI_OPTIMIZATION_LEVEL = 3


class BVSignalBackgroundAWSExperiment(AWSExperimentBase[BVSignalBackgroundConfig]):
    """BV signal-plus-background experiment runner for AWS Braket."""

    def __init__(self, shots: int = 1024, timeout: int = 600):
        output_dir = Path(__file__).resolve().parent / "data" / "qpu" / "signal_background" / "aws"
        super().__init__(shots=shots, timeout=timeout, output_subdir=str(output_dir))

    @property
    def algorithm_name(self) -> str:
        return "BV-SIGNAL-BACKGROUND"

    def run(
        self,
        config_id: str,
        device_id: str = "Cepheus-1-108Q",
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

    def get_config(self, config_id: str) -> BVSignalBackgroundConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS)

    def create_circuit(self, config: BVSignalBackgroundConfig) -> QuantumCircuit:
        return build_signal_background_circuit(config.spec)

    def create_success_criteria(self, config: BVSignalBackgroundConfig) -> Callable[[str], bool]:
        expected = set(config.expected_outcomes)

        def is_success(result: str) -> bool:
            return result.replace(" ", "").strip() in expected

        return is_success

    def get_expected_outcomes(self, config: BVSignalBackgroundConfig) -> List[str]:
        return list(config.expected_outcomes)

    def get_random_chance(self, config: BVSignalBackgroundConfig) -> float:
        return config.random_chance

    def get_config_description(self, config: BVSignalBackgroundConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: BVSignalBackgroundConfig,
        total_shots: int,
        aws_result: Optional[AWSJobResult] = None,
    ) -> Dict[str, Any]:
        """Compute counts-only metrics and explicit signal diagnostics.

        DSR is computed directly from counts via the target-weighted
        DSRProfiler (dsr_profile), matching the IBM runner exactly; the
        generic AWSJobResult DSR fields are not used here.
        """
        del aws_result, total_shots
        clean_counts = {
            str(outcome).replace(" ", ""): int(count) for outcome, count in counts.items()
        }
        profile = dsr_profile(clean_counts, config.spec)
        expected = set(config.expected_outcomes)
        target_counts = [clean_counts.get(target, 0) for target in expected]
        competitor_counts = [
            count for outcome, count in clean_counts.items() if outcome not in expected
        ]
        strongest_target = max(target_counts, default=0)
        strongest_competitor = max(competitor_counts, default=0)
        total = sum(clean_counts.values())
        branch_zero_count = sum(
            count for outcome, count in clean_counts.items() if outcome.startswith("0")
        )

        ratio = None
        if strongest_competitor > 0:
            ratio = strongest_target / strongest_competitor
        elif strongest_target > 0:
            ratio = float("inf")

        return {
            **profile,
            "expected_outcome": None,
            "expected_outcomes": sorted(expected),
            "target_mass": config.target_mass,
            "background_mass": 1.0 - config.target_mass,
            "branch_zero_rate": branch_zero_count / total if total else 0.0,
            "strongest_target_count": strongest_target,
            "strongest_competitor_count": strongest_competitor,
            "signal_to_competitor_ratio": ratio,
            "signal_detected": bool(
                profile["dsr_michelson"] > 0.0 and not profile.get("peak_mismatch", True)
            ),
            "hellinger_fidelity": None,
            "hellinger_distance": None,
            "tvd": None,
            "tvd_fidelity": None,
            "ideal_histogram_status": "not_computed_beyond_local_wall",
            "ideal_histogram_reason": (
                "Exact dense-statevector enrichment was not attempted for "
                f"{config.num_total_qubits} total qubits."
            ),
        }

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        configs = []
        for config in CONFIGS.values():
            circuit = self.create_circuit(config)
            configs.append(
                {
                    "config_id": config.config_id,
                    "qubits": config.num_total_qubits,
                    "depth": circuit.depth(),
                    "expected_success": config.target_mass,
                    "description": "Two BV targets plus broad background (coherent, AWS)",
                }
            )
        return configs

    def get_output_dir(self) -> Path:
        return Path(self.output_subdir)


if __name__ == "__main__":
    BVSignalBackgroundAWSExperiment().run_cli()
