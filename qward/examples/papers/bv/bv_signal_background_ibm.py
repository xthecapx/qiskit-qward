#!/usr/bin/env python3
"""IBM runner for the BV-derived signal-plus-background campaign.

The default execution submits one batch containing ten independent jobs at
optimization level 3. Run each configuration separately so its ten jobs share
one IBM batch and one result file.

Preflight without submitting:
    uv run -m qward.examples.papers.bv.bv_signal_background_ibm \
        --config BVSB28 --preflight-only

Submit ten jobs:
    uv run -m qward.examples.papers.bv.bv_signal_background_ibm \
        --config BVSB28 --backend ibm_fez

Recover a timed-out batch:
    uv run -m qward.examples.papers.bv.bv_signal_background_ibm \
        --recover --batch-id <BATCH_ID> --recover-config BVSB28 \
        --recover-backend ibm_fez --recover-opt-level 3
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

from qward.examples.papers.bv.bv_signal_background import (
    build_dynamic_signal_background_circuit,
    dsr_profile,
    heavy_hex_transpiled_complexity,
)
from qward.examples.papers.bv.bv_signal_background_configs import (
    CONFIGS,
    BVSignalBackgroundConfig,
    get_config,
)
from qward.examples.papers.ibm_experiment_base import (
    IBMExperimentBase,
    resolve_ibm_credentials,
)


class BVSignalBackgroundIBMExperiment(IBMExperimentBase[BVSignalBackgroundConfig]):
    """Run and recover the large signal-plus-background configurations."""

    def __init__(self, shots: int = 1024, timeout: int = 7200):
        output_dir = Path(__file__).resolve().parent / "data" / "qpu"
        output_dir = output_dir / "signal_background" / "raw"
        super().__init__(
            shots=shots,
            timeout=timeout,
            output_subdir=str(output_dir),
        )

    @property
    def algorithm_name(self) -> str:
        return "BV-SIGNAL-BACKGROUND"

    def get_config(self, config_id: str) -> BVSignalBackgroundConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS)

    def create_circuit(self, config: BVSignalBackgroundConfig) -> QuantumCircuit:
        return build_dynamic_signal_background_circuit(config.spec)

    def create_success_criteria(
        self,
        config: BVSignalBackgroundConfig,
    ) -> Callable[[str], bool]:
        expected = set(config.expected_outcomes)

        def is_success(result: str) -> bool:
            return result.replace(" ", "").strip() in expected

        return is_success

    def get_random_chance(self, config: BVSignalBackgroundConfig) -> float:
        return config.random_chance

    def get_config_description(
        self,
        config: BVSignalBackgroundConfig,
    ) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: BVSignalBackgroundConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        """Compute counts-only metrics and explicit signal diagnostics."""
        del total_shots
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

        result = {
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
        return result

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
                    "description": "Two BV targets plus broad background",
                }
            )
        return configs

    def _build_rich_result(self, *args, **kwargs) -> Dict[str, Any]:
        result = super()._build_rich_result(*args, **kwargs)
        runs = result.get("individual_results", [])
        completed = [run for run in runs if run.get("counts")]
        detected = [run for run in completed if run.get("signal_detected")]
        dsr_values = [
            float(run["dsr_michelson"]) for run in completed if run.get("dsr_michelson") is not None
        ]
        summary = result.setdefault("batch_summary", {})
        summary["signal_detected_runs"] = len(detected)
        summary["signal_evaluated_runs"] = len(completed)
        summary["signal_detection_rate"] = len(detected) / len(completed) if completed else None
        if dsr_values:
            summary["mean_dsr_michelson"] = statistics.mean(dsr_values)
            summary["median_dsr_michelson"] = statistics.median(dsr_values)
        return result

    def create_argument_parser(self):
        parser = super().create_argument_parser()
        parser.set_defaults(
            opt_levels=[3],
            runs=10,
            shots=1024,
            timeout=7200,
        )
        parser.add_argument(
            "--select-backend-only",
            action="store_true",
            help=(
                "Print the least-busy operational dynamic-circuit backend "
                "with enough qubits for --config, or the largest configuration"
            ),
        )
        parser.add_argument(
            "--preflight-only",
            action="store_true",
            help="Estimate heavy-hex routing complexity without submitting jobs",
        )
        return parser

    def run_cli(
        self,
        args: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)
        if parsed.select_backend_only:
            credentials = resolve_ibm_credentials(
                parsed.channel,
                parsed.token,
                parsed.instance,
            )
            service_kwargs = {key: value for key, value in credentials.items() if value is not None}
            service = QiskitRuntimeService(**service_kwargs)
            min_num_qubits = (
                CONFIGS[parsed.config].num_total_qubits
                if parsed.config in CONFIGS
                else max(config.num_total_qubits for config in CONFIGS.values())
            )
            backend = service.least_busy(
                operational=True,
                simulator=False,
                dynamic_circuits=True,
                min_num_qubits=min_num_qubits,
            )
            print(backend.name)
            return {
                "backend_name": backend.name,
                "pending_jobs": backend.status().pending_jobs,
                "dynamic_circuits": True,
            }
        if parsed.preflight_only:
            config_id = parsed.config or "BVSB28"
            if config_id not in CONFIGS:
                parser.error(f"unknown config {config_id!r}; available: {sorted(CONFIGS)}")
            config = self.get_config(config_id)
            circuit = self.create_circuit(config)
            report = {
                "config": config.to_dict(),
                "original_depth": circuit.depth(),
                "original_size": circuit.size(),
                "dynamic_full_circuit": heavy_hex_transpiled_complexity(
                    config.spec,
                    dynamic=True,
                ),
                "signal_path_estimate": heavy_hex_transpiled_complexity(
                    config.spec,
                    dynamic=True,
                    signal_path_only=True,
                ),
                "jobs_to_submit": parsed.runs * len(parsed.opt_levels),
                "optimization_levels": parsed.opt_levels,
                "shots": parsed.shots,
                "submission_performed": False,
            }
            print(json.dumps(report, indent=2, default=str))
            return report
        return super().run_cli(args)


if __name__ == "__main__":
    BVSignalBackgroundIBMExperiment().run_cli()
