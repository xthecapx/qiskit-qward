#!/usr/bin/env python3
"""Shor IBM QPU execution via IBMExperimentBase.

Each QPU submit first runs a noiseless AerSimulator baseline of the *same*
logical circuit (same N, a, m, strategy, shots). Baseline metrics are printed
and stored under ``simulator_baseline`` in the saved JSON.

Usage:
    uv run python qward/examples/papers/shor/shor_ibm.py \\
      --config SHOR-N15-M8 --opt-levels 3 --runs 5 --shots 4096 --timeout 3600

    # Aer baseline only (no QPU spend):
    uv run python qward/examples/papers/shor/shor_ibm.py \\
      --config SHOR-N15-M8 --shots 1024 --simulator-only

    # Skip Aer (not recommended):
    uv run python qward/examples/papers/shor/shor_ibm.py \\
      --config SHOR-N15-M8 --opt-levels 3 --runs 5 --skip-simulator
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qward.algorithms import Shor
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase
from qward.examples.papers.shor.shor_configs import (
    QPU_CONFIGS,
    ExperimentConfig,
    get_config,
)
from qward.examples.papers.shor.shor_success_metrics import (
    evaluate_counts,
    evaluation_to_dict,
)

SHOR_SAMPLER_OPTIONS = {
    "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
    "twirling": {"enable_gates": True},
}

REGION1_PRIORITY = [
    {
        "config_id": c.config_id,
        "expected_success": 0.5,
        "qubits": c.num_qubits,
        "depth": 0,
        "description": c.description,
    }
    for c in QPU_CONFIGS
]


class ShorIBMExperiment(IBMExperimentBase[ExperimentConfig]):
    """Shor order-finding on IBM QPU (with Aer baseline)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sim_baseline: Optional[Dict[str, Any]] = None
        self._run_simulator: bool = True
        self._simulator_only: bool = False

    @property
    def algorithm_name(self) -> str:
        return "SHOR"

    def get_config(self, config_id: str) -> ExperimentConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        # Never expose SIM-only configs to the QPU CLI path.
        return [c.config_id for c in QPU_CONFIGS]

    def create_circuit(self, config: ExperimentConfig) -> QuantumCircuit:
        if not config.for_qpu:
            raise ValueError(
                f"{config.config_id} is simulator-only (for_qpu=False); "
                "refusing to build a QPU job."
            )
        if config.strategy != "swap_network":
            raise ValueError(
                f"{config.config_id} strategy={config.strategy!r} is not "
                "hardware-safe; QPU configs must use swap_network."
            )
        shor = Shor(
            config.N,
            config.a,
            num_control=config.num_control,
            strategy=config.strategy,
            use_barriers=True,
        )
        return shor.circuit

    def create_success_criteria(self, config: ExperimentConfig) -> Callable[[str], bool]:
        from qward.algorithms.shor import bitstring_to_phase, phase_to_order

        true_order = config.true_order
        m = config.num_control
        N = config.N

        def is_success(result: str) -> bool:
            phase = bitstring_to_phase(result, m)
            r, _ = phase_to_order(phase, N)
            return r == true_order

        return is_success

    def get_random_chance(self, config: ExperimentConfig) -> float:
        return config.classical_random_prob

    def get_config_description(self, config: ExperimentConfig) -> Dict[str, Any]:
        return config.to_dict()

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: ExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        ev = evaluate_counts(
            counts,
            a=config.a,
            N=config.N,
            num_control=config.num_control,
            true_order=config.true_order,
            random_chance=config.classical_random_prob,
        )
        return evaluation_to_dict(ev)

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        return REGION1_PRIORITY

    def get_output_dir(self) -> Path:
        return Path(__file__).parent / "data" / "qpu" / "raw"

    def run_aer_baseline(
        self,
        config: ExperimentConfig,
        *,
        shots: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Noiseless AerSimulator baseline of the logical Shor circuit."""
        shots = shots if shots is not None else self.shots
        print("\n" + "=" * 70)
        print("AER SIMULATOR BASELINE (before QPU submit)")
        print("=" * 70)
        print(
            f"Config {config.config_id}: N={config.N} a={config.a} "
            f"m={config.num_control} strategy={config.strategy} shots={shots}"
        )

        shor = Shor(
            config.N,
            config.a,
            num_control=config.num_control,
            strategy=config.strategy,
            use_barriers=True,
        )
        circuit = shor.circuit
        sim = AerSimulator(method="statevector")
        t0 = time.perf_counter()
        isa = transpile(circuit, backend=sim, optimization_level=1)
        job = sim.run(isa, shots=shots)
        counts = {k.replace(" ", ""): v for k, v in job.result().get_counts().items()}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        evaluation = self.evaluate_result(counts, config, shots)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]

        baseline = {
            "backend": "AerSimulator",
            "method": "statevector",
            "optimization_level": 1,
            "shots": shots,
            "num_qubits": circuit.num_qubits,
            "logical_depth": circuit.depth(),
            "transpiled_depth": isa.depth(),
            "execution_time_ms": elapsed_ms,
            "counts": counts,
            "top_counts": top,
            "evaluation": evaluation,
            "note": (
                "Noiseless logical-circuit baseline (no DD/twirling). "
                "Compare QPU success_rate / factoring_success_rate against these."
            ),
        }

        print(f"  qubits={baseline['num_qubits']} logical_depth={baseline['logical_depth']} "
              f"aer_depth={baseline['transpiled_depth']} time={elapsed_ms:.0f}ms")
        print(
            f"  success_rate={evaluation['success_rate']:.2%} "
            f"(true order r={config.true_order})  "
            f"factoring={evaluation['factoring_success_rate']:.2%}  "
            f"uninformative={evaluation['uninformative_rate']:.2%}"
        )
        print(
            f"  random_chance={evaluation['random_chance']:.2%}  "
            f"advantage={evaluation['advantage_ratio']:.2f}x"
        )
        if evaluation.get("notes"):
            print(f"  note: {evaluation['notes']}")
        print("  top outcomes:")
        for bits, count in top[:5]:
            print(f"    {bits}: {count}")
        print("=" * 70)

        if evaluation["success_rate"] < 0.30 and config.num_control >= 4:
            raise RuntimeError(
                f"Aer baseline success_rate={evaluation['success_rate']:.2%} "
                f"for {config.config_id} is below 30% — aborting QPU submit. "
                "Fix the circuit/metrics before spending hardware time."
            )

        return baseline

    def _build_rich_result(
        self,
        ibm_result: Any,
        config: ExperimentConfig,
        circuit: QuantumCircuit,
        qward_metrics: Dict[str, Any],
        original_depth: int,
        original_gates: int,
    ) -> Dict[str, Any]:
        result = super()._build_rich_result(
            ibm_result, config, circuit, qward_metrics, original_depth, original_gates
        )
        if self._sim_baseline is not None:
            result["simulator_baseline"] = self._sim_baseline
        return result

    def run(self, config_id: str, *args, **kwargs):
        config = self.get_config(config_id)
        if not config.for_qpu:
            raise ValueError(
                f"{config_id} is simulator-only; refusing QPU execution. "
                f"Allowed: {self.get_all_config_ids()}"
            )
        if config.strategy != "swap_network":
            raise ValueError(
                f"{config_id} strategy={config.strategy!r} is not QPU-safe."
            )

        self._sim_baseline = None
        if self._run_simulator:
            self._sim_baseline = self.run_aer_baseline(config, shots=self.shots)

        if self._simulator_only:
            out = {
                "config_id": config_id,
                "status": "simulator_only",
                "execution_type": "AER_BASELINE",
                "simulator_baseline": self._sim_baseline,
                "config": config.to_dict(),
            }
            if kwargs.get("save_results", True) and self._sim_baseline is not None:
                save_path = self._save_sim_baseline(out, config)
                print(f"\nSimulator baseline saved to: {save_path}")
            return out

        original_run_ibm = self.executor.run_ibm

        def run_ibm_with_options(*a, **kw):
            kw.setdefault("sampler_options", SHOR_SAMPLER_OPTIONS)
            return original_run_ibm(*a, **kw)

        self.executor.run_ibm = run_ibm_with_options  # type: ignore[method-assign]
        try:
            return super().run(config_id, *args, **kwargs)
        finally:
            self.executor.run_ibm = original_run_ibm  # type: ignore[method-assign]

    def _save_sim_baseline(
        self, result: Dict[str, Any], config: ExperimentConfig
    ) -> Path:
        output_dir = Path(__file__).parent / "data" / "simulator" / "baseline"
        output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        import json

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{config.config_id}_AER_{ts}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return path

    def create_argument_parser(self):
        parser = super().create_argument_parser()
        parser.add_argument(
            "--skip-simulator",
            action="store_true",
            help="Skip AerSimulator baseline before QPU submit (not recommended)",
        )
        parser.add_argument(
            "--simulator-only",
            action="store_true",
            help="Run AerSimulator baseline only; do not submit to IBM QPU",
        )
        return parser

    def run_cli(self, args: Optional[List[str]] = None):
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)

        self._run_simulator = not parsed.skip_simulator
        self._simulator_only = parsed.simulator_only

        if self._simulator_only and parsed.skip_simulator:
            print("Error: --simulator-only and --skip-simulator are mutually exclusive")
            return None

        # Mirror parent status/update/recover/list handling, then short-circuit
        # simulator-only so we never touch IBM credentials.
        if parsed.status or parsed.update or parsed.recover or parsed.list:
            return super().run_cli(args)

        if parsed.config is None:
            priority = self.get_priority_configs()
            parsed.config = (
                priority[0]["config_id"]
                if priority
                else (self.get_all_config_ids() or [None])[0]
            )
        if parsed.config not in self.get_all_config_ids():
            print(f"Error: Unknown config '{parsed.config}'")
            print(f"Available QPU configs: {self.get_all_config_ids()}")
            return None

        self.shots = parsed.shots
        self.timeout = parsed.timeout
        from qward.algorithms.executor import QuantumCircuitExecutor

        self.executor = QuantumCircuitExecutor(shots=parsed.shots)

        if self._simulator_only:
            return self.run(
                config_id=parsed.config,
                save_results=not parsed.no_save,
            )

        return super().run_cli(args)


def main():
    experiment = ShorIBMExperiment()
    experiment.run_cli()


if __name__ == "__main__":
    main()
