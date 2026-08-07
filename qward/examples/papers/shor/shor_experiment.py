"""Shor simulator campaign runner (BaseExperimentRunner)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit

from qward.algorithms import Shor
from qward.algorithms.experiment import (
    BaseBatchResult,
    BaseExperimentResult,
    BaseExperimentRunner,
)
from qward.algorithms.noise_generator import NoiseConfig, get_preset_noise_config
from qward.examples.papers.shor.shor_configs import (
    CONFIGS_BY_ID,
    ExperimentConfig,
    get_config,
)
from qward.examples.papers.shor.shor_statistical_analysis import (
    ConfigAnalysis,
    analyze_config_results,
    print_analysis_summary,
)
from qward.examples.papers.shor.shor_success_metrics import evaluate_counts

SHOTS = 1024
NUM_RUNS = 5
OPTIMIZATION_LEVEL = 0
DEFAULT_NOISE_IDS = ["IDEAL"]


@dataclass
class ShorExperimentResult(BaseExperimentResult):
    N: int = 0
    a: int = 0
    true_order: int = 0
    factoring_success_rate: float = 0.0
    uninformative_rate: float = 0.0
    random_chance: float = 0.0
    advantage_ratio: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShorExperimentResult":
        return cls(
            experiment_id=data.get("experiment_id", ""),
            config_id=data.get("config_id", ""),
            noise_model=data.get("noise_model", ""),
            run_number=data.get("run_number", 0),
            timestamp=data.get("timestamp", ""),
            backend_type=data.get("backend_type", "simulator"),
            backend_name=data.get("backend_name", "AerSimulator"),
            num_qubits=data.get("num_qubits", 0),
            circuit_depth=data.get("circuit_depth", 0),
            total_gates=data.get("total_gates", 0),
            qward_metrics=data.get("qward_metrics"),
            shots=data.get("shots", SHOTS),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            counts=data.get("counts", {}),
            success_rate=data.get("success_rate", 0.0),
            success_count=data.get("success_count", 0),
            N=data.get("N", 0),
            a=data.get("a", 0),
            true_order=data.get("true_order", 0),
            factoring_success_rate=data.get("factoring_success_rate", 0.0),
            uninformative_rate=data.get("uninformative_rate", 0.0),
            random_chance=data.get("random_chance", 0.0),
            advantage_ratio=data.get("advantage_ratio", 0.0),
        )


@dataclass
class ShorBatchResult(BaseBatchResult[ShorExperimentResult, ConfigAnalysis]):
    pass


class ShorExperimentRunner(
    BaseExperimentRunner[ExperimentConfig, ShorExperimentResult, ShorBatchResult, ConfigAnalysis]
):
    def __init__(
        self,
        shots: int = SHOTS,
        num_runs: int = NUM_RUNS,
        optimization_level: int = OPTIMIZATION_LEVEL,
        output_dir: str = "data/simulator",
        backend_type: str = "simulator",
        backend_name: str = "AerSimulator",
    ):
        base_dir = Path(__file__).parent / output_dir
        super().__init__(
            shots=shots,
            num_runs=num_runs,
            optimization_level=optimization_level,
            output_dir=str(base_dir),
            backend_type=backend_type,
            backend_name=backend_name,
        )

    @property
    def algorithm_name(self) -> str:
        return "SHOR"

    def create_circuit(self, config: ExperimentConfig) -> Tuple[QuantumCircuit, Shor]:
        shor = Shor(
            config.N,
            config.a,
            num_control=config.num_control,
            strategy=config.strategy,
        )
        return shor.circuit, shor

    def calculate_success(
        self,
        counts: Dict[str, int],
        config: ExperimentConfig,
        circuit_metadata: Shor,
    ) -> Tuple[float, int]:
        ev = evaluate_counts(
            counts,
            a=config.a,
            N=config.N,
            num_control=config.num_control,
            true_order=config.true_order,
            random_chance=config.classical_random_prob,
        )
        return ev.success_rate, ev.success_count

    def create_result(
        self,
        config: ExperimentConfig,
        noise_config: NoiseConfig,
        run_number: int,
        transpiled_circuit: QuantumCircuit,
        counts: Dict[str, int],
        execution_time_ms: float,
        success_rate: float,
        success_count: int,
        qward_metrics: Optional[Dict[str, Any]],
        circuit_metadata: Shor,
        backend_type: str,
        backend_name: str,
    ) -> ShorExperimentResult:
        ev = evaluate_counts(
            counts,
            a=config.a,
            N=config.N,
            num_control=config.num_control,
            true_order=config.true_order,
            random_chance=config.classical_random_prob,
        )
        return ShorExperimentResult(
            experiment_id=f"{config.config_id}_{noise_config.noise_id}_{run_number:03d}",
            config_id=config.config_id,
            noise_model=noise_config.noise_id,
            run_number=run_number,
            timestamp=datetime.now().isoformat(),
            backend_type=backend_type,
            backend_name=backend_name,
            num_qubits=config.num_qubits,
            circuit_depth=transpiled_circuit.depth(),
            total_gates=sum(transpiled_circuit.count_ops().values()),
            qward_metrics=qward_metrics,
            shots=self.shots,
            execution_time_ms=execution_time_ms,
            counts=counts,
            success_rate=success_rate,
            success_count=success_count,
            N=config.N,
            a=config.a,
            true_order=config.true_order,
            factoring_success_rate=ev.factoring_success_rate,
            uninformative_rate=ev.uninformative_rate,
            random_chance=ev.random_chance,
            advantage_ratio=ev.advantage_ratio,
        )

    def get_config(self, config_id: str) -> ExperimentConfig:
        return get_config(config_id)

    def get_noise_config(self, noise_id: str) -> NoiseConfig:
        try:
            return get_preset_noise_config(noise_id)
        except Exception:
            return NoiseConfig(noise_id=noise_id, noise_type="ideal", description=noise_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS_BY_ID.keys())

    def get_all_noise_ids(self) -> List[str]:
        return DEFAULT_NOISE_IDS

    def analyze_batch(
        self,
        success_rates: List[float],
        config_id: str,
        noise_model: str,
        ideal_rates: Optional[List[float]] = None,
    ) -> Optional[ConfigAnalysis]:
        return analyze_config_results(success_rates, config_id, noise_model, ideal_rates)

    def print_batch_analysis(self, analysis: Optional[ConfigAnalysis]) -> None:
        if analysis is not None:
            print_analysis_summary(analysis)

    def load_result_from_dict(self, data: Dict[str, Any]) -> ShorExperimentResult:
        return ShorExperimentResult.from_dict(data)

    def load_analysis_from_dict(self, data: Optional[Dict[str, Any]]) -> Optional[ConfigAnalysis]:
        if data is None:
            return None
        return ConfigAnalysis(
            config_id=data.get("config_id", ""),
            noise_model=data.get("noise_model", ""),
            num_runs=data.get("num_runs", 0),
            mean=data.get("mean", 0.0),
            std=data.get("std", 0.0),
            median=data.get("median", 0.0),
            min_val=data.get("min_val", data.get("min", 0.0)),
            max_val=data.get("max_val", data.get("max", 0.0)),
            ci_lower=data.get("ci_lower", 0.0),
            ci_upper=data.get("ci_upper", 0.0),
        )


def main():
    print("Shor simulator experiment runner ready.")
    print("Example: ShorExperimentRunner().run_batch('SHOR-N15-A7-SIM', 'IDEAL', num_runs=3)")


if __name__ == "__main__":
    main()
