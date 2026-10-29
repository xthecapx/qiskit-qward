"""Local noisy-simulator baseline for the tracked BV calibration campaign.

Noise is generated from a synthetic Target with known average gate errors.
Clifford circuits use Aer stabilizer trajectories; the pi/6 diagnostic uses
matrix-product-state trajectories. Dense statevector/density-matrix simulation
and IBM services are never used by this runner.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import InstructionProperties
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qward.algorithms.executor import IBMBatchResult, IBMJobResult
from qward.examples.papers.bv.bv_calibration_analysis import export
from qward.examples.papers.bv.bv_calibration_configs import CONFIGS, SCALING_CONFIGS
from qward.examples.papers.bv.bv_calibration_ibm import (
    BVCalibrationIBMExperiment,
    _now,
    _write,
)
from qward.metrics.circuit_error_prediction import record_hash

DEFAULT_ROOT = Path(__file__).resolve().parent / "data" / "simulator" / "calibration_prediction"


@dataclass(frozen=True)
class SimulatorNoiseProfile:
    """Errors are average gate infidelities and symmetric readout-flip rates."""

    name: str
    one_qubit_error: float
    two_qubit_error: float
    readout_error: float


PROFILES = {
    "IDEAL": SimulatorNoiseProfile("IDEAL", 0.0, 0.0, 0.0),
    "NOMINAL": SimulatorNoiseProfile("NOMINAL", 0.001, 0.01, 0.02),
    "STRESS": SimulatorNoiseProfile("STRESS", 0.005, 0.05, 0.08),
}


def synthetic_backend(profile: SimulatorNoiseProfile, line_length: int = 5) -> GenericBackendV2:
    """Two disjoint line allocations; the second has twice the error.

    The default five-qubit lines reproduce the original baseline backend.
    BV with m data bits needs ``line_length >= m + 1`` for its ancilla.

    Virtual RZ has zero error. NoiseModel.from_backend converts average gate
    infidelity r to depolarization probability r*d/(d-1), consistently with the
    calibration snapshot, and reads symmetric readout errors from the Target.
    """
    edges = []
    for offset in (0, line_length):
        for left in range(offset, offset + line_length - 1):
            edges.extend([[left, left + 1], [left + 1, left]])
    backend = GenericBackendV2(
        2 * line_length,
        basis_gates=["x", "sx", "rz", "cx"],
        coupling_map=edges,
        seed=11,
    )
    backend.name = f"bv_synthetic_{profile.name.lower()}"
    for name in backend.target.operation_names:
        for qubits, props in list(backend.target[name].items()):
            if props is None:
                continue
            scale = 2 if min(qubits) >= line_length else 1
            if name == "measure":
                error = profile.readout_error * scale
            elif name in {"rz", "reset", "delay"}:
                error = 0.0
            else:
                error = (
                    profile.two_qubit_error if len(qubits) == 2 else profile.one_qubit_error
                ) * scale
            backend.target.update_instruction_properties(
                name,
                qubits,
                InstructionProperties(duration=props.duration, error=error),
            )
    return backend


def simulator_method(circuit) -> str:
    """Select stabilizer for the compiled Clifford gate set; MPS for non-Clifford RZ."""
    clifford = {
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "cx",
        "cz",
        "swap",
        "id",
        "barrier",
        "delay",
        "measure",
        "reset",
    }
    for item in circuit.data:
        name = item.operation.name
        if name == "rz":
            turns = float(item.operation.params[0]) / (np.pi / 2)
            if not np.isclose(turns, round(turns), atol=1e-10, rtol=0):
                return "matrix_product_state"
        elif name not in clifford:
            raise ValueError(f"Unexpected native simulator operation: {name}")
    return "stabilizer"


class BVCalibrationSimulatorExperiment(BVCalibrationIBMExperiment):
    """Share circuit preparation, frozen predictions and result schema with IBM."""

    execution_type = "NOISY_SIMULATOR"
    backend_type = "simulator"

    def run(
        self,
        config_id: str,
        *,
        profile: str = "NOMINAL",
        num_runs: int = 3,
        seed: int = 20261022,
        campaign_id: str = "bv-simulator-baseline",
        planned_executions: int = 75,
        noise_model_path: Optional[Path] = None,
        backend=None,
        noise_model=None,
    ) -> Dict:
        """Execute local Aer shots; no Runtime service or hardware entry point."""
        if profile not in PROFILES or num_runs < 1 or seed < 0:
            raise ValueError("Unknown noise profile or invalid repetitions/seed")
        self.noise_id = profile
        config = self.get_config(config_id)
        backend = synthetic_backend(PROFILES[profile]) if backend is None else backend
        noise = (
            NoiseModel.from_backend(backend, thermal_relaxation=False)
            if noise_model is None
            else noise_model
        )
        if noise_model_path is not None:
            if record_hash(json.loads(noise_model_path.read_text())) != record_hash(
                noise.to_dict()
            ):
                raise ValueError("Archived noise model differs from the simulator's noise model")
        else:
            noise_model_path = (
                self.get_output_dir()
                / "noise_models"
                / f"{profile}_{record_hash(noise.to_dict())[:16]}.json"
            )
            noise_model_path.parent.mkdir(parents=True, exist_ok=True)
            _write(noise_model_path, noise.to_dict())
        offset = 0 if config.layout_label == "low_predicted_error" else backend.num_qubits // 2
        context = {
            "profile": asdict(PROFILES[profile]),
            "seed_base": seed,
            "num_runs": num_runs,
            "window_kind": "stationary_simulation_replicates",
            "noise_model_hash": record_hash(noise.to_dict()),
            "noise_model_path": str(noise_model_path) if noise_model_path else None,
            "thermal_relaxation": False,
            "readout": "independent symmetric assignment errors",
            "calibration_source": "synthetic Target; not an IBM calibration snapshot",
            "gate_errors": "local depolarizing channels matched to Target average infidelities",
            "idle_noise": "disabled; timing/T1/T2 retained only as features, not injected noise",
        }
        path = self.prepare(
            config_id,
            backend,
            initial_layout=list(range(offset, offset + config.num_total_qubits)),
            campaign_id=f"{campaign_id}-{profile}",
            planned_executions=planned_executions,
            execution_context=context,
        )
        journal, prepared, circuits = self._load(path)
        method = simulator_method(circuits["isa.qpy"])
        simulator = AerSimulator(
            method=method,
            noise_model=noise,
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )
        profile_index = list(PROFILES).index(profile)
        config_index = sorted({**CONFIGS, **SCALING_CONFIGS}).index(config_id)
        jobs = []
        journal.update(status="running_locally", simulator_method=method, simulation=context)
        _write(path, journal)
        for run_index in range(num_runs):
            shot_seed = seed + profile_index * 1_000_000 + run_index * 10_000 + config_index
            entry = {
                "run_index": run_index,
                "status": "running_locally",
                "job_id": None,
                "started_at": _now(),
                "seed_simulator": shot_seed,
                "method": method,
            }
            journal["submission_journal"].append(entry)
            _write(path, journal)
            try:
                job = simulator.run(circuits["isa.qpy"], shots=self.shots, seed_simulator=shot_seed)
                entry["job_id"] = job.job_id()
                _write(path, journal)
                result = job.result(timeout=self.timeout)
                if not result.success:
                    raise RuntimeError(f"Aer failed: {result.status}")
                counts = dict(result.get_counts(0))
                actual_method = result.results[0].metadata["method"]
                if actual_method != method:
                    raise RuntimeError(f"Unexpected simulator method {actual_method}")
                entry.update(
                    status="completed", finished_at=_now(), aer_metadata=result.results[0].metadata
                )
                jobs.append(
                    IBMJobResult(
                        job_id=job.job_id(),
                        optimization_level=prepared["optimization_level"],
                        status="completed",
                        counts=counts,
                        circuit_depth=circuits["isa.qpy"].depth(),
                        run_index=run_index,
                    )
                )
                batch = IBMBatchResult(
                    batch_id=f"local-{prepared['run_id']}",
                    backend_name=backend.name,
                    status="completed" if len(jobs) == num_runs else "incomplete",
                    jobs=list(jobs),
                )
                journal = self.finish_record(path, journal, prepared, circuits, batch)
            except Exception as exc:
                entry.update(status="failed", error=str(exc))
                journal.update(status="failed", error=str(exc))
                _write(path, journal)
                raise
        return {"record_path": str(path), "status": journal["status"], "method": method}


def select_configs(scope: str) -> List[str]:
    """smoke: 2-bit low-error set; core: IBM matrix; scaling: primary BV, 2-8 bits."""
    if scope == "smoke":
        return [
            c.config_id
            for c in CONFIGS.values()
            if c.num_qubits == 2 and c.layout_label == "low_predicted_error"
        ]
    if scope == "core":
        return list(CONFIGS)
    if scope == "scaling":
        return [c.config_id for c in CONFIGS.values() if c.family == "bv_padded"] + list(
            SCALING_CONFIGS
        )
    raise ValueError(f"Unknown scope: {scope}")


def run_baseline(
    output_dir: Path,
    *,
    scope: str = "smoke",
    profiles: Optional[List[str]] = None,
    shots: int = 2048,
    num_runs: int = 3,
    seed: int = 20261022,
) -> Dict:
    """Run the small tracked baseline and export results for comparison with hardware later."""
    profiles = list(PROFILES) if profiles is None else profiles
    if num_runs < 1 or shots < 1 or seed < 0:
        raise ValueError("Shots/repetitions must be positive and seed must be nonnegative")
    if (
        not profiles
        or len(set(profiles)) != len(profiles)
        or any(p not in PROFILES for p in profiles)
    ):
        raise ValueError("Select distinct known noise profiles")
    output_dir.mkdir(parents=True, exist_ok=False)
    runner = BVCalibrationSimulatorExperiment(
        shots=shots, timeout=600, output_dir=output_dir / "raw"
    )
    configs = select_configs(scope)
    # Keep the original five-qubit lines unless a wider circuit needs more.
    all_configs = {**CONFIGS, **SCALING_CONFIGS}
    line_length = max(5, *(all_configs[c].num_total_qubits for c in configs))
    rng = np.random.default_rng(seed)
    rng.shuffle(configs)
    manifest = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": _now(),
        "campaign_id": output_dir.name,
        "execution_type": "NOISY_SIMULATOR",
        "scope": scope,
        "synthetic_line_length": line_length,
        "config_ids": configs,
        "profiles": [asdict(PROFILES[p]) for p in profiles],
        "shots_per_execution": shots,
        "num_runs": num_runs,
        "seed": seed,
        "planned_executions": len(configs) * len(profiles) * num_runs,
        "planned_shots": len(configs) * len(profiles) * num_runs * shots,
        "records": [],
    }
    _write(output_dir / "manifest.json", manifest)
    for profile in profiles:
        backend = synthetic_backend(PROFILES[profile], line_length)
        noise = NoiseModel.from_backend(backend, thermal_relaxation=False)
        noise_path = output_dir / f"noise_model_{profile.lower()}.json"
        _write(noise_path, noise.to_dict())
        for index, config_id in enumerate(configs):
            print(f"{profile} {index+1}/{len(configs)}: {config_id}", flush=True)
            manifest["records"].append(
                runner.run(
                    config_id,
                    profile=profile,
                    num_runs=num_runs,
                    seed=seed,
                    campaign_id=output_dir.name,
                    planned_executions=len(configs) * num_runs,
                    noise_model_path=noise_path,
                    backend=backend,
                    noise_model=noise,
                )
            )
            _write(output_dir / "manifest.json", manifest)
    manifest.update(status="completed", finished_at=_now())
    manifest["summary"] = export(output_dir / "raw", output_dir / "analysis")
    _write(output_dir / "manifest.json", manifest)
    from qward.examples.papers.bv.bv_calibration_baseline_report import generate_report

    generate_report(output_dir)
    manifest["report"] = "analysis/baseline_report.md"
    _write(output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["smoke", "core", "scaling"], default="smoke")
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir
        or DEFAULT_ROOT / f"baseline_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    manifest = run_baseline(
        output,
        scope=args.scope,
        profiles=args.profiles,
        shots=args.shots,
        num_runs=args.runs,
        seed=args.seed,
    )
    print(f"Saved {manifest['summary']['executions']} executions to {output}")


if __name__ == "__main__":
    main()
