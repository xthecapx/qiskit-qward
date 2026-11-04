"""Tracked BV calibration-prediction campaign; submission is an explicit CLI action.

Use --list or --preflight-only to inspect configurations without executing them.
Records use the same batch_summary/individual_results format as BVIBMExperiment.
Prepared predictions, snapshots and QPY artifacts survive timeouts and recovery.
"""

import argparse
import hashlib
import sys
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from qiskit import qpy
from qiskit.transpiler import generate_preset_pass_manager

from qward.algorithms.executor import IBMBatchResult, IBMJobResult
from qward.examples.papers.bv.bv_ibm import BVIBMExperiment
from qward.examples.papers.bv.bv_calibration_configs import (
    CONFIGS,
    BVCalibrationConfig,
    analytic_references,
    build_circuit,
    get_config,
)
from qward.examples.papers.bv.bv_calibration_analysis import CLAIMS_PER_RUN
from qward.examples.papers.ibm_campaign import json_value as _json_value
from qward.examples.papers.ibm_campaign import now as _now
from qward.examples.papers.ibm_campaign import write_json as _write
from qward.examples.papers.experiment_helpers import calculate_qward_metrics
from qward.examples.papers.ibm_experiment_base import resolve_ibm_credentials
from qward.metrics.circuit_error_prediction import (
    CircuitErrorPrediction,
    capture_calibration,
    circuit_hash,
    record_hash,
    snapshot_at_properties,
)
from qward.metrics.fidelity_metrics import FidelityMetrics
from qward.metrics.histogram_assessment import HistogramAssessment
from qward.schemas.error_prediction_schema import ErrorPredictionSchema

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "qpu" / "calibration_prediction" / "raw"


def summarize_record(journal: Dict[str, Any]) -> Dict[str, Any]:
    """Compact CLI view of a record; the JSON file keeps everything."""
    prediction = journal.get("prediction") or {}

    def error(key):
        value = prediction.get(key)
        return 1 - value if value is not None else None

    jobs = []
    for entry in journal.get("submission_journal", []):
        late = entry.get("execution_time_prediction") or {}
        result = next(
            (r for r in journal.get("individual_results", []) if r["job_id"] == entry["job_id"]),
            None,
        )
        assessment = (result or {}).get("histogram_assessment") or {}
        jobs.append(
            {
                "run_index": entry["run_index"],
                "job_id": entry["job_id"],
                "status": entry["status"],
                "error": entry.get("error"),
                "recalibrated_before_submission": entry.get("recalibrated_before_submission"),
                "execution_calibrated_at": entry.get("execution_calibrated_at"),
                "execution_time_predicted_error_m1": (
                    1 - late["m1_success"] if late.get("m1_success") is not None else None
                ),
                "execution_time_predicted_error_m2": (
                    1 - late["m2_success"] if late.get("m2_success") is not None else None
                ),
                "observed_error": assessment.get("tvd"),
                "observed_error_interval": assessment.get("tvd_interval"),
                "m1_range_status": (assessment.get("error_budget") or {}).get("range_status"),
                "m2_range_status": (assessment.get("m2_error_budget") or {}).get("range_status"),
                "rank_status": assessment.get("rank_status"),
            }
        )
    return {
        "record_path": journal.get("record_path"),
        "config_id": journal.get("config_id"),
        "status": journal.get("status"),
        "backend_name": journal.get("backend_name"),
        "frozen_snapshot_hash": prediction.get("snapshot_hash"),
        "frozen_predicted_error_m1": error("m1_success"),
        "frozen_predicted_error_m1_mixture": prediction.get("mixture_tvd"),
        "frozen_predicted_error_m2": error("m2_success"),
        "jobs": jobs,
        "next_step": (
            "--recover RECORD_JSON once IBM shows the job finished"
            if journal.get("status") in ("submitted", "incomplete")
            else None
        ),
    }


class BVCalibrationIBMExperiment(BVIBMExperiment):
    """BV runner using a frozen ISA circuit and durable per-submission journal."""

    execution_type = "IBM_QPU"
    backend_type = "qpu"
    noise_id = "IBM-QPU"

    def __init__(self, shots: int = 2048, timeout: int = 600, output_dir: Optional[Path] = None):
        super().__init__(shots=shots, timeout=timeout)
        if shots < 1 or timeout < 1:
            raise ValueError("shots and timeout must be positive")
        self.output_subdir = str(output_dir or DEFAULT_OUTPUT)

    @property
    def algorithm_name(self) -> str:
        return "BV-CALIBRATION-PREDICTION"

    def get_config(self, config_id: str) -> BVCalibrationConfig:
        return get_config(config_id)

    def get_all_config_ids(self) -> List[str]:
        return list(CONFIGS)

    def create_circuit(self, config):
        return build_circuit(config)

    def get_config_description(self, config):
        return config.to_dict()

    def create_success_criteria(self, config):
        nominal, _ = analytic_references(config)
        return lambda outcome: outcome.replace(" ", "") in nominal

    def exact_success_outcomes(self, config):
        # Injected deviations and controls can separate task success from the ideal output.
        return None

    def get_random_chance(self, config):
        nominal, _ = analytic_references(config)
        return len(nominal) / 2**config.num_qubits

    def evaluate_result(self, counts, config, total_shots):
        del total_shots
        nominal, _ = analytic_references(config)
        circuit = self.create_circuit(config)
        return (
            FidelityMetrics(
                circuit,
                counts=counts,
                expected_outcomes=list(nominal),
                target_histogram=nominal,
                num_measured_qubits=config.num_qubits,
            )
            .get_metrics()
            .model_dump(exclude_none=True)
        )

    def prepare_batch(self, plan_path: Path, service: Any) -> Dict[str, Any]:
        """Prepare every record of a plan from current calibration; submit nothing.

        Plan JSON: {"backend": str, "campaign_id": str, "window": int (default 1),
        "planned_executions": int,
        "items": [{"config": str, "layout": [int], "runs": int, "twirl_gates": bool}]}.
        Prepare each window just before submitting it, so its calibration is current.
        Writes ``<plan stem>_manifest.json`` beside the plan, listing each record.
        """
        plan = json.loads(plan_path.read_text())
        manifest_path = plan_path.with_name(f"{plan_path.stem}_manifest.json")
        if manifest_path.exists():
            raise ValueError(f"Manifest already exists: {manifest_path}")
        runs = [item.get("runs", 1) for item in plan["items"]]
        if not plan["items"] or any(
            isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in runs
        ):
            raise ValueError("Plan needs items, each with a positive integer 'runs'")
        if sum(runs) > plan["planned_executions"]:
            raise ValueError("Plan runs exceed its planned_executions budget")
        for item in plan["items"]:
            self.get_config(item["config"])  # Unknown configs fail before any record.
        backend = service.backend(plan["backend"])
        backend.refresh()
        items = []
        for item in plan["items"]:
            path = self.prepare(
                item["config"],
                backend,
                initial_layout=item["layout"],
                window=plan.get("window", 1),
                campaign_id=plan["campaign_id"],
                planned_executions=plan["planned_executions"],
                twirl_gates=item.get("twirl_gates", False),
            )
            _, prepared, _ = self._load(path)
            prediction = prepared["prediction"]
            items.append(
                {
                    "label": item.get("label", item["config"]),
                    "config": item["config"],
                    "layout": item["layout"],
                    "arm": prepared["arm"],
                    "runs": item.get("runs", 1),
                    "record_path": str(path),
                    "calibrated_at": prepared["calibration_snapshot"]["calibrated_at"],
                    "calibration_coverage": prediction["calibration_coverage"],
                    "predicted_error_m1": 1 - prediction["m1_success"],
                    "predicted_error_m1_mixture": prediction["mixture_tvd"],
                    "predicted_error_m2": 1 - prediction["m2_success"],
                }
            )
        manifest = {
            "schema_version": "1.0",
            "status": "prepared",
            "prepared_at": _now(),
            "plan": str(plan_path),
            "backend_name": backend.name,
            "campaign_id": plan["campaign_id"],
            "window": plan.get("window", 1),
            "batch_id": None,
            "jobs_planned": sum(i["runs"] for i in items),
            "items": items,
        }
        _write(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path)}

    def rank_layouts(
        self, config_id: str, backend: Any, *, top: int = 5, seed_transpiler: int = 11
    ) -> Dict[str, Any]:
        """Rank candidate physical lines by predicted error from current calibration.

        Candidates are simple paths of the coupling graph with the circuit's
        width. A stratified sample is compiled exactly as ``prepare`` would and
        scored by M1 and M2. Nothing is archived or submitted. The low label
        takes the best lines; the contrasting label takes lines nearest twice
        the best predicted error, excluding lines with a near-broken instruction
        (error >= 0.5), which would only produce uniform noise. Save this output
        with the campaign.
        """
        config = self.get_config(config_id)
        logical = self.create_circuit(config)
        nominal, injected = analytic_references(config)
        width = logical.num_qubits
        graph = backend.coupling_map.graph.to_undirected(multigraph=False)
        properties = (
            backend.properties() if callable(getattr(backend, "properties", None)) else None
        )
        faulty = set(properties.faulty_qubits()) if properties is not None else set()
        paths = set()
        for start in range(backend.num_qubits):
            stack = [[start]]
            while stack:
                path = stack.pop()
                if len(path) == width:
                    paths.add(tuple(path) if path[0] < path[-1] else tuple(reversed(path)))
                    continue
                for neighbor in graph.neighbors(path[-1]):
                    if neighbor not in path and neighbor not in faulty:
                        stack.append(path + [neighbor])
        if not paths:
            raise ValueError("No connected physical line of the required width")
        # Cheap pre-score using per-edge/readout errors, then compile the best few.
        target = backend.target

        def rough(path):
            two_qubit = next(n for n in ("cz", "ecr", "cx") if n in target.operation_names)
            score = sum(
                (target[two_qubit].get((a, b)) or target[two_qubit].get((b, a))).error or 0
                for a, b in zip(path, path[1:])
            )
            return score + sum(target["measure"][(q,)].error or 0 for q in path)

        shortlist = sorted(paths, key=rough)
        step = max(1, len(shortlist) // (8 * top))
        candidates = list(dict.fromkeys(shortlist[: 4 * top] + shortlist[::step]))
        ranked = []
        for path in candidates:
            # BV ancilla is the last logical qubit; place it at the end of the line.
            layout = list(path)
            isa = generate_preset_pass_manager(
                backend=backend,
                optimization_level=0,
                initial_layout=layout,
                seed_transpiler=seed_transpiler,
                scheduling_method="alap",
            ).run(logical)
            snapshot = capture_calibration(isa, backend)
            prediction = CircuitErrorPrediction(
                isa, snapshot=snapshot, reference=injected
            ).get_metrics()
            snapshot_errors = [e["error"] for e in snapshot["instructions"]]
            if prediction.m1_success is None or any(
                e is not None and e >= 0.5 for e in snapshot_errors
            ):
                continue
            ranked.append(
                {
                    "layout": layout,
                    "m1_error": 1 - prediction.m1_success,
                    "m1_mixture_error": prediction.mixture_tvd,
                    "m2_error": 1 - prediction.m2_success,
                    "two_qubit_gates": prediction.features["native_two_qubit_gates"],
                }
            )
        if not ranked:
            raise ValueError("No fully calibrated physical line of the required width")
        ranked.sort(key=lambda r: r["m1_error"])
        goal = min(2 * ranked[0]["m1_error"], 0.9)
        contrasting = sorted(ranked[top:] or ranked, key=lambda r: abs(r["m1_error"] - goal))
        return {
            "config_id": config_id,
            "backend_name": backend.name,
            "ranked_at": _now(),
            "calibrated_at": str(properties.last_update_date) if properties else None,
            "candidate_lines": len(paths),
            "low_predicted_error": ranked[:top],
            "contrasting_predicted_error": contrasting[:top],
            "contrasting_goal_m1_error": goal,
            "nominal_reference": nominal,
        }

    def preflight(self, config_id: str) -> Dict[str, Any]:
        """Build and inspect a circuit only; no authentication or execution."""
        config = self.get_config(config_id)
        circuit = self.create_circuit(config)
        nominal, injected = analytic_references(config)
        return {
            "config": config.to_dict(),
            "circuit_hash": circuit_hash(circuit),
            "circuit_depth": circuit.depth(),
            "total_gates": circuit.size(),
            "count_ops": dict(circuit.count_ops()),
            "nominal_reference": nominal,
            "injected_reference": injected,
            "shots": self.shots,
            "status": "offline_preflight",
            "jobs_submitted": 0,
        }

    def campaign_plan(self, *, smoke_test: bool = False, seed: int = 11) -> Dict[str, Any]:
        """Return a reproducible randomized schedule; no circuits are executed."""
        configs = [
            config.config_id
            for config in CONFIGS.values()
            if not smoke_test
            or (config.num_qubits == 2 and config.layout_label == "low_predicted_error")
        ]
        rng = np.random.default_rng(seed)
        windows = []
        for window in range(1, 2 if smoke_test else 4):
            order = list(configs)
            rng.shuffle(order)
            windows.append({"window": window, "config_ids_in_execution_order": order})
        executions = len(configs) * len(windows)
        return {
            "schema_version": "1.0",
            "status": "schedule_only",
            "randomization_seed": seed,
            "numpy_version": np.__version__,
            "planned_executions_per_window": len(configs),
            "windows": windows,
            "executions": executions,
            "shots": executions * self.shots,
            "required_inputs": "Choose backend and physical allocation for each width/layout label before submission",
        }

    def prepare(
        self,
        config_id: str,
        backend: Any,
        *,
        initial_layout: List[int],
        window: int = 1,
        campaign_id: str = "bv-calibration-pilot",
        optimization_level: int = 0,
        seed_transpiler: int = 11,
        planned_executions: int = 100,
        execution_context: Optional[Dict[str, Any]] = None,
        twirl_gates: bool = False,
    ) -> Path:
        """Save exact circuit/calibration/prediction before any submission.

        Caller selects the physical layout using pre-run information. This method
        accepts local BackendV2 fixtures for testing; it never executes a circuit.
        """
        config = self.get_config(config_id)
        logical = self.create_circuit(config)
        if window < 1 or planned_executions < 1 or not 0 <= optimization_level <= 3:
            raise ValueError("Invalid window, planned executions or optimization level")
        if len(initial_layout) != logical.num_qubits or len(set(initial_layout)) != len(
            initial_layout
        ):
            raise ValueError("Layout must assign every logical qubit, including the BV ancilla")
        if any(
            isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < backend.num_qubits
            for q in initial_layout
        ):
            raise ValueError("Layout contains invalid physical qubit indices")
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            initial_layout=initial_layout,
            seed_transpiler=seed_transpiler,
            scheduling_method="alap",
        )
        isa = pm.run(logical)
        if config.family == "bv_padded" and config.forward_layers:
            baseline_config = BVCalibrationConfig(**{**config.__dict__, "forward_layers": 0})
            baseline = pm.run(self.create_circuit(baseline_config))

            def two_qubit_count(circuit):
                return sum(
                    len(item.qubits) == 2 and item.operation.name != "barrier"
                    for item in circuit.data
                )

            if two_qubit_count(isa) <= two_qubit_count(baseline):
                raise ValueError("Transpilation removed the two-qubit exposure treatment")
        if callable(getattr(backend, "check_faulty", None)):
            backend.check_faulty(isa)  # Raises before anything is archived.
        snapshot = capture_calibration(isa, backend)
        nominal, injected = analytic_references(config)
        prediction = CircuitErrorPrediction(
            isa, snapshot=snapshot, reference=injected
        ).get_metrics()
        run_id = uuid4().hex
        provider_label = "IBM" if self.execution_type == "IBM_QPU" else "SIMULATOR"
        stem = f"{config_id}_{provider_label}_{backend.name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{run_id[:8]}"
        root = self.get_output_dir()
        root.mkdir(parents=True, exist_ok=True)
        artifacts = root / "artifacts" / stem
        artifacts.mkdir(parents=True, exist_ok=False)
        artifact_hashes = {}
        for name, circuit in (("logical.qpy", logical), ("isa.qpy", isa)):
            path = artifacts / name
            with path.open("xb") as stream:
                qpy.dump(circuit, stream)
            artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        versions = {}
        for name in (
            "qiskit",
            "qiskit-aer",
            "qiskit-ibm-runtime",
            "qiskit-qward",
            "numpy",
            "scipy",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unknown"
        prepared = {
            "schema_version": "1.0",
            "execution_type": self.execution_type,
            "execution_context": execution_context or {},
            "run_id": run_id,
            "prepared_at": _now(),
            "campaign_id": campaign_id,
            "window": window,
            "config": config.to_dict(),
            "backend_name": backend.name,
            "shots": self.shots,
            "versions": versions,
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "initial_layout": initial_layout,
            "layout_label": config.layout_label,
            "scheduling_method": "alap",
            "artifact_hashes": artifact_hashes,
            "logical_circuit_hash": circuit_hash(logical),
            "isa_circuit_hash": circuit_hash(isa),
            "output_register": isa.cregs[0].name,
            "physical_measurements": [
                e for e in snapshot["instructions"] if e["name"] == "measure"
            ],
            "nominal_reference": nominal,
            "injected_reference": injected,
            "calibration_snapshot": snapshot,
            "prediction": prediction.model_dump(),
            "qward_metrics": calculate_qward_metrics(logical),
            # Bonferroni over every per-run claim, shared with offline analysis.
            "assessment_alpha": 0.05 / (CLAIMS_PER_RUN * planned_executions),
            "claims_per_run": CLAIMS_PER_RUN,
            "planned_executions_per_window": planned_executions,
            "runtime_options": (
                {
                    # Gate twirling is a separate diagnostic arm: it converts
                    # coherent gate errors into stochastic Pauli errors.
                    "twirling": {"enable_gates": twirl_gates, "enable_measure": False},
                    "dynamical_decoupling": {"enable": False},
                    "execution": {"init_qubits": True},
                }
                if self.execution_type == "IBM_QPU"
                else None
            ),
            "arm": "gate_twirling" if twirl_gates else "baseline",
        }
        _write(artifacts / "prepared.json", prepared)
        journal = {
            "schema_version": "1.0",
            "config_id": config_id,
            "algorithm": self.algorithm_name,
            "execution_type": self.execution_type,
            "noise_id": self.noise_id,
            "backend_name": backend.name,
            "campaign_id": campaign_id,
            "window": window,
            "run_id": run_id,
            "saved_at": _now(),
            "status": "prepared",
            "batch_id": None,
            "config": config.to_dict(),
            "arm": prepared["arm"],
            "batch_summary": {},
            "individual_results": [],
            "submission_journal": [],
            "artifact_directory": str(artifacts.relative_to(root)),
            "prepared_hash": record_hash(prepared),
        }
        record_path = root / f"{stem}.json"
        _write(record_path, journal)
        return record_path

    @staticmethod
    def _load(record_path: Path):
        journal = json.loads(record_path.read_text())
        artifacts = record_path.parent / journal["artifact_directory"]
        prepared = json.loads((artifacts / "prepared.json").read_text())
        if record_hash(prepared) != journal["prepared_hash"]:
            raise ValueError("Prepared prediction record failed integrity verification")
        circuits = {}
        for name, digest in prepared["artifact_hashes"].items():
            path = artifacts / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Circuit artifact changed: {name}")
            with path.open("rb") as stream:
                circuits[name] = qpy.load(stream)[0]
        if circuit_hash(circuits["isa.qpy"]) != prepared["isa_circuit_hash"]:
            raise ValueError("ISA circuit semantics failed integrity verification")
        return journal, prepared, circuits

    def submit_prepared(
        self,
        record_path: Path,
        backend: Any,
        *,
        num_runs: int = 1,
        calibration_check: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Submit the archived ISA circuit and journal IDs immediately for recovery.

        Job mode: each repetition is one independent Runtime job on ``backend``.
        No Batch/Session is opened; one circuit per job needs no grouping.
        """
        from qiskit_ibm_runtime import SamplerV2

        journal, prepared, circuits = self._check_submittable(record_path, backend, num_runs)
        return self._submit_jobs(
            record_path,
            (journal, prepared, circuits),
            SamplerV2(mode=backend, options=prepared["runtime_options"]),
            num_runs=num_runs,
            calibration_check=calibration_check,
        )

    def _check_submittable(self, record_path: Path, backend: Any, num_runs: int):
        """Validate a fresh record, backend match and window run budget."""
        journal, prepared, circuits = self._load(record_path)
        if journal["status"] != "prepared" or num_runs < 1:
            raise ValueError(f"Only a fresh prepared record can be submitted: {record_path}")
        if num_runs > prepared["planned_executions_per_window"]:
            raise ValueError("Repetitions exceed the prespecified multiplicity allocation")
        if backend.name != prepared["backend_name"]:
            raise ValueError("Submission backend differs from the calibration snapshot")
        allocated = self._submitted_runs(record_path.parent).get(
            (prepared["campaign_id"], prepared["window"]), 0
        )
        if allocated + num_runs > prepared["planned_executions_per_window"]:
            raise ValueError("Campaign window exceeds its prespecified execution/alpha budget")
        return journal, prepared, circuits

    def _submitted_runs(self, directory: Path) -> Dict[Any, int]:
        """Jobs already journaled per (campaign_id, window) in a records directory."""
        counts: Dict[Any, int] = {}
        for path in directory.glob("*.json"):
            other = json.loads(path.read_text())
            if other.get("algorithm") == self.algorithm_name:
                key = (other.get("campaign_id"), other.get("window"))
                counts[key] = counts.get(key, 0) + len(other.get("submission_journal", []))
        return counts

    @staticmethod
    def _submit_jobs(record_path, loaded, sampler, *, num_runs, calibration_check) -> Dict:
        journal, prepared, circuits = loaded
        journal["status"] = "submitting"
        _write(record_path, journal)
        for run_index in range(num_runs):
            # Persist intent before sending. A crash in this interval requires
            # reconciliation via job tags; recovery never resubmits.
            sampler.options.environment.job_tags = [prepared["run_id"], f"repeat-{run_index}"]
            entry = {
                "run_index": run_index,
                "status": "submitting",
                "submitted_at": _now(),
                "job_id": None,
                **(calibration_check or {}),
            }
            journal["submission_journal"].append(entry)
            _write(record_path, journal)
            try:
                job = sampler.run([circuits["isa.qpy"]], shots=prepared["shots"])
                entry.update(job_id=job.job_id(), status="submitted")
                entry["effective_runtime_options"] = asdict(sampler.options)
            except Exception as exc:
                entry.update(status="submission_unknown", error=str(exc))
                journal["status"] = "submission_unknown"
                _write(record_path, journal)
                raise
            _write(record_path, journal)
        journal["status"] = "submitted"
        _write(record_path, journal)
        return {**journal, "prediction": prepared["prediction"], "record_path": str(record_path)}

    def submit_batch(self, manifest_path: Path, backend: Any) -> Dict[str, Any]:
        """Submit every record in a batch manifest inside one Runtime Batch.

        One Batch groups the jobs for scheduling and organization; each record
        still gets its own job(s), journal and frozen prediction. All records
        are validated before any job is sent. The Batch ID is written to the
        manifest and to every record.
        """
        from qiskit_ibm_runtime import Batch, SamplerV2

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "prepared":
            raise ValueError("Only a fresh prepared batch manifest can be submitted")
        items = [
            (
                Path(item["record_path"]),
                item["runs"],
                *self._check_submittable(Path(item["record_path"]), backend, item["runs"]),
            )
            for item in manifest["items"]
        ]
        # Per-record checks see prior submissions; also bound the batch as a whole.
        prior = self._submitted_runs(items[0][0].parent)
        total: Dict[Any, int] = {}
        limit: Dict[Any, int] = {}
        for _, runs, _, prepared, _ in items:
            key = (prepared["campaign_id"], prepared["window"])
            total[key] = total.get(key, 0) + runs
            limit[key] = min(limit.get(key, 10**9), prepared["planned_executions_per_window"])
        for key, runs in total.items():
            if prior.get(key, 0) + runs > limit[key]:
                raise ValueError(f"Batch exceeds run budget of campaign/window {key}")
        current_date = self._current_calibration_date(backend)
        manifest["status"] = "submitting"
        _write(manifest_path, manifest)
        with Batch(backend=backend) as batch:
            manifest["batch_id"] = batch.session_id
            _write(manifest_path, manifest)
            for (path, runs, journal, prepared, circuits), item in zip(items, manifest["items"]):
                journal["batch_id"] = batch.session_id
                check = self._calibration_at_submission(prepared, backend, current_date)
                sampler = SamplerV2(mode=batch, options=prepared["runtime_options"])
                result = self._submit_jobs(
                    path,
                    (journal, prepared, circuits),
                    sampler,
                    num_runs=runs,
                    calibration_check=check,
                )
                item["job_ids"] = [e["job_id"] for e in result["submission_journal"]]
                _write(manifest_path, manifest)
        manifest.update(status="submitted", submitted_at=_now())
        _write(manifest_path, manifest)
        return manifest

    def recover_manifest(self, manifest_path: Path, service: Any) -> Dict[str, Any]:
        """Collect every record in a manifest; never resubmits."""
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] == "prepared" or not manifest.get("batch_id"):
            raise ValueError("Batch was never submitted; run --submit-batch first")
        summaries = []
        for item in manifest["items"]:
            record = self.collect(Path(item["record_path"]), service)
            summaries.append(summarize_record(record))
        done = all(s["status"] == "completed" for s in summaries)
        manifest["status"] = "completed" if done else "incomplete"
        _write(manifest_path, manifest)
        return {"manifest": str(manifest_path), "status": manifest["status"], "records": summaries}

    def collect(self, record_path: Path, service: Any) -> Dict:
        """Retrieve only saved job IDs; retain the original prediction on timeout/recovery."""
        journal, prepared, circuits = self._load(record_path)
        if journal["status"] == "prepared" and not journal["submission_journal"]:
            # Never submitted: leave the record untouched so it stays submittable.
            raise ValueError(f"Record was never submitted; nothing to recover: {record_path}")
        self.get_config(prepared["config"]["config_id"])  # Validate before retrieving jobs.
        self.shots = prepared["shots"]
        jobs, incomplete = [], not bool(journal["submission_journal"])
        previous = {r["job_id"]: r for r in journal["individual_results"]}
        for entry in journal["submission_journal"]:
            if not entry["job_id"]:
                incomplete = True
                continue
            try:
                job = service.job(entry["job_id"])
                result = job.result(timeout=self.timeout)
                counts = getattr(result[0].data, prepared["output_register"]).get_counts()
                jobs.append(
                    IBMJobResult(
                        job_id=entry["job_id"],
                        optimization_level=prepared["optimization_level"],
                        status="completed",
                        counts=counts,
                        circuit_depth=circuits["isa.qpy"].depth(),
                        run_index=entry["run_index"],
                    )
                )
                entry.update(
                    status="completed", collected_at=_now(), pub_metadata=result[0].metadata
                )
                try:
                    entry["execution_metrics"] = job.metrics()
                except Exception as exc:
                    entry["execution_metrics_error"] = str(exc)
                entry.update(self._execution_time_prediction(job, prepared, circuits))
            except Exception as exc:
                incomplete = True
                entry.update(status="retrieval_failed", error=str(exc))
                # A temporary retrieval failure must not erase previously saved counts.
                if entry["job_id"] in previous:
                    old = previous[entry["job_id"]]
                    jobs.append(
                        IBMJobResult(
                            job_id=entry["job_id"],
                            optimization_level=prepared["optimization_level"],
                            status=old["status"],
                            counts=old["counts"],
                            circuit_depth=old["transpiled_depth"],
                            run_index=entry["run_index"],
                        )
                    )
        batch_result = IBMBatchResult(
            # Older records used a Batch; job-mode records have no batch ID.
            batch_id=journal.get("batch_id") or f"job-mode-{prepared['run_id']}",
            backend_name=prepared["backend_name"],
            status="incomplete" if incomplete else "completed",
            jobs=jobs,
        )
        return self.finish_record(record_path, journal, prepared, circuits, batch_result)

    @staticmethod
    def _execution_time_prediction(job: Any, prepared: Dict, circuits: Dict) -> Dict[str, Any]:
        """Diagnostic re-prediction from calibration in effect when the job ran.

        IBM recalibrates between submission and execution. Comparing the frozen
        prediction, this one, and the counts separates calibration drift from
        model error. It never replaces the frozen prediction.
        """
        # Diagnostic only: any failure here must not make retrieved counts incomplete.
        try:
            properties = job.properties()
            if properties is None:
                return {"execution_calibration_error": "job.properties() returned None"}
            snapshot = snapshot_at_properties(prepared["calibration_snapshot"], properties)
            prediction = CircuitErrorPrediction(
                circuits["isa.qpy"], snapshot=snapshot, reference=prepared["injected_reference"]
            ).get_metrics()
        except Exception as exc:  # pylint: disable=broad-except
            return {"execution_calibration_error": str(exc)}
        return {
            "execution_calibrated_at": snapshot["calibrated_at"],
            "execution_calibration_changed": (
                snapshot["calibrated_at"] != prepared["calibration_snapshot"]["calibrated_at"]
            ),
            "execution_time_prediction": prediction.model_dump(
                exclude={"mixture_probabilities", "m2_probabilities", "assumptions"}
            ),
        }

    def finish_record(self, record_path, journal, prepared, circuits, batch_result):
        """Share rich result construction between IBM and local simulator runners."""
        config = self.get_config(prepared["config"]["config_id"])
        self.shots = prepared["shots"]
        logical = circuits["logical.qpy"]
        result = self._build_rich_result(
            batch_result,
            config,
            logical,
            prepared["qward_metrics"],
            logical.depth(),
            logical.size(),
        )
        result["batch_summary"].pop("quantum_advantage_demonstrated", None)
        result["batch_summary"].pop("mean_quantum_advantage_ratio", None)
        result.update(execution_type=self.execution_type, noise_id=self.noise_id)
        result["batch_summary"].update(backend_type=self.backend_type, noise_model=self.noise_id)
        prediction = ErrorPredictionSchema.model_validate(prepared["prediction"])
        for item in result["individual_results"]:
            item.update(
                campaign_id=prepared["campaign_id"],
                window=prepared["window"],
                run_id=prepared["run_id"],
                experiment_id=f"{prepared['run_id']}-{item['run_index']:03d}",
                prediction=prepared["prediction"],
                backend_type=self.backend_type,
                noise_model=self.noise_id,
                execution_type=self.execution_type,
                histogram_assessment=HistogramAssessment(
                    circuits["isa.qpy"],
                    counts=item["counts"],
                    reference=prepared["nominal_reference"],
                    prediction=prediction,
                    alpha=prepared["assessment_alpha"],
                )
                .get_metrics()
                .model_dump(),
            )
        journal.update(result)
        journal.update(
            saved_at=_now(),
            prediction=prepared["prediction"],
            calibration_snapshot=prepared["calibration_snapshot"],
        )
        _write(record_path, journal)
        return {**journal, "record_path": str(record_path)}

    def run(
        self,
        config_id,
        backend_name=None,
        optimization_levels=None,
        num_runs=1,
        save_results=True,
        channel=None,
        token=None,
        instance=None,
        *,
        initial_layout=None,
        window=1,
        campaign_id="bv-calibration-pilot",
        planned_executions=100,
    ):
        """Explicit hardware entry point; archive and submit exactly one compiler configuration."""
        from qiskit_ibm_runtime import QiskitRuntimeService

        levels = [0] if optimization_levels is None else optimization_levels
        if not backend_name or initial_layout is None or len(levels) != 1 or not save_results:
            raise ValueError("Require backend, physical layout, one opt level, and result saving")
        credentials = resolve_ibm_credentials(channel, token, instance)
        service = QiskitRuntimeService(**{k: v for k, v in credentials.items() if v is not None})
        backend = service.backend(backend_name)
        backend.refresh()
        path = self.prepare(
            config_id,
            backend,
            initial_layout=initial_layout,
            window=window,
            campaign_id=campaign_id,
            optimization_level=levels[0],
            planned_executions=planned_executions,
        )
        print(f"Prepared prediction and recovery record: {path}")
        check = self._calibration_at_submission(self._load(path)[1], backend)
        self.submit_prepared(path, backend, num_runs=num_runs, calibration_check=check)
        return self.collect(path, service)

    @staticmethod
    def _service():
        from qiskit_ibm_runtime import QiskitRuntimeService

        credentials = resolve_ibm_credentials()
        return QiskitRuntimeService(**{k: v for k, v in credentials.items() if v is not None})

    @staticmethod
    def _current_calibration_date(backend: Any) -> Optional[str]:
        try:
            current = backend.properties(refresh=True)
        except TypeError:
            current = backend.properties()
        return str(current.last_update_date) if current is not None else None

    def _calibration_at_submission(
        self, prepared: Dict[str, Any], backend: Any, current_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare the frozen calibration date with the backend's at submission.

        Queues can outlast a calibration cycle, so drift may also happen after
        submission; refusing here would not prevent it. The flag is recorded and
        ``collect`` re-predicts from the calibration the job actually ran under.
        """
        frozen = prepared["calibration_snapshot"]["calibrated_at"]
        if current_date is None:
            current_date = self._current_calibration_date(backend)
        changed = frozen is not None and current_date is not None and frozen != current_date
        if changed:
            print(
                f"WARNING: backend recalibrated since preparation ({frozen} -> {current_date}). "
                "Submitting anyway; drift is measured from job.properties() at recovery.",
                file=sys.stderr,
            )
        return {
            "calibrated_at_submission": current_date,
            "recalibrated_before_submission": changed,
        }

    def create_argument_parser(self):
        parser = argparse.ArgumentParser(description=__doc__)
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--list", action="store_true")
        action.add_argument("--preflight-only", action="store_true")
        action.add_argument("--plan", action="store_true")
        action.add_argument("--rank-layouts", action="store_true")
        action.add_argument("--prepare-only", action="store_true")
        action.add_argument("--submit", action="store_true")
        action.add_argument("--submit-prepared", type=Path, metavar="RECORD_JSON")
        action.add_argument("--prepare-batch", type=Path, metavar="PLAN_JSON")
        action.add_argument("--submit-batch", type=Path, metavar="MANIFEST_JSON")
        action.add_argument("--recover-batch", type=Path, metavar="MANIFEST_JSON")
        action.add_argument("--recover", type=Path, metavar="RESULT_JSON")
        parser.add_argument("--config", choices=list(CONFIGS))
        parser.add_argument("--smoke-test", action="store_true")
        parser.add_argument("--plan-file", type=Path)
        parser.add_argument("--backend")
        parser.add_argument("--layout", type=int, nargs="+")
        parser.add_argument("--window", type=int, default=1)
        parser.add_argument("--campaign-id", default="bv-calibration-pilot")
        parser.add_argument("--planned-executions", type=int, default=100)
        parser.add_argument("--opt-levels", type=int, nargs=1, default=[0])
        parser.add_argument("--runs", type=int, default=1)
        parser.add_argument("--shots", type=int, default=2048)
        parser.add_argument("--timeout", type=int, default=600)
        parser.add_argument("--output-dir", type=Path)
        parser.add_argument(
            "--wait",
            action="store_true",
            help="With --submit-prepared, wait up to --timeout for results (default: exit)",
        )
        parser.add_argument(
            "--full-json", action="store_true", help="Print the whole record instead of a summary"
        )
        return parser

    def run_cli(self, args: Optional[List[str]] = None):
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)
        self.shots, self.timeout = parsed.shots, parsed.timeout
        if parsed.output_dir:
            self.output_subdir = str(parsed.output_dir)
        if parsed.list:
            result = {key: config.to_dict() for key, config in CONFIGS.items()}
        elif parsed.plan:
            result = self.campaign_plan(smoke_test=parsed.smoke_test)
            if parsed.plan_file:
                parsed.plan_file.parent.mkdir(parents=True, exist_ok=True)
                _write(parsed.plan_file, result)
        elif parsed.preflight_only:
            if not parsed.config:
                parser.error("--preflight-only requires --config")
            result = self.preflight(parsed.config)
        elif parsed.recover:
            result = self.collect(parsed.recover, self._service())
        elif parsed.rank_layouts:
            if not parsed.config or not parsed.backend:
                parser.error("--rank-layouts requires --config and --backend")
            result = self.rank_layouts(parsed.config, self._service().backend(parsed.backend))
        elif parsed.prepare_only:
            if not parsed.config or not parsed.backend or parsed.layout is None:
                parser.error("--prepare-only requires --config, --backend and --layout")
            path = self.prepare(
                parsed.config,
                self._service().backend(parsed.backend),
                initial_layout=parsed.layout,
                window=parsed.window,
                campaign_id=parsed.campaign_id,
                optimization_level=parsed.opt_levels[0],
                planned_executions=parsed.planned_executions,
            )
            _, prepared, _ = self._load(path)
            prediction = prepared["prediction"]
            result = {
                "record_path": str(path),
                "status": "prepared_not_submitted",
                "calibrated_at": prepared["calibration_snapshot"]["calibrated_at"],
                "calibration_coverage": prediction["calibration_coverage"],
                "predicted_error_m1": (
                    1 - prediction["m1_success"] if prediction["m1_success"] is not None else None
                ),
                "predicted_error_m1_mixture": prediction["mixture_tvd"],
                "predicted_error_m2": (
                    1 - prediction["m2_success"] if prediction["m2_success"] is not None else None
                ),
                "jobs_submitted": 0,
            }
        elif parsed.prepare_batch:
            result = self.prepare_batch(parsed.prepare_batch, self._service())
        elif parsed.submit_batch:
            manifest = json.loads(parsed.submit_batch.read_text())
            result = self.submit_batch(
                parsed.submit_batch, self._service().backend(manifest["backend_name"])
            )
        elif parsed.recover_batch:
            result = self.recover_manifest(parsed.recover_batch, self._service())
        elif parsed.submit_prepared:
            service = self._service()
            journal = json.loads(parsed.submit_prepared.read_text())
            backend = service.backend(journal["backend_name"])
            check = self._calibration_at_submission(self._load(parsed.submit_prepared)[1], backend)
            result = self.submit_prepared(
                parsed.submit_prepared, backend, num_runs=parsed.runs, calibration_check=check
            )
            if parsed.wait:
                result = self.collect(parsed.submit_prepared, service)
        else:
            if not parsed.config or not parsed.backend or parsed.layout is None:
                parser.error("--submit requires --config, --backend and --layout")
            result = self.run(
                parsed.config,
                parsed.backend,
                optimization_levels=parsed.opt_levels,
                num_runs=parsed.runs,
                initial_layout=parsed.layout,
                window=parsed.window,
                campaign_id=parsed.campaign_id,
                planned_executions=parsed.planned_executions,
            )
        is_record = isinstance(result, dict) and "submission_journal" in result
        shown = summarize_record(result) if is_record and not parsed.full_json else result
        print(json.dumps(shown, indent=2, allow_nan=False, default=_json_value))
        return result


if __name__ == "__main__":
    BVCalibrationIBMExperiment().run_cli()
