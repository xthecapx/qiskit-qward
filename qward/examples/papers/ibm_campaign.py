"""Shared IBM experiment lifecycle: prepare -> submit -> recover -> export.

Every QPU execution is backed by a durable *record* written before submission:

    <output_dir>/<stem>.json                 journal (status, job IDs, results)
    <output_dir>/artifacts/<stem>/
        logical.qpy, isa.qpy                 exact circuits (ISA is what runs)
        prepared.json                        frozen inputs, hashed

``prepared.json`` holds the configuration, transpiler settings, calibration
snapshot of the exact ISA circuit, ideal reference and any pre-run payload.
Its hash is stored in the journal; artifact hashes are stored in it. Records
are therefore verifiable and a submitted job can always be recovered by its
saved ID, without resubmitting and without rebuilding anything from newer
calibration.

Lifecycle states of a record:

    prepared -> submitting -> submitted -> completed
                           \\-> submission_unknown   (crash while sending)
                submitted   -> incomplete            (some jobs not retrievable yet)

A *plan* groups records for one algorithm and one calibration window; its
*manifest* lists the records and, once submitted, the Runtime Batch ID and job
IDs. Jobs of a plan are submitted in one Batch, ordered by a seeded shuffle with
run 1 of every config before run 2.

The lifecycle is algorithm-agnostic. ``CampaignHooks`` supplies what differs per
algorithm: circuit construction, ideal reference, record payloads and the
post-run evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import generate_preset_pass_manager

from qward.metrics.circuit_error_prediction import (
    capture_calibration,
    circuit_hash,
    is_dynamic,
    record_hash,
)

RECORD_SCHEMA_VERSION = "2.0"
MANIFEST_SCHEMA_VERSION = "2.0"
VERSIONED_PACKAGES = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "qiskit-qward", "numpy")
DEFAULT_RUNTIME_OPTIONS: Dict[str, Any] = {
    "twirling": {"enable_gates": False, "enable_measure": False},
    "dynamical_decoupling": {"enable": False},
    "execution": {"init_qubits": True},
}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def now() -> str:
    """UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    """Encode Runtime/numpy objects; fail rather than silently stringify unknowns."""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if type(value).__name__ == "UnsetType":
        return "server_default_unknown_at_submission"
    # Runtime metadata includes ExecutionSpans and other structured objects.
    from qiskit_ibm_runtime.utils.json import (
        RuntimeEncoder,
    )  # pylint: disable=import-outside-toplevel

    return RuntimeEncoder().default(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file; NaN/inf are rejected."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_value) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def package_versions(names: Sequence[str] = VERSIONED_PACKAGES) -> Dict[str, str]:
    """Installed versions of the packages that shape a record."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def merge_options(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recursive dict merge for Runtime option trees."""
    merged = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = merge_options(merged[key], value)
        else:
            merged[key] = value
    return merged


def extract_counts(pub_result: Any, register: Optional[str] = None) -> Dict[str, int]:
    """Counts from a SamplerV2 pub result.

    With several classical registers the counts are joined into space-separated
    labels in Qiskit's order (last register leftmost), matching ``get_counts()``
    of Aer and legacy records. ``register`` selects a single register instead.
    """
    data = pub_result.data
    if register is not None:
        return dict(getattr(data, register).get_counts())
    names = list(data.keys())
    if len(names) == 1:
        return dict(getattr(data, names[0]).get_counts())
    # Per-shot bitstrings of each register, joined last-register-first.
    per_register = [getattr(data, name).get_bitstrings() for name in reversed(names)]
    counts: Dict[str, int] = {}
    for parts in zip(*per_register):
        label = " ".join(parts)
        counts[label] = counts.get(label, 0) + 1
    return counts


def current_calibration_date(backend: Any) -> Optional[str]:
    """Backend's own calibration timestamp, refreshed when supported."""
    if not callable(getattr(backend, "properties", None)):
        return None
    try:
        props = backend.properties(refresh=True)
    except TypeError:
        props = backend.properties()
    return str(props.last_update_date) if props is not None else None


def service_from_env(channel=None, token=None, instance=None):
    """QiskitRuntimeService from explicit values or IBM_QUANTUM_* (.env loaded by caller)."""
    from qiskit_ibm_runtime import QiskitRuntimeService  # pylint: disable=import-outside-toplevel

    kwargs = {
        "channel": channel or os.getenv("IBM_QUANTUM_CHANNEL"),
        "token": token or os.getenv("IBM_QUANTUM_TOKEN"),
        "instance": instance or os.getenv("IBM_QUANTUM_INSTANCE"),
    }
    return QiskitRuntimeService(**{k: v for k, v in kwargs.items() if v is not None})


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@dataclass
class PreparedCircuit:
    """What a runner provides for one config before transpilation."""

    config_id: str
    config: Dict[str, Any]
    circuit: QuantumCircuit
    reference: Optional[Dict[str, float]] = None  # ideal output distribution
    payload: Dict[str, Any] = field(default_factory=dict)  # frozen pre-run extras
    runtime_options: Optional[Dict[str, Any]] = None  # merged onto defaults
    initial_layout: Optional[List[int]] = None


@dataclass
class CampaignHooks:
    """Per-algorithm behavior plugged into the generic lifecycle.

    ``build(config_id)`` returns a PreparedCircuit. ``before_freeze`` may add
    pre-run payload computed from the ISA circuit and calibration snapshot (for
    example metrics or predictions). ``evaluate`` turns one job's counts into a
    result row. ``after_job`` may add per-job diagnostics from the Runtime job
    (for example execution-time calibration).
    """

    algorithm: str
    build: Callable[[str], PreparedCircuit]
    evaluate: Callable[[Dict[str, int], Dict[str, Any], QuantumCircuit], Dict[str, Any]]
    before_freeze: Optional[
        Callable[[PreparedCircuit, QuantumCircuit, Dict[str, Any]], Dict[str, Any]]
    ] = None
    after_job: Optional[Callable[[Any, Dict[str, Any], Dict[str, QuantumCircuit]], Dict]] = None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class RecordStore:
    """Durable records for one algorithm under one output directory."""

    def __init__(self, output_dir: Path, hooks: CampaignHooks, *, timeout: int = 600):
        self.output_dir = Path(output_dir)
        self.hooks = hooks
        self.timeout = timeout

    # -- prepare ------------------------------------------------------------

    def prepare(
        self,
        config_id: str,
        backend: Any,
        *,
        shots: int,
        optimization_level: int = 3,
        seed_transpiler: int = 11,
        campaign_id: str = "adhoc",
        window: int = 1,
        planned_executions: int = 1,
        include_backend_properties: bool = False,
        execution_type: str = "IBM_QPU",
    ) -> Path:
        """Transpile once with a fixed seed and freeze everything a job needs.

        Nothing is submitted. ALAP scheduling is used for static circuits so
        idle times are captured; dynamic circuits are transpiled unscheduled.
        """
        if not 0 <= optimization_level <= 3 or shots < 1 or planned_executions < 1:
            raise ValueError("Invalid optimization level, shots or planned executions")
        item = self.hooks.build(config_id)
        logical = item.circuit
        dynamic = is_dynamic(logical)
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            initial_layout=item.initial_layout,
            seed_transpiler=seed_transpiler,
            scheduling_method=None if dynamic else "alap",
        )
        isa = pm.run(logical)
        if callable(getattr(backend, "check_faulty", None)):
            backend.check_faulty(isa)  # Raises before anything is archived.
        snapshot = capture_calibration(
            isa, backend, include_backend_properties=include_backend_properties
        )
        payload = dict(item.payload)
        if self.hooks.before_freeze is not None:
            payload.update(self.hooks.before_freeze(item, isa, snapshot))
        run_id = uuid4().hex
        stem = (
            f"{config_id}_IBM_{backend.name}_"
            f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{run_id[:8]}"
        )
        artifacts = self.output_dir / "artifacts" / stem
        artifacts.mkdir(parents=True, exist_ok=False)
        artifact_hashes = {}
        for name, circuit in (("logical.qpy", logical), ("isa.qpy", isa)):
            path = artifacts / name
            with path.open("xb") as stream:
                qpy.dump(circuit, stream)
            artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        layout = isa.layout.final_index_layout(filter_ancillas=True) if isa.layout else None
        prepared = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "algorithm": self.hooks.algorithm,
            "execution_type": execution_type,
            "run_id": run_id,
            "prepared_at": now(),
            "campaign_id": campaign_id,
            "window": window,
            "config_id": config_id,
            "config": item.config,
            "backend_name": backend.name,
            "shots": shots,
            "versions": package_versions(),
            "optimization_level": optimization_level,
            "seed_transpiler": seed_transpiler,
            "initial_layout": item.initial_layout,
            "physical_layout": layout,
            "scheduling_method": None if dynamic else "alap",
            "dynamic_circuit": dynamic,
            "artifact_hashes": artifact_hashes,
            "logical_circuit_hash": circuit_hash(logical, allow_dynamic=True),
            "isa_circuit_hash": circuit_hash(isa, allow_dynamic=True),
            "reference": item.reference,
            "calibration_snapshot": snapshot,
            "payload": payload,
            "planned_executions_per_window": planned_executions,
            "runtime_options": merge_options(DEFAULT_RUNTIME_OPTIONS, item.runtime_options),
        }
        write_json(artifacts / "prepared.json", prepared)
        journal = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "algorithm": self.hooks.algorithm,
            "config_id": config_id,
            "execution_type": execution_type,
            "backend_name": backend.name,
            "campaign_id": campaign_id,
            "window": window,
            "run_id": run_id,
            "status": "prepared",
            "saved_at": now(),
            "batch_id": None,
            "config": item.config,
            "submission_journal": [],
            "individual_results": [],
            "artifact_directory": str(artifacts.relative_to(self.output_dir)),
            "prepared_hash": record_hash(prepared),
        }
        record_path = self.output_dir / f"{stem}.json"
        write_json(record_path, journal)
        return record_path

    @staticmethod
    def load(record_path: Path) -> Tuple[Dict, Dict, Dict[str, QuantumCircuit]]:
        """Read a record and verify prepared/artifact/ISA integrity."""
        record_path = Path(record_path)
        journal = json.loads(record_path.read_text(encoding="utf-8"))
        artifacts = record_path.parent / journal["artifact_directory"]
        prepared = json.loads((artifacts / "prepared.json").read_text(encoding="utf-8"))
        if record_hash(prepared) != journal["prepared_hash"]:
            raise ValueError(f"Prepared record failed integrity verification: {record_path}")
        circuits = {}
        for name, digest in prepared["artifact_hashes"].items():
            path = artifacts / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Circuit artifact changed: {name}")
            with path.open("rb") as stream:
                circuits[name] = qpy.load(stream)[0]
        if circuit_hash(circuits["isa.qpy"], allow_dynamic=True) != prepared["isa_circuit_hash"]:
            raise ValueError("ISA circuit semantics failed integrity verification")
        return journal, prepared, circuits

    # -- budgets ------------------------------------------------------------

    def submitted_runs(self) -> Dict[Tuple[str, int], int]:
        """Jobs already journaled per (campaign_id, window) for this algorithm."""
        counts: Dict[Tuple[str, int], int] = {}
        for path in self.output_dir.glob("*.json"):
            other = json.loads(path.read_text(encoding="utf-8"))
            if other.get("algorithm") == self.hooks.algorithm:
                key = (other.get("campaign_id"), other.get("window"))
                counts[key] = counts.get(key, 0) + len(other.get("submission_journal", []))
        return counts

    def check_submittable(self, record_path: Path, backend: Any, runs: int):
        """Fresh record, matching backend, and within its window budget."""
        journal, prepared, circuits = self.load(record_path)
        if journal["status"] != "prepared" or runs < 1:
            raise ValueError(f"Only a fresh prepared record can be submitted: {record_path}")
        if backend.name != prepared["backend_name"]:
            raise ValueError("Submission backend differs from the calibration snapshot")
        used = self.submitted_runs().get((prepared["campaign_id"], prepared["window"]), 0)
        if used + runs > prepared["planned_executions_per_window"]:
            raise ValueError("Campaign window exceeds its planned execution budget")
        return journal, prepared, circuits

    # -- submit -------------------------------------------------------------

    @staticmethod
    def calibration_flag(prepared: Mapping[str, Any], current: Optional[str]) -> Dict[str, Any]:
        """Record whether the backend recalibrated between freeze and submission."""
        frozen = prepared["calibration_snapshot"]["calibrated_at"]
        changed = frozen is not None and current is not None and frozen != current
        if changed:
            print(
                f"WARNING: backend recalibrated since preparation ({frozen} -> {current}); "
                "submitting anyway, drift is measured from job.properties() at recovery.",
                file=sys.stderr,
            )
        return {"calibrated_at_submission": current, "recalibrated_before_submission": changed}

    def submit_job(
        self,
        record_path: Path,
        loaded: Tuple[Dict, Dict, Dict[str, QuantumCircuit]],
        sampler: Any,
        *,
        run_index: int,
        calibration: Mapping[str, Any],
        batch_id: Optional[str] = None,
    ) -> str:
        """Send one job for a record, journaling intent before and the ID after."""
        journal, prepared, circuits = loaded
        journal["status"] = "submitting"
        if batch_id is not None:
            journal["batch_id"] = batch_id
        sampler.options.environment.job_tags = [prepared["run_id"], f"repeat-{run_index}"]
        entry = {
            "run_index": run_index,
            "status": "submitting",
            "submitted_at": now(),
            "job_id": None,
            **calibration,
        }
        journal["submission_journal"].append(entry)
        write_json(record_path, journal)
        try:
            job = sampler.run([circuits["isa.qpy"]], shots=prepared["shots"])
            entry.update(job_id=job.job_id(), status="submitted")
        except Exception as exc:
            entry.update(status="submission_unknown", error=str(exc))
            journal["status"] = "submission_unknown"
            write_json(record_path, journal)
            raise
        journal["status"] = "submitted"
        write_json(record_path, journal)
        return entry["job_id"]

    def submit(self, record_path: Path, backend: Any, *, runs: int = 1) -> Dict[str, Any]:
        """Job mode: each run is one independent Runtime job, no Batch."""
        from qiskit_ibm_runtime import SamplerV2  # pylint: disable=import-outside-toplevel

        loaded = self.check_submittable(record_path, backend, runs)
        calibration = self.calibration_flag(loaded[1], current_calibration_date(backend))
        sampler = SamplerV2(mode=backend, options=loaded[1]["runtime_options"])
        for run_index in range(runs):
            self.submit_job(
                record_path, loaded, sampler, run_index=run_index, calibration=calibration
            )
        return loaded[0]

    # -- recover ------------------------------------------------------------

    def recover(self, record_path: Path, service: Any) -> Dict[str, Any]:
        """Retrieve saved job IDs, evaluate, and update the record. Never resubmits."""
        record_path = Path(record_path)
        journal, prepared, circuits = self.load(record_path)
        if journal["status"] == "prepared" and not journal["submission_journal"]:
            raise ValueError(f"Record was never submitted; nothing to recover: {record_path}")
        results = {r["job_id"]: r for r in journal["individual_results"]}
        incomplete = False
        for entry in journal["submission_journal"]:
            job_id = entry.get("job_id")
            if not job_id:
                incomplete = True
                continue
            if entry.get("status") == "completed" and job_id in results:
                continue
            try:
                job = service.job(job_id)
                pub = job.result(timeout=self.timeout)[0]
                counts = extract_counts(pub)
            except Exception as exc:  # pylint: disable=broad-except
                incomplete = True
                entry.update(status="retrieval_failed", error=str(exc))
                continue
            entry.update(status="completed", collected_at=now(), pub_metadata=pub.metadata)
            entry.pop("error", None)
            try:
                entry["execution_metrics"] = job.metrics()
            except Exception as exc:  # pylint: disable=broad-except
                entry["execution_metrics_error"] = str(exc)
            if self.hooks.after_job is not None:
                try:
                    entry.update(self.hooks.after_job(job, prepared, circuits))
                except Exception as exc:  # pylint: disable=broad-except
                    entry["after_job_error"] = str(exc)
            row = {
                "experiment_id": f"{prepared['run_id']}-{entry['run_index']:03d}",
                "config_id": prepared["config_id"],
                "job_id": job_id,
                "run_index": entry["run_index"],
                "optimization_level": prepared["optimization_level"],
                "backend_name": prepared["backend_name"],
                "shots": sum(counts.values()),
                "counts": counts,
                "status": "completed",
            }
            row.update(self.hooks.evaluate(counts, prepared, circuits["isa.qpy"]))
            results[job_id] = row
        journal["individual_results"] = sorted(results.values(), key=lambda r: r["run_index"])
        journal["status"] = "incomplete" if incomplete else "completed"
        journal["saved_at"] = now()
        write_json(record_path, journal)
        return journal


# ---------------------------------------------------------------------------
# Plans and manifests (one Batch per algorithm per window)
# ---------------------------------------------------------------------------


def job_order(items: Sequence[Mapping[str, Any]], seed: int) -> List[Tuple[int, int]]:
    """(item index, run index) pairs: seeded config shuffle, run 1 of all before run 2."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    rounds = max(item["runs"] for item in items)
    return [(i, r) for r in range(rounds) for i in order if r < items[i]["runs"]]


class BatchUnavailable(RuntimeError):
    """IBM would not open a Batch; the manifest was not touched."""


class Campaign:
    """Plan -> manifest -> Batch submission -> recovery for one RecordStore."""

    def __init__(self, store: RecordStore):
        self.store = store

    def prepare_plan(self, plan_path: Path, service: Any) -> Dict[str, Any]:
        """Validate a plan, then prepare one record per item; submit nothing.

        Plan JSON::

            {"backend": "ibm_fez", "campaign_id": "...", "window": 1,
             "shots": 1024, "optimization_level": 3, "seed_transpiler": 11,
             "order_seed": 7, "runs": 2,
             "items": [{"config": "GHZ4"}, {"config": "GHZ8", "runs": 1}]}

        Manifest is written to ``<plan stem>_manifest.json`` next to the plan.
        """
        plan_path = Path(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        manifest_path = plan_path.with_name(f"{plan_path.stem}_manifest.json")
        if manifest_path.exists():
            raise ValueError(f"Manifest already exists: {manifest_path}")
        items = plan.get("items") or []
        default_runs = plan.get("runs", 1)
        runs = [item.get("runs", default_runs) for item in items]
        if not items or any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in runs):
            raise ValueError("Plan needs items, each with a positive integer 'runs'")
        configs = [item["config"] for item in items]
        if len(set(configs)) != len(configs):
            raise ValueError("Plan lists a config more than once; use 'runs' instead")
        for config_id in configs:
            self.store.hooks.build(config_id)  # Unknown configs fail before any record.
        planned = plan.get("planned_executions", sum(runs))
        if sum(runs) > planned:
            raise ValueError("Plan runs exceed its planned_executions budget")
        backend = service.backend(plan["backend"])
        if callable(getattr(backend, "refresh", None)):
            backend.refresh()
        entries = []
        for item, item_runs in zip(items, runs):
            path = self.store.prepare(
                item["config"],
                backend,
                shots=plan.get("shots", 1024),
                optimization_level=plan.get("optimization_level", 3),
                seed_transpiler=plan.get("seed_transpiler", 11),
                campaign_id=plan["campaign_id"],
                window=plan.get("window", 1),
                planned_executions=planned,
            )
            _, prepared, circuits = self.store.load(path)
            isa = circuits["isa.qpy"]
            entries.append(
                {
                    "config": item["config"],
                    "runs": item_runs,
                    "record_path": str(path),
                    "calibrated_at": prepared["calibration_snapshot"]["calibrated_at"],
                    "physical_layout": prepared["physical_layout"],
                    "isa_depth": isa.depth(),
                    "isa_two_qubit_gates": sum(len(i.qubits) == 2 for i in isa.data),
                    "job_ids": [],
                }
            )
        order_seed = plan.get("order_seed", 0)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "algorithm": self.store.hooks.algorithm,
            "status": "prepared",
            "prepared_at": now(),
            "plan": str(plan_path),
            "backend_name": backend.name,
            "campaign_id": plan["campaign_id"],
            "window": plan.get("window", 1),
            "order_seed": order_seed,
            "job_order": job_order(entries, order_seed),
            "batch_id": None,
            "jobs_planned": sum(runs),
            "items": entries,
        }
        write_json(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path)}

    def submit_manifest(self, manifest_path: Path, backend: Any) -> Dict[str, Any]:
        """Validate every record, then submit all jobs in one Runtime Batch."""
        from qiskit_ibm_runtime import Batch, SamplerV2  # pylint: disable=import-outside-toplevel

        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "prepared":
            raise ValueError("Only a fresh prepared manifest can be submitted")
        loaded = []
        for item in manifest["items"]:
            path = Path(item["record_path"])
            loaded.append(self.store.check_submittable(path, backend, item["runs"]))
        prior = self.store.submitted_runs()
        totals: Dict[Tuple[str, int], int] = {}
        limits: Dict[Tuple[str, int], int] = {}
        for (_, prepared, _), item in zip(loaded, manifest["items"]):
            key = (prepared["campaign_id"], prepared["window"])
            totals[key] = totals.get(key, 0) + item["runs"]
            limits[key] = min(limits.get(key, 10**9), prepared["planned_executions_per_window"])
        for key, total in totals.items():
            if prior.get(key, 0) + total > limits[key]:
                raise ValueError(f"Manifest exceeds run budget of campaign/window {key}")
        current = current_calibration_date(backend)
        try:
            batch_cm = Batch(backend=backend)
        except Exception as exc:  # pylint: disable=broad-except
            # IBM refused to open the Batch (for example too many active ones):
            # nothing was sent, so the manifest stays prepared and can be retried.
            raise BatchUnavailable(
                f"Could not open a Batch; no job was sent and the manifest is still "
                f"prepared: {exc}"
            ) from exc
        manifest["status"] = "submitting"
        write_json(manifest_path, manifest)
        with batch_cm as batch:
            manifest["batch_id"] = batch.session_id
            write_json(manifest_path, manifest)
            samplers = {}
            for item_index, run_index in manifest["job_order"]:
                item = manifest["items"][item_index]
                journal, prepared, circuits = loaded[item_index]
                key = json.dumps(prepared["runtime_options"], sort_keys=True)
                if key not in samplers:
                    samplers[key] = SamplerV2(mode=batch, options=prepared["runtime_options"])
                job_id = self.store.submit_job(
                    Path(item["record_path"]),
                    (journal, prepared, circuits),
                    samplers[key],
                    run_index=run_index,
                    calibration=self.store.calibration_flag(prepared, current),
                    batch_id=batch.session_id,
                )
                item["job_ids"].append(job_id)
                write_json(manifest_path, manifest)
        manifest.update(status="submitted", submitted_at=now())
        write_json(manifest_path, manifest)
        return manifest

    @staticmethod
    def wait_manifest(
        manifest_path: Path,
        service: Any,
        *,
        poll_seconds: float = 30.0,
        max_seconds: Optional[float] = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
    ) -> Dict[str, Any]:
        """Block until every job of a submitted manifest reaches a final state.

        IBM may limit how many Batches are active at once, and a closed Batch
        stays active while its jobs are queued or running. Waiting here before
        opening the next Batch keeps only one active. Polling is read-only;
        timing out leaves the jobs running and the manifest untouched.
        """
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest["status"] == "prepared" or not manifest.get("batch_id"):
            raise ValueError("Manifest was never submitted; nothing to wait for")
        job_ids = [job_id for item in manifest["items"] for job_id in item.get("job_ids", [])]
        started = time.monotonic()
        while True:
            pending: Dict[str, str] = {}
            for job_id in job_ids:
                job = service.job(job_id)
                if not job.in_final_state():
                    pending[job_id] = str(job.status())
            done = len(job_ids) - len(pending)
            log(f"{Path(manifest_path).name}: {done}/{len(job_ids)} jobs finished")
            if not pending:
                return {
                    "jobs": len(job_ids),
                    "finished": done,
                    "waited_seconds": time.monotonic() - started,
                }
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                raise TimeoutError(
                    f"{len(pending)} of {len(job_ids)} jobs still pending after "
                    f"{max_seconds:.0f} s; they keep running (recover later)"
                )
            sleep(poll_seconds)

    def recover_manifest(self, manifest_path: Path, service: Any) -> Dict[str, Any]:
        """Recover every record of a submitted manifest; never resubmits."""
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] == "prepared" or not manifest.get("batch_id"):
            raise ValueError("Manifest was never submitted; submit it first")
        records = [self.store.recover(Path(i["record_path"]), service) for i in manifest["items"]]
        done = all(r["status"] == "completed" for r in records)
        manifest["status"] = "completed" if done else "incomplete"
        write_json(manifest_path, manifest)
        return {
            "manifest": str(manifest_path),
            "status": manifest["status"],
            "records": [summarize(r) for r in records],
        }

    @staticmethod
    def status(manifest_path: Path) -> Dict[str, Any]:
        """Offline status of a manifest and its records."""
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        counts: Dict[str, int] = {}
        for item in manifest["items"]:
            journal = json.loads(Path(item["record_path"]).read_text(encoding="utf-8"))
            counts[journal["status"]] = counts.get(journal["status"], 0) + 1
        return {
            "manifest": str(manifest_path),
            "algorithm": manifest.get("algorithm"),
            "campaign_id": manifest["campaign_id"],
            "window": manifest.get("window"),
            "status": manifest["status"],
            "batch_id": manifest.get("batch_id"),
            "jobs_planned": manifest["jobs_planned"],
            "jobs_submitted": sum(len(i.get("job_ids", [])) for i in manifest["items"]),
            "records_by_status": counts,
        }


def summarize(journal: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact view of a record for the CLI; the file keeps everything."""
    return {
        "config_id": journal.get("config_id"),
        "status": journal.get("status"),
        "backend_name": journal.get("backend_name"),
        "batch_id": journal.get("batch_id"),
        "jobs": [
            {
                "run_index": e.get("run_index"),
                "job_id": e.get("job_id"),
                "status": e.get("status"),
                "error": e.get("error"),
                "recalibrated_before_submission": e.get("recalibrated_before_submission"),
            }
            for e in journal.get("submission_journal", [])
        ],
        "success_rates": [r.get("success_rate") for r in journal.get("individual_results", [])],
    }
