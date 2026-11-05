"""Run the full IBM experiment profile with a backend-and-date identity."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import date
from importlib.metadata import PackageNotFoundError, version

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "qward/examples/papers"
BACKENDS = ("ibm_fez", "ibm_kingston", "ibm_marrakesh")
DIRECTORIES = {
    "ghz": "ghz",
    "bv": "bv",
    "vtp": "vtp",
    "qft": "qft",
    "pe": "phase_estimation",
    "grover": "grover",
    "rv": "random_volumetric",
    "shor": "shor",
}


def checkpoint():
    """Push durable state after preparation and each submitted/recovered Batch."""
    if os.environ.get("CAMPAIGN_CHECKPOINT") != "1":
        return
    subprocess.run(
        [
            "git",
            "add",
            "--",
            "qward/examples/papers/*/plans/*.json",
            "qward/examples/papers/*/data/qpu/v2/*.json",
            "qward/examples/papers/*/data/qpu/v2/artifacts/*/*.json",
            "qward/examples/papers/*/data/qpu/v2/artifacts/*/*.qpy",
            "qward/examples/papers/data/quantum_specific_cache/",
        ],
        check=True,
        cwd=ROOT,
    )
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False, cwd=ROOT)
    if changed.returncode not in (0, 1):
        raise RuntimeError("Could not inspect campaign checkpoint")
    if changed.returncode:
        subprocess.run(
            ["git", "commit", "-m", "data(experiments): checkpoint IBM campaign"],
            check=True,
            cwd=ROOT,
        )
        subprocess.run(["git", "push", "origin", "HEAD"], check=True, cwd=ROOT)


def campaign_day(value):
    """Require a canonical calendar date before using it in paths or IDs."""
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Campaign date must be YYYY-MM-DD")
    return parsed


def templates():
    """Use the reviewed full window-2 selection as an immutable profile."""
    plans_by_key = {}
    for key, directory in DIRECTORIES.items():
        path = PAPERS / directory / "plans" / f"{key}-w2.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        count = sum(item.get("runs", plan["runs"]) for item in plan["items"])
        if (
            plan["backend"] != "ibm_fez"
            or plan["window"] != 2
            or plan["campaign_id"] != "fez-dataset-v2"
            or plan["shots"] != 1024
            or count != plan["planned_executions"]
        ):
            raise ValueError(f"Unexpected campaign plan: {path}")
        plans_by_key[key] = plan
    if sum(p["planned_executions"] for p in plans_by_key.values()) != 344:
        raise ValueError("Expected 344 jobs; review the campaign before running")
    return plans_by_key


def identity(campaign_date, backend="ibm_fez"):
    """Keep existing Fez identities stable while isolating other chips."""
    day = campaign_day(campaign_date).isoformat()
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    chip = backend.removeprefix("ibm_")
    return {
        "branch": f"codex/ibm-{chip}-{day}",
        "marker": f".github/campaign-runs/{day if chip == 'fez' else chip + '-' + day}.json",
        "plan_suffix": f"d{day}" if chip == "fez" else f"{chip}-d{day}",
        "campaign_id": f"{chip}-dataset-v2-{day}",
    }


def plans(campaign_date, *, create=False, backend="ibm_fez"):
    """Derive dated plans; preserve existing numbered plans and all prior records.

    The legacy window field stays numeric for compatibility, using YYYYMMDD.
    Identity and duplicate protection are based on the full ISO date.
    """
    day = campaign_day(campaign_date)
    collection = identity(campaign_date, backend)
    paths = {}
    for index, (key, template) in enumerate(templates().items()):
        path = PAPERS / DIRECTORIES[key] / "plans" / f"{key}-{collection['plan_suffix']}.json"
        expected = {
            **template,
            "backend": backend,
            "campaign_id": collection["campaign_id"],
            "window": int(day.strftime("%Y%m%d")),
            "order_seed": int(day.strftime("%Y%m%d")) * 10 + index,
            "purpose": (
                f"Full IBM Fez profile collected for {day.isoformat()}."
                if backend == "ibm_fez"
                else f"Full IBM profile on {backend} collected for {day.isoformat()}."
            ),
        }
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Existing dated plan differs from the full profile: {path}")
        elif create:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(expected, stream, indent=2)
                stream.write("\n")
        paths[key] = path
    return paths


def manifests(paths):
    """Companion state files, shared with the local campaign driver."""
    return {key: path.with_name(path.stem + "_manifest.json") for key, path in paths.items()}


def recover_batches(state, service, *, attempts=3, retry_seconds=5):
    """Retry reads only, collecting other batches even when one remains incomplete."""
    from qward.examples.papers.campaign_matrix import load_runner
    from qward.examples.papers.campaign_parts import manifest_parts, sync_parents
    from qward.examples.papers.campaign_progress import report, stage

    parts = manifest_parts(state)
    for attempt in range(1, attempts + 1):
        retry = False
        for key, path in parts.items():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["status"] == "completed" or not manifest.get("batch_id"):
                continue
            campaign = load_runner(key.split(":", 1)[0]).campaign()
            try:
                with stage(f"{key}: retrieve results (attempt {attempt}/{attempts})"):
                    result = campaign.recover_manifest(path, service, skip_pending=True)
                if result["status"] != "completed":
                    retry = True
                    report(f"{key}: incomplete; keeping original job IDs, no replacement jobs")
            except Exception as exc:  # pylint: disable=broad-except
                retry = True
                report(f"{key}: retrieval failed: {type(exc).__name__}: {exc}")
            # A storage failure must stop work rather than be retried as an IBM read.
            sync_parents(state)
            checkpoint()
        if not retry:
            break
        if attempt < attempts:
            report(f"Retrying incomplete retrievals in {retry_seconds}s; no jobs will be submitted")
            time.sleep(retry_seconds)


def verify_complete(paths, state):
    """Require the complete, unique planned submission set across the collection."""
    from qward.examples.papers.campaign_parts import sync_parents

    sync_parents(state)
    seen = set()
    for key, path in state.items():
        if not path.exists():
            raise ValueError(f"Campaign incomplete: missing {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = json.loads(paths[key].read_text(encoding="utf-8"))["planned_executions"]
        ids = [job for item in manifest["items"] for job in item["job_ids"]]
        if (
            manifest["status"] != "completed"
            or len(ids) != expected
            or len(set(ids)) != expected
            or not all(ids)
            or seen.intersection(ids)
        ):
            raise ValueError(f"Campaign incomplete: {path}")
        seen.update(ids)


def execute(
    campaign_date,
    mode,
    max_batches,
    backend="ibm_fez",
    *,
    max_jobs_per_batch=10,
    batch_max_time=3600,
):
    """Run independent stages; resume only untouched batches, never failed jobs."""
    modes = ("identity", "validate", "run", "prepare", "submit", "recover", "complete")
    if mode not in modes:
        raise ValueError(f"Unknown campaign mode: {mode}")
    if max_batches not in (1, 2, 3) or max_jobs_per_batch not in (6, 10, 14):
        raise ValueError("Expected 1-3 active batches and 6, 10 or 14 jobs per batch")
    if batch_max_time not in (600, 1800, 3600):
        raise ValueError("Batch lifetime must be 600, 1800 or 3600 seconds")
    collection = identity(campaign_date, backend)
    if mode == "identity":
        for key, value in collection.items():
            print(f"{key}={value}")
        return
    paths = plans(campaign_date, backend=backend)
    state = manifests(paths)
    if mode == "validate":
        print(
            f"Campaign {campaign_date}: 172 configurations, 344 jobs, 1024 shots each on {backend}; "
            f"up to {max_batches} active batches, {max_jobs_per_batch} jobs/batch, "
            f"requested lifetime {batch_max_time}s"
        )
        return
    if mode == "run" and any(path.exists() for path in state.values()):
        raise ValueError(
            "Campaign date already has manifests; use prepare/submit/recover to resume"
        )
    for path in state.values():
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest.get("backend_name") != backend
                or manifest.get("campaign_id") != collection["campaign_id"]
            ):
                raise ValueError(f"Manifest belongs to another backend or campaign: {path}")
    if mode == "complete":
        verify_complete(paths, state)
        return
    if mode == "submit" and not all(path.exists() for path in state.values()):
        raise ValueError("Prepare every algorithm before submitting any jobs")

    from qward.examples.papers.campaign_matrix import load_runner
    from qward.examples.papers.campaign_parts import manifest_parts, split_manifest, sync_parents
    from qward.examples.papers.campaign_submit import submit_window
    from qward.examples.papers.ibm_campaign import service_from_env
    from qward.examples.papers.ibm_experiment_base import resolve_ibm_credentials
    from qward.examples.papers.campaign_progress import report, stage

    report(f"Campaign {backend}/{campaign_date}: mode={mode}, max active Batches={max_batches}")
    for package in ("qiskit", "qiskit-ibm-runtime", "torch"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        report(f"Environment: {package} {installed}")
    with stage("Connect to IBM Runtime"):
        service = service_from_env(**resolve_ibm_credentials())
    if mode in ("run", "prepare"):
        paths = plans(campaign_date, create=True, backend=backend)
        report("Preparing all 172 configurations before submission; no QPU jobs sent yet")
        for index, (key, path) in enumerate(paths.items(), 1):
            if state[key].exists():
                report(f"{key}: preserving existing preparation and submission state")
            else:
                with stage(f"Algorithm {index}/{len(paths)}: {key} preparation"):
                    load_runner(key).campaign().prepare_plan(path, service)
            split_manifest(state[key], max_jobs_per_batch)
            manifest_parts({key: state[key]})
            checkpoint()
        report(f"Preparation complete: {len(manifest_parts(state))} batches, 344 planned jobs")
    if mode in ("run", "submit"):
        try:
            submit_window(
                manifest_parts(state),
                max_batches,
                service,
                checkpoint=checkpoint,
                recover_results=False,
                batch_max_time=batch_max_time,
            )
        finally:
            sync_parents(state)
            checkpoint()
    if mode in ("run", "recover"):
        recover_batches(state, service)
        verify_complete(paths, state)
        report(f"Campaign {backend}/{campaign_date}: verified all 344 jobs completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("identity", "validate", "run", "prepare", "submit", "recover", "complete")
    )
    parser.add_argument("--date", required=True, help="Collection date in YYYY-MM-DD (Bogotá)")
    parser.add_argument("--max-batches", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--backend", choices=BACKENDS, default="ibm_fez")
    parser.add_argument("--max-jobs-per-batch", type=int, choices=(6, 10, 14), default=10)
    parser.add_argument("--batch-max-time", type=int, choices=(600, 1800, 3600), default=3600)
    args = parser.parse_args()
    execute(
        args.date,
        args.mode,
        args.max_batches,
        backend=args.backend,
        max_jobs_per_batch=args.max_jobs_per_batch,
        batch_max_time=args.batch_max_time,
    )
