"""Run the full IBM Fez profile with a date-based collection identity."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from datetime import date
from importlib.metadata import PackageNotFoundError, version

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "qward/examples/papers"
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
    subprocess.run(["git", "add", "--", "qward/examples/papers"], check=True, cwd=ROOT)
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


def plans(campaign_date, *, create=False):
    """Derive dated plans; preserve existing numbered plans and all prior records.

    The legacy window field stays numeric for compatibility, using YYYYMMDD.
    Identity and duplicate protection are based on the full ISO date.
    """
    day = campaign_day(campaign_date)
    paths = {}
    for index, (key, template) in enumerate(templates().items()):
        path = PAPERS / DIRECTORIES[key] / "plans" / f"{key}-d{day.isoformat()}.json"
        expected = {
            **template,
            "campaign_id": f"fez-dataset-v2-{day.isoformat()}",
            "window": int(day.strftime("%Y%m%d")),
            "order_seed": int(day.strftime("%Y%m%d")) * 10 + index,
            "purpose": f"Full IBM Fez profile collected for {day.isoformat()}.",
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


def execute(campaign_date, mode, max_batches):
    """Run once, or recover existing submissions without creating more jobs."""
    paths = plans(campaign_date)
    state = manifests(paths)
    if mode == "validate":
        print(f"Campaign {campaign_date}: 172 configurations, 344 jobs, 1024 shots each on ibm_fez")
        return
    if mode == "complete":
        for key, path in state.items():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            expected = json.loads(paths[key].read_text(encoding="utf-8"))["planned_executions"]
            ids = [job for item in manifest["items"] for job in item["job_ids"]]
            if (
                manifest["status"] != "completed"
                or len(ids) != expected
                or len(set(ids)) != expected
                or not all(ids)
            ):
                raise ValueError(f"Campaign incomplete: {path}")
        return
    if mode == "run" and any(path.exists() for path in state.values()):
        raise ValueError("Campaign date already has manifests; use recovery, do not prepare again")

    from qward.examples.papers.campaign_matrix import load_runner
    from qward.examples.papers.campaign_submit import submit_window
    from qward.examples.papers.ibm_campaign import service_from_env
    from qward.examples.papers.ibm_experiment_base import resolve_ibm_credentials
    from qward.examples.papers.campaign_progress import report, stage

    report(f"Campaign {campaign_date}: mode={mode}, max active Batches={max_batches}")
    for package in ("qiskit", "qiskit-ibm-runtime", "torch"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        report(f"Environment: {package} {installed}")
    with stage("Connect to IBM Runtime"):
        service = service_from_env(**resolve_ibm_credentials())
    if mode == "run":
        paths = plans(campaign_date, create=True)
        report("Preparing all 172 configurations before submission; no QPU jobs sent yet")
        for index, (key, path) in enumerate(paths.items(), 1):
            report(f"Algorithm {index}/{len(paths)}: {key} preparation started")
            load_runner(key).campaign().prepare_plan(path, service)
            report(f"Algorithm {index}/{len(paths)}: {key} preparation completed")
        with stage("Push preparation checkpoint"):
            checkpoint()
        report("Preparation complete: submitting and collecting 344 jobs")
        submit_window(state, max_batches, service, checkpoint=checkpoint)
    else:
        for key, path in state.items():
            if not path.exists():
                report(f"{key}: no manifest; nothing to recover")
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["status"] in ("prepared", "completed"):
                report(f"{key}: {manifest['status']}; skipping recovery")
                continue
            campaign = load_runner(key).campaign()
            campaign.wait_manifest(path, service, max_seconds=3600)
            campaign.recover_manifest(path, service)
            checkpoint()
    execute(campaign_date, "complete", max_batches)
    report(f"Campaign {campaign_date}: verified all 344 jobs completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "run", "recover", "complete"))
    parser.add_argument("--date", required=True, help="Collection date in YYYY-MM-DD (Bogotá)")
    parser.add_argument("--max-batches", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    execute(args.date, args.mode, args.max_batches)
