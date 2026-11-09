"""Plan and complete replacements for confirmed failures in the November 7 Fez Grover run."""

import argparse
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from qward.examples.papers.campaign_matrix import PAPERS, load_runner
from qward.examples.papers.campaign_parts import manifest_parts, split_manifest, sync_parents
from qward.examples.papers.campaign_submit import submit_window
from qward.examples.papers.ibm_campaign import manifest_record_path, service_from_env, write_json
from qward.examples.papers.ibm_experiment_base import resolve_ibm_credentials

SOURCE = "fez-oracle-growth-v1-2026-11-07"
DIRECTORY = PAPERS / "grover/plans"
PLAN = DIRECTORY / "grover-fez-oracle-growth-d2026-11-07-makeup.json"
MANIFEST = PLAN.with_stem(PLAN.stem + "_manifest")


def connect():
    return service_from_env(**resolve_ibm_credentials())


def failed_items(service):
    original = DIRECTORY / "grover-fez-oracle-growth-d2026-11-07_manifest.json"
    manifest = json.loads(original.read_text())
    if manifest["campaign_id"] != SOURCE or manifest["backend_name"] != "ibm_fez":
        raise ValueError("Unexpected source manifest")
    items = []
    for item in manifest["items"]:
        record = json.loads(manifest_record_path(original, item).read_text())
        journal = sorted(record["submission_journal"], key=lambda entry: entry["run_index"])
        if (
            record["campaign_id"] != SOURCE
            or record["config_id"] != item["config"]
            or len(journal) != item["runs"]
            or {entry["job_id"] for entry in journal} != set(item["job_ids"])
        ):
            raise ValueError(f"Original journal does not match manifest: {item['config']}")
        results = {r["job_id"] for r in record["individual_results"]}
        failed = []
        for entry in journal:
            job_id = entry["job_id"]
            if job_id in results:
                continue
            status = service.job(job_id).status()
            if getattr(status, "name", status) not in ("ERROR", "CANCELLED", "CANCELED"):
                raise ValueError(
                    f"{job_id}: IBM status {status}; resolve original job before replacement"
                )
            failed.append({"job_id": job_id, "run_index": entry["run_index"]})
        if failed:
            items.append({"config": item["config"], "runs": len(failed), "replaces": failed})
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "submit", "recover"))
    mode = parser.parse_args().mode
    if mode == "plan":
        items = failed_items(connect())
        if not items:
            print("All original Grover results are present.")
            return
        total = sum(item["runs"] for item in items)
        plan = {
            "backend": "ibm_fez",
            "campaign_id": SOURCE + "-makeup",
            "shots": 1024,
            "optimization_level": 3,
            "seed_transpiler": 11,
            "order_seed": 20261107,
            "window": int(datetime.now(ZoneInfo("America/Bogota")).strftime("%Y%m%d")),
            "planned_executions": total,
            "items": items,
        }
        if PLAN.exists() or MANIFEST.exists():
            raise ValueError(f"Makeup already planned; inspect {PLAN} before continuing")
        write_json(PLAN, plan)
        print(f"{total} replacements planned: {PLAN}")
        return
    plan = json.loads(PLAN.read_text())
    state = {"grover": MANIFEST}
    campaign = load_runner("grover").campaign()
    if mode == "submit":
        if PLAN.with_stem(PLAN.stem + "-jobs").exists():
            raise ValueError("Standalone recovery planned; use scripts.fez_grover_jobs submit")
        service = connect()
        if failed_items(service) != plan["items"]:
            raise ValueError("Original job state changed since planning; inspect before submitting")
        if not MANIFEST.exists():
            campaign.prepare_plan(PLAN, service)
        split_manifest(MANIFEST, max_jobs=6)
        try:
            submit_window(
                manifest_parts(state),
                1,
                service,
                batch_max_time=3600,
                refresh_service=connect,
                recover_results=False,
            )
        finally:
            sync_parents(state)
        return
    if not MANIFEST.exists():
        raise ValueError("Missing makeup manifest; inspect submission state")
    service = connect()
    incomplete = []
    try:
        for key, path in manifest_parts(state).items():
            saved = json.loads(path.read_text())
            if saved["status"] == "completed":
                continue
            if not saved.get("batch_id"):
                incomplete.append(f"{key}: {saved['status']} (no batch ID)")
                continue
            result = campaign.recover_manifest(path, service, skip_pending=True)
            if result["status"] != "completed":
                incomplete.append(f"{key}: {result['status']}")
    finally:
        sync_parents(state)
    if incomplete:
        raise SystemExit("Recovery saved available results. Remaining: " + ", ".join(incomplete))
    from scripts.ibm_campaign_ci import verify_complete

    verify_complete({"grover": PLAN}, state)
    print(f"Verified {plan['planned_executions']} makeup jobs. Original job IDs are in {PLAN}.")


if __name__ == "__main__":
    main()
