"""Finish the seven remaining November 7 Fez oracle repetitions using standalone jobs."""

import argparse
from datetime import datetime
import json
from zoneinfo import ZoneInfo

from scripts import fez_grover_makeup as batch
from qward.examples.papers.campaign_submit import lock_plan
from qward.examples.papers.ibm_campaign import manifest_record_path, write_json

STATE = batch.PLAN.with_stem(batch.PLAN.stem + "-jobs")
CAMPAIGN = batch.SOURCE + "-standalone"
EXPECTED = {"H5-1": 2, "H5-3": 1, "H5-4": 2, "H5-5": 2}


def remaining():
    """Map each missing result to its original repetition and optional failed batch retry."""
    source = json.loads(batch.PLAN.read_text())
    replacements = {i["config"]: i["replaces"] for i in source["items"]}
    slots = []
    parent = json.loads(batch.MANIFEST.read_text())
    for filename in parent["parts"]:
        part = batch.DIRECTORY / filename
        manifest = json.loads(part.read_text())
        for item in manifest["items"]:
            record = json.loads(manifest_record_path(part, item).read_text())
            config = item["config"]
            entries = record["submission_journal"]
            if [e["job_id"] for e in entries] != item["job_ids"]:
                raise ValueError(f"Reconcile job IDs for {config}")
            results = {r["job_id"] for r in record["individual_results"]}
            for index, original in enumerate(replacements[config]):
                retry = next((e for e in entries if e["run_index"] == index), None)
                if retry and retry["job_id"] in results:
                    continue
                if retry:
                    if not retry["job_id"] or retry["status"] != "retrieval_failed":
                        raise ValueError(f"Unresolved batch submission for {config}")
                elif manifest["status"] != "prepared" or entries or manifest.get("batch_id"):
                    raise ValueError(f"Ambiguous unsubmitted repetition for {config}")
                slots.append(
                    dict(
                        config=config,
                        original=original,
                        batch_job_id=retry["job_id"] if retry else None,
                    )
                )
    if {c: sum(s["config"] == c for s in slots) for c in EXPECTED} != EXPECTED or len(slots) != 7:
        raise ValueError("The unresolved workload has changed. Review the journals.")
    return slots


def execute(mode):
    slots = remaining()
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if [
            {k: s[k] for k in ("config", "original", "batch_job_id")} for s in state["slots"]
        ] != slots:
            raise ValueError("Standalone plan differs from source journals")
    elif mode == "plan":
        state = {"campaign_id": CAMPAIGN, "backend": "ibm_fez", "shots": 1024, "slots": slots}
        write_json(STATE, state)
    else:
        raise ValueError("Run plan first")
    if mode == "plan":
        print(f"Seven standalone jobs: {EXPECTED}. Plan: {STATE}")
        return
    service = batch.connect()
    store = batch.load_runner("grover").campaign().store
    if mode == "submit":
        # A retrieval error alone does not establish that a replacement is safe.
        for slot in slots:
            for job_id in (slot["original"]["job_id"], slot["batch_job_id"]):
                if job_id:
                    status = service.job(job_id).status()
                    if getattr(status, "name", status) not in ("ERROR", "CANCELLED", "CANCELED"):
                        raise ValueError(f"Resolve {job_id} ({status}) before submitting")
        backend = service.backend("ibm_fez")
        if "window" not in state:
            state["window"] = int(datetime.now(ZoneInfo("America/Bogota")).strftime("%Y%m%d"))
            write_json(STATE, state)
    unresolved = []
    ids = set()
    for slot in state["slots"]:
        if "record" not in slot:
            if mode == "recover":
                unresolved.append(f"{slot['config']}: unsubmitted")
                continue
            path = store.prepare(
                slot["config"],
                backend,
                shots=1024,
                optimization_level=3,
                seed_transpiler=11,
                campaign_id=CAMPAIGN,
                window=state["window"],
                planned_executions=7,
            )
            slot["record"] = path.name
            write_json(STATE, state)
        path = batch.DIRECTORY.parent / "data/qpu/v2" / slot["record"]
        record = json.loads(path.read_text())
        if record["config_id"] != slot["config"] or record["campaign_id"] != CAMPAIGN:
            raise ValueError(f"Unexpected record: {path}")
        if mode == "submit" and record["status"] == "prepared" and not record["submission_journal"]:
            record = store.submit(path, backend, runs=1)
        entries = record["submission_journal"]
        if mode == "recover" and not entries and record["status"] == "prepared":
            unresolved.append(f"{slot['config']}: prepared only")
            continue
        if len(entries) != 1 or not entries[0].get("job_id") or entries[0]["job_id"] in ids:
            raise ValueError(f"Inspect ambiguous submission before continuing: {path}")
        job_id = entries[0]["job_id"]
        ids.add(job_id)
        slot["job_id"] = job_id
        write_json(STATE, state)
        print(slot["config"], job_id, flush=True)
        if mode == "recover":
            record = store.recover(path, service, skip_pending=True)
            if record["status"] != "completed":
                unresolved.append(f"{slot['config']}: {job_id}")
    if unresolved:
        raise SystemExit("Outstanding: " + ", ".join(unresolved))
    print(f"{mode} verified {len(ids)} distinct standalone job IDs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "submit", "recover"))
    args = parser.parse_args()
    with batch.PLAN.open("rb") as lock:
        lock_plan(lock)
        execute(args.mode)
