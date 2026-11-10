"""Schedule prepared algorithm manifests within one window's active Batch limit."""

import argparse
import errno
import json
import os
import time
from collections import deque
from pathlib import Path

from qward.examples.papers.campaign_matrix import RUNNERS, load_runner
from qward.examples.papers.campaign_progress import report, stage
from qward.examples.papers.ibm_campaign import service_from_env, write_json
from qward.examples.papers.ibm_experiment_base import resolve_ibm_credentials


def submitted_ids(manifest):
    """An interrupted submission needs reconciliation if any job ID is missing."""
    ids = [job for item in manifest["items"] for job in item["job_ids"]]
    if (
        not manifest["batch_id"]
        or len(ids) != manifest["jobs_planned"]
        or len(set(ids)) != len(ids)
        or not all(ids)
        or any(len(item["job_ids"]) != item["runs"] for item in manifest["items"])
    ):
        raise ValueError(
            f"{manifest['algorithm']}: partial submission; inspect its manifest before resuming"
        )
    return deque(ids)


def submit_window(
    paths,
    max_batches,
    service=None,
    *,
    poll_seconds=30,
    checkpoint=None,
    recover_results=True,
    batch_max_time=None,
    refresh_service=None,
    submit_only=False,
):
    """Track existing batches first. With a client factory, drain each wave before reconnecting."""
    if submit_only and max_batches is None and refresh_service is None:
        # Offer every part to IBM; service-side admission still applies.
        max_batches = len(paths)
    elif max_batches not in (1, 2, 3):
        raise ValueError("max_batches must be 1, 2 or 3")
    pending, active = deque(), {}
    campaigns = {}
    window_keys = set()
    group_jobs = [0, 0, 0]
    groups = {}
    for key, path in paths.items():
        manifest = json.loads(path.read_text())
        if submit_only:
            # Balance intact parts in plan order, independently of their submission status.
            group = group_jobs.index(min(group_jobs))
            groups[key] = group + 1
            group_jobs[group] += manifest["jobs_planned"]
            if manifest.get("submission_group", group + 1) != group + 1:
                raise ValueError(f"{path}: submission group changed")
        window_keys.add((manifest["backend_name"], manifest["campaign_id"], manifest["window"]))
        status = manifest["status"]
        if status == "prepared":
            if manifest["batch_id"] or any(item["job_ids"] for item in manifest["items"]):
                raise ValueError(f"{path}: prepared manifest already contains submission IDs")
            pending.append((key, manifest["backend_name"]))
        elif status in ("submitted", "submitting", "incomplete", "completed"):
            ids = submitted_ids(manifest)
            if status == "completed":
                print(f"{key}: already completed", flush=True)
                continue
            active[key] = ids
            print(f"{key}: tracking existing Batch {manifest['batch_id']}", flush=True)
        else:
            raise ValueError(f"{path}: unexpected manifest status {status}")
        campaigns[key] = load_runner(key.split(":", 1)[0]).campaign()

    if len(window_keys) != 1:
        raise ValueError("Manifests must belong to the same backend, campaign and window")
    if submit_only:
        if recover_results:
            raise ValueError("Submit-only mode requires deferred recovery")
        pending = deque(sorted(pending, key=lambda entry: groups[entry[0]]))
        print(f"Planned jobs in submission groups 1–3: {group_jobs}", flush=True)
    if campaigns and service is None:
        service = service_from_env(**resolve_ibm_credentials())
    while pending or (active and not submit_only):
        can_open = refresh_service is None or not active
        while can_open and pending and len(active) < max_batches:
            key, backend = pending.popleft()
            if submit_only:
                path = paths[key]
                manifest = json.loads(path.read_text())
                manifest["submission_group"] = groups[key]
                write_json(path, manifest)
                print(f"Submission group {groups[key]}/3: {key}", flush=True)
            print(f"{key}: opening Batch", flush=True)
            with stage(f"{key}: submit Batch"):
                options = {} if batch_max_time is None else {"batch_max_time": batch_max_time}
                manifest = campaigns[key].submit_manifest(
                    paths[key], service.backend(backend), **options
                )
            if checkpoint is not None:
                checkpoint()
            active[key] = submitted_ids(manifest)
            print(f"{key}: tracking {manifest['batch_id']} ({len(active)} active)", flush=True)

        if submit_only and not pending:
            break
        for key, ids in list(active.items()):
            # A single unfinished job keeps this Batch in its slot. Finished
            # IDs can be discarded here; recovery uses the durable manifest.
            if refresh_service is not None or submit_only:
                for job_id in tuple(ids):
                    job = service.job(job_id)
                    status = job.status()
                    status = getattr(status, "name", status)
                    if status in ("ERROR", "CANCELLED", "CANCELED"):
                        raise RuntimeError(
                            f"{key}: IBM job {job_id} is {status}: {job.error_message()}. "
                            "Submission stopped; recover the recorded job IDs."
                        )
                    if status == "DONE":
                        ids.remove(job_id)
            else:
                while ids and service.job(ids[0]).in_final_state():
                    ids.popleft()
            if ids:
                report(
                    f"{key}: waiting on IBM job {ids[0]}; "
                    f"{len(ids)} jobs not yet confirmed terminal, {len(active)} active Batches"
                )
            else:
                if recover_results:
                    result = campaigns[key].recover_manifest(paths[key], service)
                    if result["status"] != "completed":
                        raise RuntimeError(f"{key}: recovery incomplete; inspect {paths[key]}")
                del active[key]
                if checkpoint is not None:
                    checkpoint()
                print(
                    f"{key}: {'recovered' if recover_results else 'terminal; collect in recovery step'}",
                    flush=True,
                )
        if refresh_service is not None and pending and not active:
            report(
                "Wave terminal. Waiting 120 seconds before connecting a fresh IBM Runtime client."
            )
            time.sleep(120)
            service = refresh_service()
        elif active and (refresh_service is not None or not pending or len(active) >= max_batches):
            time.sleep(poll_seconds)


def lock_plan(lock):
    """Acquire a nonblocking lock for the lifetime of this open plan file."""
    if os.name == "nt":
        import msvcrt

        # Lock one byte beyond the JSON so other processes can still read it.
        lock.seek(0, os.SEEK_END)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batches", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    paths = {p.name.split("-w", 1)[0]: p for p in args.manifests}
    if len(paths) != len(args.manifests) or set(paths) != set(RUNNERS):
        parser.error("expected one manifest for each campaign algorithm")
    # Plan files stay in place while their manifests are atomically replaced.
    # Lock the same plan on every invocation, including after Ctrl-C.
    plan = paths["ghz"].with_name(paths["ghz"].name.replace("_manifest.json", ".json"))
    with plan.open("rb") as lock:
        try:
            lock_plan(lock)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            parser.exit(1, "A submit driver is already running for this window.\n")
        submit_window(paths, args.max_batches)


if __name__ == "__main__":
    main()
