"""Bound Batch sizes without duplicating prepared records or planned executions."""

import copy
import json
from pathlib import Path

from qward.examples.papers.ibm_campaign import write_json


def split_manifest(path, max_jobs=10):
    """Partition fresh records, keeping each configuration's repetitions together."""
    path = Path(path)
    parent = json.loads(path.read_text(encoding="utf-8"))
    if parent.get("parts") or parent["status"] != "prepared":
        return
    if parent.get("batch_id") or any(item["job_ids"] for item in parent["items"]):
        raise ValueError(f"Cannot partition submitted state: {path}")
    if not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs < 1:
        raise ValueError("max_jobs must be a positive integer")
    groups, group, size = [], [], 0
    for index, item in enumerate(parent["items"]):
        if item["runs"] > max_jobs:
            raise ValueError("Batch limit must fit all repetitions of one configuration")
        if group and size + item["runs"] > max_jobs:
            groups.append(group)
            group, size = [], 0
        group.append(index)
        size += item["runs"]
    if group:
        groups.append(group)
    if len(groups) <= 1:
        return
    parts = []
    for number, indices in enumerate(groups, 1):
        child = copy.deepcopy(parent)
        child["items"] = [copy.deepcopy(parent["items"][i]) for i in indices]
        positions = {old: new for new, old in enumerate(indices)}
        child["job_order"] = [
            [positions[i], repetition] for i, repetition in parent["job_order"] if i in positions
        ]
        child["jobs_planned"] = sum(item["runs"] for item in child["items"])
        child["parent_manifest"] = path.name
        part = path.with_name(f"{path.stem}_part{number:02d}.json")
        # A crash before the parent was saved may leave identical fresh children.
        if part.exists():
            if json.loads(part.read_text(encoding="utf-8")) != child:
                raise ValueError(f"Existing Batch part differs; inspect before resuming: {part}")
        else:
            write_json(part, child)
        parts.append(part.name)
    parent.update(parts=parts, status="partitioned", batch_job_limit=max_jobs)
    write_json(path, parent)


def manifest_parts(paths):
    """Resolve parts, verifying exact coverage before any IBM submission."""
    out = {}
    for key, path in paths.items():
        path = Path(path)
        if not path.exists():
            continue
        parent = json.loads(path.read_text(encoding="utf-8"))
        if not parent.get("parts"):
            out[key] = path
            continue
        expected = {item["config"]: item for item in parent["items"]}
        seen = set()
        for index, name in enumerate(parent["parts"], 1):
            if Path(name).name != name or name != f"{path.stem}_part{index:02d}.json":
                raise ValueError(f"Invalid Batch part path: {name}")
            part = path.with_name(name)
            child = json.loads(part.read_text(encoding="utf-8"))
            if child.get("parent_manifest") != path.name or any(
                child[field] != parent[field]
                for field in ("algorithm", "backend_name", "campaign_id", "window")
            ):
                raise ValueError(f"Batch part belongs to another collection: {part}")
            for item in child["items"]:
                config = item["config"]
                original = expected.get(config)
                if (
                    config in seen
                    or original is None
                    or any(item[field] != original[field] for field in ("runs", "record_path"))
                ):
                    raise ValueError(f"Duplicate or changed configuration in {part}")
                seen.add(config)
            order = sorted(tuple(entry) for entry in child["job_order"])
            wanted = sorted(
                (i, r) for i, item in enumerate(child["items"]) for r in range(item["runs"])
            )
            if (
                order != wanted
                or child["jobs_planned"] != len(wanted)
                or len(wanted) > parent["batch_job_limit"]
            ):
                raise ValueError(f"Invalid Batch workload: {part}")
            out[f"{key}:{index:02d}"] = part
        if seen != set(expected):
            raise ValueError(f"Batch parts do not cover all configurations: {path}")
    return out


def sync_parents(paths):
    """Summarize child state; preserve per-Batch IDs and each original job ID."""
    manifest_parts(paths)  # Reject corrupt partitions before updating parents.
    for path in paths.values():
        path = Path(path)
        if not path.exists():
            continue
        parent = json.loads(path.read_text(encoding="utf-8"))
        if not parent.get("parts"):
            continue
        children = [
            json.loads(path.with_name(name).read_text(encoding="utf-8")) for name in parent["parts"]
        ]
        items = {item["config"]: item for child in children for item in child["items"]}
        for item in parent["items"]:
            item["job_ids"] = items[item["config"]]["job_ids"]
        parent["batch_ids"] = [child["batch_id"] for child in children if child.get("batch_id")]
        parent["status"] = (
            "completed"
            if all(child["status"] == "completed" for child in children)
            else "incomplete"
        )
        write_json(path, parent)
