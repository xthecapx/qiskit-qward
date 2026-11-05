"""Public, circuit-scoped calibration tables for the static explorer."""

import csv
import json


def calibration_record(prepared):
    """Keep measured qubit identities and unique operations, never instruction weights.

    Measurement mappings retain circuit order, including overwritten classical bits.
    Only the preparation snapshot is persisted at element granularity today.
    """
    snapshot = prepared.get("calibration_snapshot") or {}
    operations = {}
    measurements = []
    for instruction in snapshot.get("instructions", []):
        name = instruction["name"]
        qubits = instruction["qubits"]
        if name == "measure":
            measurements.append({"qubits": qubits, "bits": instruction["clbits"]})
        if name in {"delay", "barrier", "if_else"} or len(qubits) not in (1, 2):
            continue
        key = (name, tuple(qubits))
        value = {"name": name, "qubits": qubits, "error": instruction.get("error")}
        if key in operations and operations[key] != value:
            raise ValueError(f"Conflicting calibration for {key}")
        operations[key] = value
    qubits = {
        key: {
            field: value.get(field)
            for field in ("t1_s", "t2_s", "prob_meas1_prep0", "prob_meas0_prep1")
        }
        for key, value in snapshot.get("qubits", {}).items()
    }
    return {
        "run_id": prepared["run_id"],
        "backend": prepared["backend_name"],
        "calibrated_at": snapshot.get("calibrated_at"),
        "collected_at": snapshot.get("collected_at"),
        "source": "prepare_snapshot",
        "qubits_json": json.dumps(qubits, sort_keys=True),
        "operations_json": json.dumps(list(operations.values()), sort_keys=True),
        "measurements_json": json.dumps(measurements, sort_keys=True),
    }


def write_calibrations(path, records):
    """Write a portable sidecar without paths, account data, or IBM job IDs."""
    fields = [
        "run_id",
        "backend",
        "calibrated_at",
        "collected_at",
        "source",
        "qubits_json",
        "operations_json",
        "measurements_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
