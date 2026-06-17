#!/usr/bin/env python3
"""
IBM QPU Pipeline Validation Script

Runs the SMALLEST config for each of 6 algorithms on IBM QPU to validate
the full data collection pipeline (QWARD metrics + calibration + gate errors + DSR).

Algorithms tested:
1. Grover (2 qubits)
2. BV (2 qubits)
3. GHZ (2 qubits)
4. Phase Estimation (3 total qubits)
5. vTP (4 total qubits, variation protocol)
6. Random Volumetric (2 qubits, depth 2)

Usage:
    PYTHONPATH=. uv run python qward/examples/papers/validate_ibm_pipeline.py
    PYTHONPATH=. uv run python qward/examples/papers/validate_ibm_pipeline.py --backend ibm_fez
    PYTHONPATH=. uv run python qward/examples/papers/validate_ibm_pipeline.py --dry-run
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Validation configs: smallest possible per algorithm
VALIDATION_CONFIGS = [
    {
        "algorithm": "Grover",
        "module": "qward.examples.papers.grover.grover_ibm",
        "class": "GroverIBMExperiment",
        "config_id": "S2-1",
    },
    {
        "algorithm": "BV",
        "module": "qward.examples.papers.bv.bv_ibm",
        "class": "BVIBMExperiment",
        "config_id": "BV2-ONES",
    },
    {
        "algorithm": "GHZ",
        "module": "qward.examples.papers.ghz.ghz_ibm",
        "class": "GHZIBMExperiment",
        "config_id": "GHZ2",
    },
    {
        "algorithm": "PE",
        "module": "qward.examples.papers.phase_estimation.pe_ibm",
        "class": "PEIBMExperiment",
        "config_id": "PE-Z2",
    },
    {
        "algorithm": "vTP",
        "module": "qward.examples.papers.vtp.vtp_ibm",
        "class": "VTPIBMExperiment",
        "config_id": "VTP-V1-SINGLE",
    },
    {
        "algorithm": "RV",
        "module": "qward.examples.papers.random_volumetric.rv_ibm",
        "class": "RVIBMExperiment",
        "config_id": "RV2-D2",
    },
]


def load_ibm_credentials():
    """Load IBM credentials from .env file."""
    import os

    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "IBM_QUANTUM" in line and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"')
    return {
        "channel": os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_cloud"),
        "token": os.environ.get("IBM_QUANTUM_TOKEN"),
        "instance": os.environ.get("IBM_QUANTUM_INSTANCE"),
    }


def run_validation(backend_name=None, dry_run=False, opt_levels=None):
    """Run validation across all 6 algorithms."""
    if opt_levels is None:
        opt_levels = [2, 3]

    print("=" * 70)
    print("IBM QPU PIPELINE VALIDATION")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Backend: {backend_name or 'least_busy'}")
    print(f"Optimization levels: {opt_levels}")
    print(f"Dry run: {dry_run}")
    print("=" * 70)

    results = {}

    for i, cfg in enumerate(VALIDATION_CONFIGS, 1):
        algo = cfg["algorithm"]
        config_id = cfg["config_id"]
        print(f"\n{'─' * 70}")
        print(f"[{i}/6] {algo} — config: {config_id}")
        print(f"{'─' * 70}")

        if dry_run:
            # Dry run: just import and create circuit
            try:
                import importlib

                mod = importlib.import_module(cfg["module"])
                ExperimentClass = getattr(mod, cfg["class"])
                experiment = ExperimentClass(shots=1024, timeout=600)
                config = experiment.get_config(config_id)
                circuit = experiment.create_circuit(config)
                print(
                    f"  Circuit: {circuit.num_qubits} qubits, depth={circuit.depth()}, gates={circuit.size()}"
                )
                print(f"  Config: {experiment.get_config_description(config)}")
                results[algo] = {
                    "status": "dry_run_ok",
                    "qubits": circuit.num_qubits,
                    "depth": circuit.depth(),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                results[algo] = {"status": "dry_run_error", "error": str(e)}
            continue

        # Real execution
        try:
            import importlib

            mod = importlib.import_module(cfg["module"])
            ExperimentClass = getattr(mod, cfg["class"])
            experiment = ExperimentClass(shots=1024, timeout=600)

            creds = load_ibm_credentials()
            result = experiment.run(
                config_id=config_id,
                backend_name=backend_name,
                optimization_levels=opt_levels,
                save_results=True,
                channel=creds["channel"],
                token=creds["token"],
                instance=creds["instance"],
            )

            # Summarize
            if result and "error" not in result:
                sr = result.get("algorithm_metrics", {}).get("success_rate", "N/A")
                has_cal = result.get("backend_calibration") is not None
                has_ge = bool(result.get("gate_error_characterization", {}))
                has_qward = result.get("qward_metrics") is not None
                print(
                    f"\n  SUCCESS: rate={sr}, calibration={has_cal}, gate_errors={has_ge}, qward={has_qward}"
                )
                results[algo] = {
                    "status": "success",
                    "success_rate": sr,
                    "has_calibration": has_cal,
                    "has_gate_errors": has_ge,
                    "has_qward_metrics": has_qward,
                }
            else:
                error = result.get("error", "unknown") if result else "no result"
                print(f"\n  FAILED: {error}")
                results[algo] = {"status": "failed", "error": error}

        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            traceback.print_exc()
            results[algo] = {"status": "exception", "error": str(e)}

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    for algo, res in results.items():
        status = res["status"]
        if status == "success":
            print(
                f"  {algo:6s}: OK (rate={res['success_rate']}, cal={res['has_calibration']}, ge={res['has_gate_errors']})"
            )
        elif status == "dry_run_ok":
            print(f"  {algo:6s}: DRY RUN OK ({res['qubits']}q, depth={res['depth']})")
        else:
            print(f"  {algo:6s}: {status} — {res.get('error', '')[:60]}")

    # Save summary
    out_path = Path(__file__).resolve().parent / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "backend": backend_name or "least_busy",
                "opt_levels": opt_levels,
                "dry_run": dry_run,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM QPU Pipeline Validation")
    parser.add_argument("--backend", type=str, default=None, help="IBM backend name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only create circuits, no QPU execution"
    )
    parser.add_argument(
        "--opt-levels", type=int, nargs="+", default=[2, 3], help="Optimization levels"
    )
    args = parser.parse_args()

    run_validation(
        backend_name=args.backend,
        dry_run=args.dry_run,
        opt_levels=args.opt_levels,
    )
