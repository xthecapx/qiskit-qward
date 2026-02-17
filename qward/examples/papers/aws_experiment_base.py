#!/usr/bin/env python3
"""AWS Braket experiment base module."""

import argparse
import json
import os
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from dotenv import load_dotenv
from qiskit import QuantumCircuit

from qward.algorithms import AWSJobResult, QuantumCircuitExecutor
from qward.examples.papers.experiment_helpers import (
    calculate_qward_metrics,
    calculate_statistical_analysis,
)

# Type variable for config classes
ConfigT = TypeVar("ConfigT")

# Load .env credentials when available
load_dotenv()


@dataclass
class AWSExperimentConfig:
    """Base configuration for AWS experiments."""

    config_id: str
    num_qubits: int
    expected_success: float
    circuit_depth: int
    region: str = "us-west-1"


class AWSExperimentBase(ABC, Generic[ConfigT]):
    """Base class for AWS Braket experiments."""

    def __init__(
        self,
        shots: int = 1024,
        timeout: int = 600,
        output_subdir: str = "data/qpu/aws",
    ):
        self.shots = shots
        self.timeout = timeout
        self.output_subdir = output_subdir
        self.executor = QuantumCircuitExecutor(shots=shots)

    # =========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # =========================================================================

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the algorithm name (e.g., 'GROVER', 'QFT')."""
        pass

    @abstractmethod
    def get_config(self, config_id: str) -> ConfigT:
        """Get the configuration for the given config_id."""
        pass

    @abstractmethod
    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        pass

    @abstractmethod
    def create_circuit(self, config: ConfigT) -> QuantumCircuit:
        """Create the quantum circuit for the given configuration."""
        pass

    @abstractmethod
    def create_success_criteria(self, config: ConfigT) -> Callable[[str], bool]:
        """Create a success criteria function for the configuration."""
        pass

    @abstractmethod
    def get_expected_outcomes(self, config: ConfigT) -> List[str]:
        """Return expected outcomes (marked bitstrings) for DSR."""
        pass

    @abstractmethod
    def get_random_chance(self, config: ConfigT) -> float:
        """Get the classical random chance for the configuration."""
        pass

    @abstractmethod
    def get_config_description(self, config: ConfigT) -> Dict[str, Any]:
        """Get a dictionary describing the configuration for saving."""
        pass

    # =========================================================================
    # Optional Methods - Can be overridden by subclasses
    # =========================================================================

    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: ConfigT,
        total_shots: int,
        aws_result: Optional[AWSJobResult] = None,
    ) -> Dict[str, Any]:
        """Evaluate the result and return algorithm-specific metrics."""
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        result: Dict[str, Any] = {
            "success_rate": s_rate,
            "success_count": s_count,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
        }

        if aws_result is not None:
            result.update(
                {
                    "dsr_michelson": aws_result.dsr_michelson,
                    "dsr_ratio": aws_result.dsr_ratio,
                    "dsr_log_ratio": aws_result.dsr_log_ratio,
                    "dsr_normalized_margin": aws_result.dsr_normalized_margin,
                    "peak_mismatch": aws_result.peak_mismatch,
                }
            )

        return result

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return []

    def get_output_dir(self) -> Path:
        """Get the output directory for saving results."""
        return Path(self.output_subdir)

    # =========================================================================
    # Core Execution Methods
    # =========================================================================

    def run(
        self,
        config_id: str,
        device_id: str = "Ankaa-3",
        region: str = "us-west-1",
        save_results: bool = True,
        wait_for_results: bool = True,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the algorithm on AWS Braket hardware."""
        config = self.get_config(config_id)

        self._print_header(config)

        print("\nCreating circuit...")
        circuit = self.create_circuit(config)
        original_depth = circuit.depth()
        original_gates = circuit.size()
        print(f"Circuit depth: {original_depth}")
        print(f"Circuit gates: {original_gates}")

        print("\nCalculating QWARD metrics on original circuit...")
        qward_metrics = calculate_qward_metrics(circuit)
        if "error" in qward_metrics:
            print(f"  Warning: QWARD metrics error: {qward_metrics['error']}")
        else:
            print("  QWARD metrics calculated successfully")

        expected_outcomes = self.get_expected_outcomes(config)
        print(f"Expected outcomes for DSR: {expected_outcomes}")

        key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        print("\nSubmitting to AWS Braket...")
        aws_result = self.executor.run_aws(
            circuit,
            device_id=device_id,
            region=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            expected_outcomes=expected_outcomes,
            timeout=self.timeout,
            poll_interval=10,
            show_progress=True,
            wait_for_results=wait_for_results,
        )

        result = self._build_rich_result(
            aws_result=aws_result,
            config=config,
            circuit=circuit,
            qward_metrics=qward_metrics,
            original_depth=original_depth,
            original_gates=original_gates,
            expected_outcomes=expected_outcomes,
        )

        self._print_analysis(result, config)

        if save_results:
            save_path = self._save_results(result, config, aws_result.device_name)
            print(f"\nResults saved to: {save_path}")

        # Always log the job ARN for later retrieval
        self._log_job(
            config_id=config_id,
            device_name=aws_result.device_name or device_id,
            region=aws_result.region or region,
            aws_result=aws_result,
            circuit_depth=original_depth,
            transpiled_depth=aws_result.circuit_depth,
        )

        return result

    def _print_header(self, config: ConfigT) -> None:
        """Print execution header."""
        config_desc = self.get_config_description(config)

        print("=" * 70)
        print(f"{self.algorithm_name} AWS BRAKET EXECUTION")
        print("=" * 70)
        print(f"Config ID: {config_desc.get('config_id', 'unknown')}")

        for key, value in config_desc.items():
            if key != "config_id":
                print(f"{key}: {value}")

        print("=" * 70)

    def _build_rich_result(
        self,
        aws_result: AWSJobResult,
        config: ConfigT,
        circuit: QuantumCircuit,
        qward_metrics: Dict[str, Any],
        original_depth: int,
        original_gates: int,
        expected_outcomes: List[str],
    ) -> Dict[str, Any]:
        """Build a rich result structure for a single AWS run."""
        config_desc = self.get_config_description(config)
        counts = aws_result.counts or {}
        total_shots = sum(counts.values()) if counts else self.shots

        evaluation = self.evaluate_result(counts, config, total_shots, aws_result)

        run_result: Dict[str, Any] = {
            "experiment_id": f"{config_desc.get('config_id', 'unknown')}_AWS-QPU_000",
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_model": "AWS-QPU",
            "job_id": aws_result.job_id,
            "timestamp": datetime.now().isoformat(),
            "num_qubits": config_desc.get("num_qubits", circuit.num_qubits),
            "circuit_depth": original_depth,
            "total_gates": original_gates,
            "transpiled_depth": aws_result.circuit_depth,
            "qward_metrics": qward_metrics,
            "backend_type": "qpu",
            "backend_name": aws_result.device_name,
            "device_name": aws_result.device_name,
            "region": aws_result.region,
            "shots": total_shots,
            "counts": counts,
            "status": self._normalize_status(aws_result.status),
            "error": aws_result.error,
            "expected_outcomes": expected_outcomes,
            "dsr_michelson": aws_result.dsr_michelson,
            "dsr_ratio": aws_result.dsr_ratio,
            "dsr_log_ratio": aws_result.dsr_log_ratio,
            "dsr_normalized_margin": aws_result.dsr_normalized_margin,
            "peak_mismatch": aws_result.peak_mismatch,
        }
        run_result.update(evaluation)

        success_rates = []
        if run_result.get("success_rate") is not None:
            success_rates.append(run_result["success_rate"])

        batch_summary: Dict[str, Any] = {
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_model": "AWS-QPU",
            "backend_name": aws_result.device_name,
            "num_runs": 1,
            "shots_per_run": self.shots,
            "backend_type": "qpu",
        }

        if success_rates:
            batch_summary.update(
                {
                    "mean_success_rate": success_rates[0],
                    "std_success_rate": 0.0,
                    "min_success_rate": success_rates[0],
                    "max_success_rate": success_rates[0],
                    "median_success_rate": success_rates[0],
                }
            )

            random_chance = self.get_random_chance(config)
            if random_chance > 0:
                mean_advantage = batch_summary["mean_success_rate"] / random_chance
                batch_summary["mean_quantum_advantage_ratio"] = mean_advantage
                batch_summary["quantum_advantage_demonstrated"] = mean_advantage > 2.0

        if run_result.get("dsr_michelson") is not None:
            batch_summary["mean_dsr_michelson"] = run_result["dsr_michelson"]
            batch_summary["mean_dsr_ratio"] = run_result.get("dsr_ratio")
            batch_summary["mean_dsr_log_ratio"] = run_result.get("dsr_log_ratio")
            batch_summary["mean_dsr_normalized_margin"] = run_result.get("dsr_normalized_margin")

        return {
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_id": "AWS-QPU",
            "algorithm": self.algorithm_name,
            "execution_type": "AWS_BRAKET",
            "saved_at": datetime.now().isoformat(),
            "device_name": aws_result.device_name,
            "region": aws_result.region,
            "status": self._normalize_status(aws_result.status),
            "config": config_desc,
            "batch_summary": batch_summary,
            "individual_results": [run_result],
        }

    def _print_analysis(self, result: Dict[str, Any], config: ConfigT) -> None:
        """Print success and DSR analysis."""
        print("\n" + "=" * 70)
        print("AWS QPU ANALYSIS")
        print("=" * 70)

        random_chance = self.get_random_chance(config)

        for run in result.get("individual_results", []):
            s_rate = run.get("success_rate", 0.0)
            advantage_ratio = run.get("advantage_ratio", 0.0)

            print(f"\nJob ID: {run.get('job_id', 'unknown')}")
            print(f"  Success rate: {s_rate:.2%}")
            print(f"  Random chance: {random_chance:.2%}")
            print(f"  Quantum advantage ratio: {advantage_ratio:.2f}x")

            dsr_michelson = run.get("dsr_michelson")
            if dsr_michelson is not None:
                print(f"  DSR Michelson: {dsr_michelson:.4f}")
                print(f"  DSR Ratio: {run.get('dsr_ratio')}")
                print(f"  DSR Log-Ratio: {run.get('dsr_log_ratio')}")
                print(f"  DSR Normalized Margin: {run.get('dsr_normalized_margin')}")
                print(f"  Peak mismatch: {run.get('peak_mismatch')}")

            thresholds = []
            for threshold in [30, 50, 70, 90]:
                if run.get(f"threshold_{threshold}", False):
                    thresholds.append(f"{threshold}%")
            print(f"  Thresholds passed: {', '.join(thresholds) if thresholds else 'None'}")

            if advantage_ratio > 2:
                print("  Status: QUANTUM ADVANTAGE DEMONSTRATED")
            elif advantage_ratio > 1:
                print("  Status: Better than random (marginal)")
            else:
                print("  Status: No advantage")

    def _save_results(
        self,
        result: Dict[str, Any],
        config: ConfigT,
        device_name: str,
    ) -> Path:
        """Save results to JSON file."""
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        config_desc = self.get_config_description(config)
        config_id = config_desc.get("config_id", "unknown")

        safe_device = (device_name or "unknown").replace("/", "-").replace(" ", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config_id}_AWS_{safe_device}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, default=str)

        return filepath

    def _log_job(
        self,
        config_id: str,
        device_name: str,
        region: str,
        aws_result: "AWSJobResult",
        circuit_depth: int = 0,
        transpiled_depth: int = 0,
        notes: str = "",
    ) -> None:
        """Append a row to the CSV job log for later retrieval.

        The log lives at ``<output_dir>/../aws_job_log.csv`` (one level above
        the JSON results folder).
        """
        import csv

        log_dir = self.get_output_dir().parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "aws_job_log.csv"

        header = [
            "timestamp", "config_id", "device", "region", "job_arn",
            "status", "shots", "circuit_depth", "transpiled_depth",
            "success_rate", "dsr_michelson", "notes",
        ]
        write_header = not log_path.exists()

        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                datetime.now().isoformat(),
                config_id,
                device_name,
                region,
                aws_result.job_id,
                aws_result.status,
                sum(aws_result.counts.values()) if aws_result.counts else self.shots,
                circuit_depth,
                transpiled_depth,
                f"{aws_result.dsr_michelson:.4f}" if aws_result.dsr_michelson is not None else "",
                f"{aws_result.dsr_ratio:.4f}" if aws_result.dsr_ratio is not None else "",
                notes,
            ])

    # =========================================================================
    # Batch Execution Methods
    # =========================================================================

    def run_batch(
        self,
        config_ids: Optional[List[str]] = None,
        device_id: str = "Ankaa-3",
        region: str = "us-west-1",
        save_results: bool = True,
        batch_timeout: int = 30,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run multiple configs in rapid-fire mode with a short per-job timeout.

        Each job is submitted and polled for at most ``batch_timeout`` seconds.
        If the job resolves in time, results are saved with full DSR/counts.
        If it doesn't, the pending result (with job ARN) is saved and the loop
        moves on to the next config.  Pending jobs can be resolved later with
        ``update_from_aws()`` or the ``--update`` CLI flag.

        Args:
            config_ids: List of config IDs to run. If ``None``, uses
                ``get_priority_configs()`` (or falls back to all config IDs).
            device_id: AWS Braket device name (default: "Ankaa-3").
            region: AWS region (default: "us-west-1").
            save_results: Whether to save each result to disk (default: True).
            batch_timeout: Max seconds to wait per job (default: 30).
            aws_access_key_id: Optional AWS access key.
            aws_secret_access_key: Optional AWS secret key.

        Returns:
            Summary dict with completed/pending/failed counts and per-config details.
        """
        if config_ids is None:
            priority = self.get_priority_configs()
            if priority:
                config_ids = [cfg["config_id"] for cfg in priority]
            else:
                config_ids = self.get_all_config_ids()

        key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        original_timeout = self.timeout
        self.timeout = batch_timeout

        completed = 0
        pending = 0
        failed = 0
        details: List[Dict[str, Any]] = []

        total = len(config_ids)
        print("=" * 70)
        print(f"{self.algorithm_name} AWS BATCH SUBMISSION")
        print("=" * 70)
        print(f"Configs to run:   {total}")
        print(f"Per-job timeout:  {batch_timeout}s")
        print(f"Device:           {device_id}")
        print(f"Region:           {region}")
        print("=" * 70)

        for idx, config_id in enumerate(config_ids, 1):
            print(f"\n[{idx}/{total}] Submitting {config_id}...")

            try:
                result = self.run(
                    config_id=config_id,
                    device_id=device_id,
                    region=region,
                    save_results=save_results,
                    wait_for_results=True,
                    aws_access_key_id=key_id,
                    aws_secret_access_key=secret,
                )

                status = result.get("status", "unknown")
                job_id = ""
                individual = result.get("individual_results", [])
                if individual:
                    job_id = individual[0].get("job_id", "")

                if status == "completed":
                    completed += 1
                    sr = result.get("batch_summary", {}).get("mean_success_rate")
                    sr_str = f"{sr:.2%}" if sr is not None else "N/A"
                    print(f"    -> COMPLETED (success rate: {sr_str})")
                elif status in {"timeout", "submitted", "queued", "running"}:
                    pending += 1
                    print(f"    -> PENDING ({status}) - will resolve later with --update")
                else:
                    failed += 1
                    print(f"    -> {status.upper()}")

                details.append(
                    {
                        "config_id": config_id,
                        "status": status,
                        "job_id": job_id,
                    }
                )

            except Exception as exc:
                failed += 1
                print(f"    -> ERROR: {exc}")
                details.append(
                    {
                        "config_id": config_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        self.timeout = original_timeout

        # Summary
        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print(f"Total:     {total}")
        print(f"Completed: {completed}")
        print(f"Pending:   {pending}")
        print(f"Failed:    {failed}")

        if pending > 0:
            print(f"\n{pending} job(s) still pending on AWS.")
            print("Retrieve results later with:")
            print(f"  python <script>.py --update")
        print("=" * 70)

        return {
            "total": total,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "details": details,
        }

    # =========================================================================
    # Data Update & Verification Methods
    # =========================================================================

    def scan_data_files(self, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Scan all AWS JSON data files and report their status."""
        data_dir = data_dir or self.get_output_dir()
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            return []

        files = sorted(data_dir.glob("*.json"))
        file_reports: List[Dict[str, Any]] = []

        for filepath in files:
            try:
                with open(filepath, encoding="utf-8") as file:
                    data = json.load(file)

                results = data.get("individual_results", [])
                report: Dict[str, Any] = {
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "config_id": data.get("config_id", "unknown"),
                    "algorithm": data.get("algorithm", "unknown"),
                    "device_name": data.get("device_name", data.get("backend_name", "unknown")),
                    "region": data.get("region", "unknown"),
                    "status": self._normalize_status(data.get("status", "unknown")),
                    "saved_at": data.get("saved_at", "unknown"),
                    "issues": [],
                }

                if not results:
                    report["issues"].append("no_results")

                jobs_info = []
                for result in results:
                    status = self._normalize_status(result.get("status", "unknown"))
                    counts = result.get("counts") or {}
                    job_info = {
                        "job_id": result.get("job_id", "unknown"),
                        "status": status,
                        "has_counts": bool(counts),
                        "success_rate": result.get("success_rate"),
                        "dsr_michelson": result.get("dsr_michelson"),
                    }

                    if not counts and status in {"completed", "done"}:
                        report["issues"].append("completed_without_counts")
                    if status in {
                        "submitted",
                        "queued",
                        "running",
                        "initializing",
                        "created",
                        "timeout",
                    }:
                        report["issues"].append(f"pending_status:{status}")
                    if result.get("dsr_michelson") is None and bool(counts):
                        report["issues"].append("missing_dsr")

                    jobs_info.append(job_info)

                report["jobs"] = jobs_info
                report["num_jobs"] = len(jobs_info)
                report["healthy"] = len(report["issues"]) == 0
                file_reports.append(report)

            except Exception as exc:
                file_reports.append(
                    {
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "healthy": False,
                        "issues": [f"parse_error: {exc}"],
                    }
                )

        return file_reports

    def print_data_status(self, data_dir: Optional[Path] = None) -> None:
        """Print a human-readable status report of all AWS data files."""
        reports = self.scan_data_files(data_dir)

        if not reports:
            print("No data files found.")
            return

        healthy_count = sum(1 for report in reports if report.get("healthy", False))
        issue_count = len(reports) - healthy_count

        print("=" * 70)
        print(f"{self.algorithm_name} AWS DATA STATUS")
        print("=" * 70)
        print(f"Total files: {len(reports)}")
        print(f"Healthy: {healthy_count}")
        print(f"With issues: {issue_count}")
        print("-" * 70)

        for report in reports:
            status_icon = "[OK]" if report.get("healthy") else "[!!]"
            print(f"  {status_icon} {report['filename']}")
            print(
                f"       Config: {report.get('config_id', '?')} | "
                f"Device: {report.get('device_name', '?')} | "
                f"Region: {report.get('region', '?')} | "
                f"Jobs: {report.get('num_jobs', 0)}"
            )

            if not report.get("healthy"):
                for issue in report.get("issues", []):
                    print(f"       ISSUE: {issue}")

            for job in report.get("jobs", []):
                counts_status = "has counts" if job["has_counts"] else "EMPTY COUNTS"
                success_rate = (
                    f"{job['success_rate']:.4f}" if job["success_rate"] is not None else "null"
                )
                dsr = f"{job['dsr_michelson']:.4f}" if job["dsr_michelson"] is not None else "null"
                print(
                    f"       Job {job['job_id']}: {job['status']} | {counts_status} | "
                    f"SR={success_rate} | DSR={dsr}"
                )

        print("=" * 70)

    def update_from_aws(
        self,
        data_dir: Optional[Path] = None,
        device_id: Optional[str] = None,
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Scan JSON files and update any with missing data from AWS Braket."""
        data_dir = data_dir or self.get_output_dir()
        reports = self.scan_data_files(data_dir)

        files_to_update = [report for report in reports if not report.get("healthy") or force]

        if not files_to_update and not force:
            print("All data files are healthy. Nothing to update.")
            return {
                "status": "healthy",
                "files_checked": len(reports),
                "files_updated": 0,
            }

        if force:
            files_to_update = reports
            print(f"Force mode: will re-check all {len(files_to_update)} files.")
        else:
            print(f"Found {len(files_to_update)} file(s) with issues.")

        if dry_run:
            print("\n[DRY RUN] Would update the following files:")
            for report in files_to_update:
                print(f"  - {report['filename']}: {report.get('issues', [])}")
            return {
                "status": "dry_run",
                "files_checked": len(reports),
                "files_to_update": len(files_to_update),
            }

        updated_count = 0
        error_count = 0
        results_summary = []

        key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        for report in files_to_update:
            filepath = Path(report["filepath"])
            print(f"\nProcessing: {report['filename']}")

            try:
                with open(filepath, encoding="utf-8") as file:
                    data = json.load(file)

                updated = self._update_file_from_aws(
                    data=data,
                    device_id=device_id,
                    region=region,
                    aws_access_key_id=key_id,
                    aws_secret_access_key=secret,
                    force=force,
                )

                if updated:
                    with open(filepath, "w", encoding="utf-8") as file:
                        json.dump(data, file, indent=2, default=str)
                    print(f"  Updated and saved: {filepath.name}")
                    updated_count += 1
                    results_summary.append({"file": filepath.name, "status": "updated"})
                else:
                    print(f"  No updates needed: {filepath.name}")
                    results_summary.append({"file": filepath.name, "status": "no_change"})

            except Exception as exc:
                print(f"  ERROR updating {filepath.name}: {exc}")
                error_count += 1
                results_summary.append(
                    {
                        "file": filepath.name,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        print("\n" + "=" * 70)
        print("UPDATE SUMMARY")
        print("=" * 70)
        print(f"Files checked: {len(reports)}")
        print(f"Files processed: {len(files_to_update)}")
        print(f"Files updated: {updated_count}")
        print(f"Errors: {error_count}")
        print("=" * 70)

        return {
            "status": "completed",
            "files_checked": len(reports),
            "files_updated": updated_count,
            "errors": error_count,
            "details": results_summary,
        }

    def _update_file_from_aws(
        self,
        data: Dict[str, Any],
        device_id: Optional[str],
        region: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        force: bool = False,
    ) -> bool:
        """Update a single file's data from AWS Braket."""
        modified = False
        results = data.get("individual_results", [])

        if not results:
            print("  No results found in file.")
            return False

        config = None
        expected_from_config: List[str] = []
        config_id = data.get("config_id", "")

        if config_id:
            try:
                config = self.get_config(config_id)
                expected_from_config = self.get_expected_outcomes(config)
            except Exception as exc:
                print(f"  Warning: could not load config '{config_id}': {exc}")

        pending_statuses = {"submitted", "queued", "running", "initializing", "created", "timeout"}

        for result in results:
            job_id = result.get("job_id", "")
            if not job_id:
                continue

            status = self._normalize_status(result.get("status", "unknown"))
            counts = result.get("counts") or {}
            needs_update = (
                force
                or status in pending_statuses
                or (status in {"completed", "done"} and not counts)
            )

            if not needs_update:
                continue

            resolved_device = (
                device_id or result.get("backend_name") or data.get("device_name") or "Ankaa-3"
            )
            resolved_region = region or result.get("region") or data.get("region") or "us-west-1"
            expected_outcomes = self._resolve_expected_outcomes(
                result=result,
                payload=data,
                fallback=expected_from_config,
            )

            print(f"  Retrieving job {job_id}...")
            aws_result = self.executor.retrieve_aws_job(
                job_id,
                device_id=resolved_device,
                region=resolved_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                expected_outcomes=expected_outcomes,
                show_progress=True,
            )

            normalized = self._normalize_status(aws_result.status)
            result["status"] = normalized
            result["backend_name"] = aws_result.device_name or resolved_device
            result["device_name"] = aws_result.device_name or resolved_device
            result["region"] = aws_result.region or resolved_region
            result["error"] = aws_result.error
            result["expected_outcomes"] = expected_outcomes
            result["dsr_michelson"] = aws_result.dsr_michelson
            result["dsr_ratio"] = aws_result.dsr_ratio
            result["dsr_log_ratio"] = aws_result.dsr_log_ratio
            result["dsr_normalized_margin"] = aws_result.dsr_normalized_margin
            result["peak_mismatch"] = aws_result.peak_mismatch

            if aws_result.circuit_depth:
                result["transpiled_depth"] = aws_result.circuit_depth

            if aws_result.counts:
                new_counts = aws_result.counts
                total_shots = sum(new_counts.values())
                result["counts"] = new_counts
                result["shots"] = total_shots

                if config is not None:
                    evaluation = self.evaluate_result(new_counts, config, total_shots, aws_result)
                    result.update(evaluation)

                success_rate_value = result.get("success_rate")
                if success_rate_value is None:
                    success_rate_value = 0.0
                print(
                    f"    Updated counts ({total_shots} shots), "
                    f"success_rate={success_rate_value:.4f}"
                )

            modified = True

        if modified:
            data["updated_at"] = datetime.now().isoformat()
            data["status"] = self._aggregate_file_status(results)
            if region:
                data["region"] = region
            if device_id:
                data["device_name"] = device_id
            self._recalculate_batch_summary(data, config)

        return modified

    @staticmethod
    def _normalize_status(status: str) -> str:
        """Normalize executor/cloud statuses to lowercase framework statuses."""
        normalized = str(status).strip().lower()
        mapping = {
            "done": "completed",
            "completed": "completed",
            "success": "completed",
            "cancelled": "failed",
            "canceled": "failed",
            "failed": "failed",
            "error": "error",
            "submitted": "submitted",
            "queued": "queued",
            "running": "running",
            "initializing": "initializing",
            "created": "created",
            "timeout": "timeout",
        }
        return mapping.get(normalized, normalized)

    @staticmethod
    def _aggregate_file_status(results: List[Dict[str, Any]]) -> str:
        """Aggregate per-job statuses into one file status."""
        statuses = [str(r.get("status", "unknown")).lower() for r in results]
        if not statuses:
            return "unknown"
        if all(status == "completed" for status in statuses):
            return "completed"
        if any(status in {"error", "failed"} for status in statuses):
            return "partial"
        if any(
            status in {"submitted", "queued", "running", "initializing", "created"}
            for status in statuses
        ):
            return "running"
        if any(status == "timeout" for status in statuses):
            return "timeout"
        return "unknown"

    def _resolve_expected_outcomes(
        self,
        result: Dict[str, Any],
        payload: Dict[str, Any],
        fallback: List[str],
    ) -> List[str]:
        """Resolve expected outcomes from result/config with fallback."""
        if result.get("expected_outcomes"):
            value = result["expected_outcomes"]
            if isinstance(value, str):
                return [v.strip() for v in value.split(",") if v.strip()]
            return list(value)
        if result.get("marked_states"):
            value = result["marked_states"]
            if isinstance(value, str):
                return [value]
            return list(value)

        config = payload.get("config", {})
        if config.get("marked_states"):
            value = config["marked_states"]
            if isinstance(value, str):
                return [value]
            return list(value)

        return list(fallback)

    def _recalculate_batch_summary(self, data: Dict[str, Any], config: Optional[ConfigT]) -> None:
        """Recalculate batch summary from individual results."""
        results = data.get("individual_results", [])
        success_rates = [
            result["success_rate"] for result in results if result.get("success_rate") is not None
        ]
        dsr_values = [
            result["dsr_michelson"] for result in results if result.get("dsr_michelson") is not None
        ]

        backend_name = data.get("device_name", data.get("backend_name", "unknown"))
        batch_summary: Dict[str, Any] = {
            "config_id": data.get("config_id", "unknown"),
            "noise_model": "AWS-QPU",
            "backend_name": backend_name,
            "num_runs": len(results),
            "shots_per_run": self.shots,
            "backend_type": "qpu",
        }

        if success_rates:
            batch_summary.update(
                {
                    "mean_success_rate": statistics.mean(success_rates),
                    "std_success_rate": (
                        statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0
                    ),
                    "min_success_rate": min(success_rates),
                    "max_success_rate": max(success_rates),
                    "median_success_rate": statistics.median(success_rates),
                }
            )

            if config is not None:
                random_chance = self.get_random_chance(config)
                if random_chance > 0:
                    mean_advantage = batch_summary["mean_success_rate"] / random_chance
                    batch_summary["mean_quantum_advantage_ratio"] = mean_advantage
                    batch_summary["quantum_advantage_demonstrated"] = mean_advantage > 2.0

            if len(success_rates) >= 2:
                analysis = calculate_statistical_analysis(
                    success_rates,
                    data.get("config_id", "unknown"),
                    "AWS-QPU",
                )
                batch_summary["analysis"] = analysis

        if dsr_values:
            batch_summary.update(
                {
                    "mean_dsr_michelson": statistics.mean(dsr_values),
                    "std_dsr_michelson": (
                        statistics.stdev(dsr_values) if len(dsr_values) > 1 else 0.0
                    ),
                    "min_dsr_michelson": min(dsr_values),
                    "max_dsr_michelson": max(dsr_values),
                    "median_dsr_michelson": statistics.median(dsr_values),
                }
            )

        data["batch_summary"] = batch_summary

    # =========================================================================
    # CLI Support
    # =========================================================================

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI usage."""
        parser = argparse.ArgumentParser(
            description=f"Run {self.algorithm_name} on AWS Braket hardware"
        )
        parser.add_argument("--config", "-c", default=None, help="Configuration ID to run")
        parser.add_argument(
            "--device",
            "-d",
            default=None,
            help="AWS Braket device name (default for runs: Ankaa-3)",
        )
        parser.add_argument(
            "--region",
            "-r",
            default=None,
            help="AWS region (default for runs: us-west-1)",
        )
        parser.add_argument(
            "--shots", "-s", type=int, default=1024, help="Number of shots (default: 1024)"
        )
        parser.add_argument(
            "--timeout", "-t", type=int, default=600, help="Timeout in seconds (default: 600)"
        )
        parser.add_argument(
            "--list", "-l", action="store_true", help="List available configurations"
        )
        parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
        parser.add_argument(
            "--no-wait",
            action="store_true",
            help="Submit job and return immediately without waiting",
        )
        parser.add_argument(
            "--aws-access-key-id",
            default=None,
            help="AWS access key id (default: AWS_ACCESS_KEY_ID env var)",
        )
        parser.add_argument(
            "--aws-secret-access-key",
            default=None,
            help="AWS secret access key (default: AWS_SECRET_ACCESS_KEY env var)",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            help="Show status of all saved experiment data files",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help="Update experiment data files with missing results from AWS",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="With --update: show what would be updated without making changes",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="With --update: re-fetch all jobs from AWS even if data looks healthy",
        )
        parser.add_argument(
            "--batch",
            action="store_true",
            help="Run all priority configs in rapid-fire mode with short per-job timeout",
        )
        parser.add_argument(
            "--batch-timeout",
            type=int,
            default=30,
            help="Per-job timeout in seconds for --batch mode (default: 30)",
        )
        parser.add_argument(
            "--batch-configs",
            nargs="+",
            default=None,
            help="Specific config IDs for --batch mode (default: all priority configs)",
        )
        return parser

    def list_configs(self) -> None:
        """Print available configurations."""
        print("=" * 70)
        print(f"AVAILABLE {self.algorithm_name} CONFIGURATIONS FOR AWS")
        print("=" * 70)

        priority_configs = self.get_priority_configs()

        if priority_configs:
            print("\nREGION 1 (Recommended - High Expected Success):")
            print("-" * 70)
            print(f"{'Config':<12} {'Qubits':<8} {'Depth':<8} {'Expected':<12} {'Description'}")
            print("-" * 70)

            for cfg in priority_configs:
                desc = cfg.get("description", "")[:30]
                print(
                    f"{cfg['config_id']:<12} {cfg['qubits']:<8} {cfg['depth']:<8} "
                    f"{cfg['expected_success']:.1%}        {desc}"
                )
        else:
            print("\nAvailable configurations:")
            for config_id in self.get_all_config_ids():
                print(f"  - {config_id}")

        print("\n" + "=" * 70)
        print("Usage: python <script>.py --config <CONFIG_ID>")
        print("=" * 70)

    def run_cli(self, args: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Run from command line arguments."""
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)

        if parsed.status:
            self.print_data_status()
            return None

        if parsed.update:
            return self.update_from_aws(
                device_id=parsed.device,
                region=parsed.region,
                aws_access_key_id=parsed.aws_access_key_id,
                aws_secret_access_key=parsed.aws_secret_access_key,
                dry_run=parsed.dry_run,
                force=parsed.force,
            )

        if parsed.batch:
            self.shots = parsed.shots
            self.executor = QuantumCircuitExecutor(shots=parsed.shots)
            return self.run_batch(
                config_ids=parsed.batch_configs,
                device_id=parsed.device or "Ankaa-3",
                region=parsed.region or "us-west-1",
                save_results=not parsed.no_save,
                batch_timeout=parsed.batch_timeout,
                aws_access_key_id=parsed.aws_access_key_id,
                aws_secret_access_key=parsed.aws_secret_access_key,
            )

        if parsed.list:
            self.list_configs()
            return None

        if parsed.config is None:
            priority = self.get_priority_configs()
            if priority:
                parsed.config = priority[0]["config_id"]
            else:
                configs = self.get_all_config_ids()
                if configs:
                    parsed.config = configs[0]
                else:
                    print("Error: No configurations available")
                    return None

        if parsed.config not in self.get_all_config_ids():
            print(f"Error: Unknown config '{parsed.config}'")
            print(f"Available configs: {self.get_all_config_ids()}")
            return None

        self.shots = parsed.shots
        self.timeout = parsed.timeout
        self.executor = QuantumCircuitExecutor(shots=parsed.shots)

        result = self.run(
            config_id=parsed.config,
            device_id=parsed.device or "Ankaa-3",
            region=parsed.region or "us-west-1",
            save_results=not parsed.no_save,
            wait_for_results=not parsed.no_wait,
            aws_access_key_id=parsed.aws_access_key_id,
            aws_secret_access_key=parsed.aws_secret_access_key,
        )

        print("\n" + "=" * 70)
        if result.get("status") == "completed":
            print("EXECUTION COMPLETE")
            summary = result.get("batch_summary", {})
            if summary.get("mean_success_rate") is not None:
                print(f"Mean success rate: {summary['mean_success_rate']:.2%}")
                print(f"Quantum advantage: {summary.get('mean_quantum_advantage_ratio', 0):.2f}x")
            if summary.get("mean_dsr_michelson") is not None:
                print(f"Mean DSR Michelson: {summary['mean_dsr_michelson']:.4f}")
        else:
            print(f"EXECUTION STATUS: {result.get('status', 'unknown')}")
        print("=" * 70)

        return result
