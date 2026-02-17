#!/usr/bin/env python3
"""
IBM QPU Experiment Base Module

This module provides reusable infrastructure for running quantum algorithms
on IBM Quantum hardware with rich data output similar to simulator experiments.

QWARD metrics are calculated on the original circuit BEFORE transpilation/execution,
as the transpiled circuit will be different from what we designed.

Subclass `IBMExperimentBase` to create algorithm-specific implementations
(e.g., Grover, QFT, Phase Estimation).
"""

import argparse
import json
import statistics
import glob as glob_module
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic

from qiskit import QuantumCircuit

from qward.algorithms import QuantumCircuitExecutor, IBMBatchResult
from qward.examples.papers.experiment_helpers import (
    calculate_qward_metrics,
    calculate_statistical_analysis,
)

# IBM Quantum imports (optional)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService

    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False


# Type variable for config classes
ConfigT = TypeVar("ConfigT")


@dataclass
class IBMExperimentConfig:
    """Base configuration for IBM experiments."""

    config_id: str
    num_qubits: int
    expected_success: float
    circuit_depth: int
    region: str = "Region 1"


class IBMExperimentBase(ABC, Generic[ConfigT]):
    """Base class for IBM QPU experiments.

    Provides common infrastructure for:
    - Circuit execution on IBM QPU
    - QWARD metrics calculation (on original circuit before transpilation)
    - Rich result building with statistical analysis
    - Result saving
    - Command-line interface

    Subclass and implement the abstract methods for your specific algorithm.
    """

    def __init__(
        self,
        shots: int = 1024,
        timeout: int = 600,
        output_subdir: str = "data/qpu/raw",
    ):
        """Initialize the experiment base.

        Args:
            shots: Number of shots per circuit execution
            timeout: Maximum wait time for job completion (seconds)
            output_subdir: Subdirectory for saving results (relative to algorithm folder)
        """
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
    ) -> Dict[str, Any]:
        """Evaluate the result and return algorithm-specific metrics.

        Override this method to add algorithm-specific evaluation.
        """
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0

        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0

        return {
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

    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return []

    def get_output_dir(self) -> Path:
        """Get the output directory for saving results."""
        return Path(self.output_subdir)

    def get_timed_out_batches(self) -> List[Dict[str, Any]]:
        """Return a list of known timed-out batches for recovery.

        Override in subclasses to provide batch IDs from timed-out experiments.
        Each entry should be a dict with keys:
            - batch_id: str (IBM batch/session ID)
            - config_id: str (experiment config ID)
            - backend_name: str (IBM backend name)
            - job_ids: Optional[List[str]] (known job IDs)
            - transpiled_depths: Optional[Dict[int, int]] (opt_level -> depth)
            - original_depth: Optional[int] (original circuit depth)

        Returns:
            List of batch info dicts
        """
        return []

    # =========================================================================
    # Core Execution Methods
    # =========================================================================

    def run(
        self,
        config_id: str,
        backend_name: Optional[str] = None,
        optimization_levels: Optional[List[int]] = None,
        save_results: bool = True,
        channel: Optional[str] = None,
        token: Optional[str] = None,
        instance: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the algorithm on IBM Quantum hardware.

        QWARD metrics are calculated on the original circuit BEFORE execution.
        """
        # Get configuration
        config = self.get_config(config_id)

        # Print header
        self._print_header(config)

        # Create circuit
        print("\nCreating circuit...")
        circuit = self.create_circuit(config)
        original_depth = circuit.depth()
        original_gates = circuit.size()
        print(f"Circuit depth: {original_depth}")
        print(f"Circuit gates: {original_gates}")

        # Calculate QWARD metrics on ORIGINAL circuit (before transpilation)
        print("\nCalculating QWARD metrics on original circuit...")
        qward_metrics = calculate_qward_metrics(circuit)
        if "error" in qward_metrics:
            print(f"  Warning: QWARD metrics error: {qward_metrics['error']}")
        else:
            print(f"  QWARD metrics calculated successfully")

        # Create success criteria
        success_criteria = self.create_success_criteria(config)

        # Set default optimization levels (all 4 levels: 0, 1, 2, 3)
        if optimization_levels is None:
            optimization_levels = [0, 1, 2, 3]

        # Execute on IBM QPU
        print("\nSubmitting to IBM Quantum...")
        ibm_result = self.executor.run_ibm(
            circuit,
            backend_name=backend_name,
            optimization_levels=optimization_levels,
            success_criteria=success_criteria,
            timeout=self.timeout,
            poll_interval=10,
            show_progress=True,
            channel=channel,
            token=token,
            instance=instance,
        )

        # Build rich result (pass pre-calculated QWARD metrics)
        result = self._build_rich_result(
            ibm_result, config, circuit, qward_metrics, original_depth, original_gates
        )

        # Print analysis
        self._print_analysis(result, config)

        # Save results
        if save_results and ibm_result.status == "completed":
            save_path = self._save_results(result, config, ibm_result.backend_name)
            print(f"\nResults saved to: {save_path}")

        return result

    def _print_header(self, config: ConfigT) -> None:
        """Print execution header."""
        config_desc = self.get_config_description(config)

        print("=" * 70)
        print(f"{self.algorithm_name} IBM QPU EXECUTION")
        print("=" * 70)
        print(f"Config ID: {config_desc.get('config_id', 'unknown')}")

        for key, value in config_desc.items():
            if key != "config_id":
                print(f"{key}: {value}")

        print("=" * 70)

    def _build_rich_result(
        self,
        ibm_result: IBMBatchResult,
        config: ConfigT,
        circuit: QuantumCircuit,
        qward_metrics: Dict[str, Any],
        original_depth: int,
        original_gates: int,
    ) -> Dict[str, Any]:
        """Build rich result structure with pre-calculated QWARD metrics."""
        individual_results = []
        success_rates = []

        config_desc = self.get_config_description(config)

        for job in ibm_result.jobs:
            # Get counts and evaluate
            counts = job.counts or {}
            total_shots = sum(counts.values()) if counts else self.shots

            # Evaluate result (algorithm-specific)
            evaluation = self.evaluate_result(counts, config, total_shots)
            success_rates.append(evaluation.get("success_rate", 0.0))

            # Build individual result
            run_result = {
                "experiment_id": f"{config_desc.get('config_id', 'unknown')}_IBM-QPU_{job.optimization_level:03d}",
                "config_id": config_desc.get("config_id", "unknown"),
                "noise_model": "IBM-QPU",
                "optimization_level": job.optimization_level,
                "job_id": job.job_id,
                "timestamp": datetime.now().isoformat(),
                "num_qubits": config_desc.get("num_qubits", circuit.num_qubits),
                # Original circuit metrics (what we designed)
                "circuit_depth": original_depth,
                "total_gates": original_gates,
                # Transpiled circuit metrics (what actually ran)
                "transpiled_depth": job.circuit_depth,
                # QWARD metrics (from original circuit)
                "qward_metrics": qward_metrics,
                "backend_type": "qpu",
                "backend_name": ibm_result.backend_name,
                "shots": total_shots,
                "counts": counts,
                "status": job.status,
                "error": job.error,
            }

            # Add evaluation metrics
            run_result.update(evaluation)

            individual_results.append(run_result)

        # Build batch summary with statistical analysis
        batch_summary = {
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_model": "IBM-QPU",
            "backend_name": ibm_result.backend_name,
            "num_runs": len(individual_results),
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

            random_chance = self.get_random_chance(config)
            if random_chance > 0:
                mean_advantage = batch_summary["mean_success_rate"] / random_chance
                batch_summary["mean_quantum_advantage_ratio"] = mean_advantage
                batch_summary["quantum_advantage_demonstrated"] = mean_advantage > 2.0

            # Add statistical analysis
            if len(success_rates) >= 2:
                analysis = calculate_statistical_analysis(
                    success_rates, config_desc.get("config_id", "unknown"), "IBM-QPU"
                )
                batch_summary["analysis"] = analysis

        # Build full result
        result = {
            "config_id": config_desc.get("config_id", "unknown"),
            "noise_id": "IBM-QPU",
            "algorithm": self.algorithm_name,
            "execution_type": "IBM_QPU",
            "saved_at": datetime.now().isoformat(),
            "batch_id": ibm_result.batch_id,
            "backend_name": ibm_result.backend_name,
            "status": ibm_result.status,
            "config": config_desc,
            "batch_summary": batch_summary,
            "individual_results": individual_results,
        }

        return result

    def _print_analysis(self, result: Dict[str, Any], config: ConfigT) -> None:
        """Print quantum advantage analysis."""
        print("\n" + "=" * 70)
        print("QUANTUM ADVANTAGE ANALYSIS")
        print("=" * 70)

        random_chance = self.get_random_chance(config)

        for run in result.get("individual_results", []):
            s_rate = run.get("success_rate", 0)
            opt_level = run.get("optimization_level", "?")
            advantage_ratio = run.get("advantage_ratio", 0)

            print(f"\nOptimization Level {opt_level}:")
            print(f"  Success rate: {s_rate:.2%}")
            print(f"  Random chance: {random_chance:.2%}")
            print(f"  Quantum advantage ratio: {advantage_ratio:.2f}x")
            print(
                f"  Original depth: {run.get('circuit_depth', '?')}, Transpiled: {run.get('transpiled_depth', '?')}"
            )

            thresholds = []
            for t in [30, 50, 70, 90]:
                if run.get(f"threshold_{t}", False):
                    thresholds.append(f"{t}%")
            print(f"  Thresholds passed: {', '.join(thresholds) if thresholds else 'None'}")

            if advantage_ratio > 2:
                print("  Status: QUANTUM ADVANTAGE DEMONSTRATED")
            elif advantage_ratio > 1:
                print("  Status: Better than random (marginal)")
            else:
                print("  Status: No advantage")

        # Summary
        summary = result.get("batch_summary", {})
        if summary.get("mean_success_rate") is not None:
            print("\n--- SUMMARY ---")
            print(f"Mean success rate: {summary['mean_success_rate']:.2%}")
            print(f"Mean quantum advantage: {summary.get('mean_quantum_advantage_ratio', 0):.2f}x")
            if summary.get("quantum_advantage_demonstrated"):
                print("Result: QUANTUM ADVANTAGE DEMONSTRATED")
            else:
                print("Result: No consistent quantum advantage")

    def _save_results(
        self,
        result: Dict[str, Any],
        config: ConfigT,
        backend_name: str,
    ) -> Path:
        """Save results to JSON file."""
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        config_desc = self.get_config_description(config)
        config_id = config_desc.get("config_id", "unknown")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config_id}_IBM_{backend_name}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return filepath

    # =========================================================================
    # Data Update & Verification Methods
    # =========================================================================

    def scan_data_files(self, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Scan all JSON data files and report their status.

        Args:
            data_dir: Directory to scan (default: output directory)

        Returns:
            List of file status dictionaries
        """
        data_dir = data_dir or self.get_output_dir()
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            return []

        files = sorted(data_dir.glob("*.json"))
        file_reports = []

        for filepath in files:
            try:
                with open(filepath) as f:
                    data = json.load(f)

                report = {
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "config_id": data.get("config_id", "unknown"),
                    "algorithm": data.get("algorithm", "unknown"),
                    "batch_id": data.get("batch_id", ""),
                    "backend_name": data.get("backend_name", "unknown"),
                    "status": data.get("status", "unknown"),
                    "saved_at": data.get("saved_at", "unknown"),
                    "format": "new" if "individual_results" in data else "old",
                    "issues": [],
                }

                # Check format
                if "individual_results" in data:
                    results = data["individual_results"]
                elif "jobs" in data:
                    results = data["jobs"]
                    report["issues"].append("old_format")
                else:
                    results = []
                    report["issues"].append("no_results")

                # Check each job/result
                jobs_info = []
                for i, result in enumerate(results):
                    job_info = {
                        "job_id": result.get("job_id", "unknown"),
                        "optimization_level": result.get("optimization_level", i),
                        "status": result.get("status", "unknown"),
                        "has_counts": bool(result.get("counts")),
                        "success_rate": result.get("success_rate"),
                    }

                    # Check for empty counts
                    counts = result.get("counts", {})
                    if not counts and result.get("status") == "DONE":
                        report["issues"].append(
                            f"empty_counts_opt{result.get('optimization_level', i)}"
                        )
                    if result.get("success_rate") is None and result.get("status") == "DONE":
                        report["issues"].append(
                            f"null_success_rate_opt{result.get('optimization_level', i)}"
                        )

                    jobs_info.append(job_info)

                report["jobs"] = jobs_info
                report["num_jobs"] = len(jobs_info)
                report["healthy"] = len(report["issues"]) == 0

                file_reports.append(report)

            except Exception as e:
                file_reports.append(
                    {
                        "filepath": str(filepath),
                        "filename": filepath.name,
                        "healthy": False,
                        "issues": [f"parse_error: {e}"],
                    }
                )

        return file_reports

    def print_data_status(self, data_dir: Optional[Path] = None) -> None:
        """Print a human-readable status report of all data files.

        Args:
            data_dir: Directory to scan (default: output directory)
        """
        reports = self.scan_data_files(data_dir)

        if not reports:
            print("No data files found.")
            return

        healthy_count = sum(1 for r in reports if r.get("healthy", False))
        issue_count = len(reports) - healthy_count

        print("=" * 70)
        print(f"{self.algorithm_name} IBM QPU DATA STATUS")
        print("=" * 70)
        print(f"Total files: {len(reports)}")
        print(f"Healthy: {healthy_count}")
        print(f"With issues: {issue_count}")
        print("-" * 70)

        for report in reports:
            status_icon = "[OK]" if report.get("healthy") else "[!!]"
            config_id = report.get("config_id", "?")
            backend = report.get("backend_name", "?")
            num_jobs = report.get("num_jobs", 0)
            batch_id = report.get("batch_id", "?")[:12]

            print(f"  {status_icon} {report['filename']}")
            print(
                f"       Config: {config_id} | Backend: {backend} | "
                f"Jobs: {num_jobs} | Batch: {batch_id}..."
            )

            if not report.get("healthy"):
                for issue in report.get("issues", []):
                    print(f"       ISSUE: {issue}")

            # Print job details
            for job in report.get("jobs", []):
                counts_status = "has counts" if job["has_counts"] else "EMPTY COUNTS"
                sr = f"{job['success_rate']:.4f}" if job["success_rate"] is not None else "null"
                print(
                    f"       Job opt={job['optimization_level']}: "
                    f"{job['status']} | {counts_status} | SR={sr} | "
                    f"ID={job['job_id']}"
                )

        print("=" * 70)

    def update_from_ibm(
        self,
        data_dir: Optional[Path] = None,
        channel: Optional[str] = None,
        token: Optional[str] = None,
        instance: Optional[str] = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Scan JSON files and update any with missing data from IBM Cloud.

        Connects to IBM Quantum service, retrieves job results for any
        experiments with missing histogram counts, and updates the JSON files.

        Handles both old format ("jobs" key) and new format ("individual_results" key).

        Args:
            data_dir: Directory to scan (default: output directory)
            channel: IBM Quantum channel
            token: IBM Quantum API token
            instance: IBM Quantum instance
            dry_run: If True, only report issues without fixing
            force: If True, re-fetch all jobs regardless of status

        Returns:
            Summary dictionary with update results
        """
        if not IBM_QUANTUM_AVAILABLE:
            raise ImportError(
                "IBM Quantum Runtime is not available. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        data_dir = data_dir or self.get_output_dir()
        reports = self.scan_data_files(data_dir)

        # Find files that need updating
        files_to_update = [r for r in reports if not r.get("healthy") or force]

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

        # Connect to IBM Quantum service
        print("\nConnecting to IBM Quantum service...")
        service_kwargs = {}
        if channel:
            service_kwargs["channel"] = channel
        if token:
            service_kwargs["token"] = token
        if instance:
            service_kwargs["instance"] = instance

        service = QiskitRuntimeService(**service_kwargs)
        print("Connected successfully.")

        updated_count = 0
        error_count = 0
        results_summary = []

        for report in files_to_update:
            filepath = Path(report["filepath"])
            print(f"\nProcessing: {report['filename']}")

            try:
                with open(filepath) as f:
                    data = json.load(f)

                updated = self._update_file_from_ibm(data, service, report, force)

                if updated:
                    # Save updated data
                    with open(filepath, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                    print(f"  Updated and saved: {filepath.name}")
                    updated_count += 1
                    results_summary.append({"file": filepath.name, "status": "updated"})
                else:
                    print(f"  No updates needed: {filepath.name}")
                    results_summary.append({"file": filepath.name, "status": "no_change"})

            except Exception as e:
                print(f"  ERROR updating {filepath.name}: {e}")
                error_count += 1
                results_summary.append({"file": filepath.name, "status": "error", "error": str(e)})

        # Print summary
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

    def _update_file_from_ibm(
        self,
        data: Dict[str, Any],
        service: "QiskitRuntimeService",
        report: Dict[str, Any],
        force: bool = False,
    ) -> bool:
        """Update a single file's data from IBM Cloud.

        Args:
            data: The loaded JSON data (will be modified in place)
            service: Connected QiskitRuntimeService
            report: File status report from scan_data_files
            force: If True, re-fetch all jobs

        Returns:
            True if data was modified
        """
        modified = False
        batch_id = data.get("batch_id", "")

        # Determine which key holds the results
        is_old_format = "jobs" in data and "individual_results" not in data
        results_key = "jobs" if is_old_format else "individual_results"
        results_list = data.get(results_key, [])

        if not results_list:
            print("  No jobs/results found in file.")
            return False

        # Fetch jobs from IBM using batch_id (session_id)
        print(f"  Fetching jobs for batch: {batch_id}")
        try:
            ibm_jobs = service.jobs(session_id=batch_id, limit=100)
            print(f"  Found {len(ibm_jobs)} jobs in batch")
        except Exception as e:
            print(f"  Warning: Could not fetch batch jobs: {e}")
            print("  Falling back to individual job retrieval...")
            ibm_jobs = None

        # Build a map of job_id -> IBM job object
        ibm_job_map = {}
        if ibm_jobs:
            for ibm_job in ibm_jobs:
                ibm_job_map[ibm_job.job_id()] = ibm_job

        # Process each result entry
        for i, result in enumerate(results_list):
            job_id = result.get("job_id", "")
            counts = result.get("counts", {})
            status = result.get("status", "")

            needs_update = force or (not counts and status == "DONE")

            if not needs_update:
                continue

            print(f"  Updating job {job_id} (opt_level={result.get('optimization_level', i)})...")

            # Get the IBM job
            ibm_job = ibm_job_map.get(job_id)

            if ibm_job is None:
                # Try fetching individually
                try:
                    ibm_job = service.job(job_id)
                except Exception as e:
                    print(f"    Could not retrieve job {job_id}: {e}")
                    continue

            # Check job status
            try:
                job_status = str(ibm_job.status())
                print(f"    IBM job status: {job_status}")
                result["status"] = job_status.replace("JobStatus.", "")

                if "DONE" in job_status:
                    # Extract counts
                    try:
                        primitive_result = ibm_job.result()
                        new_counts = self._extract_counts_from_primitive_result(primitive_result)
                        if new_counts:
                            result["counts"] = new_counts
                            total_shots = sum(new_counts.values())
                            result["shots"] = total_shots

                            # Recalculate success rate if we have config
                            config_id = data.get("config_id", "")
                            if config_id:
                                try:
                                    config = self.get_config(config_id)
                                    evaluation = self.evaluate_result(
                                        new_counts, config, total_shots
                                    )
                                    result.update(evaluation)
                                    print(
                                        f"    Updated counts ({total_shots} shots), "
                                        f"success_rate={evaluation.get('success_rate', 0):.4f}"
                                    )
                                except Exception as eval_e:
                                    # Fallback: just calculate basic success rate
                                    print(f"    Warning: Could not evaluate: {eval_e}")
                                    result["success_rate"] = None

                            modified = True
                        else:
                            print("    Warning: Could not extract counts from result")

                    except Exception as e:
                        print(f"    Error extracting results: {e}")
                        result["error"] = str(e)
                        modified = True

                elif "CANCELLED" in job_status or "ERROR" in job_status:
                    result["error"] = f"Job {job_status}"
                    modified = True

            except Exception as e:
                print(f"    Error checking job status: {e}")

        # If old format, convert to new format
        if is_old_format and modified:
            print("  Converting from old format to new format...")
            data["individual_results"] = self._convert_old_to_new_format(data, results_list)
            if "jobs" in data:
                del data["jobs"]
            data["noise_id"] = "IBM-QPU"
            data["updated_at"] = datetime.now().isoformat()
            modified = True
        elif modified:
            data["updated_at"] = datetime.now().isoformat()

        # Recalculate batch_summary if modified
        if modified and "individual_results" in data:
            self._recalculate_batch_summary(data)

        return modified

    def _extract_counts_from_primitive_result(self, primitive_result) -> Dict[str, int]:
        """Extract measurement counts from a PrimitiveResult (SamplerV2 format).

        Args:
            primitive_result: PrimitiveResult object from IBM Runtime

        Returns:
            Dictionary of measurement outcomes and their counts
        """
        try:
            pub_result = primitive_result[0]

            # Try to find the classical register data
            bit_array = None
            for attr in ["c", "meas", "cr"]:
                if hasattr(pub_result.data, attr):
                    bit_array = getattr(pub_result.data, attr)
                    break

            if bit_array is None:
                data_attrs = [a for a in dir(pub_result.data) if not a.startswith("_")]
                for attr in data_attrs:
                    obj = getattr(pub_result.data, attr)
                    if hasattr(obj, "get_counts") or hasattr(obj, "num_shots"):
                        bit_array = obj
                        break

            if bit_array is None:
                return {}

            # Method 1: Use get_counts() if available (preferred)
            if hasattr(bit_array, "get_counts"):
                return dict(bit_array.get_counts())

            # Method 2: Use get_bitstrings() if available
            if hasattr(bit_array, "get_bitstrings"):
                bitstrings = bit_array.get_bitstrings()
                return dict(Counter(bitstrings))

            # Method 3: Manual extraction from array
            num_shots = bit_array.num_shots
            bit_strings = []

            if hasattr(bit_array, "array"):
                arr = bit_array.array
                num_bits = bit_array.num_bits
                for row in arr:
                    bit_string = "".join(str(int(b)) for b in row[:num_bits])
                    bit_strings.append(bit_string)
            else:
                for shot in range(num_shots):
                    try:
                        bits = bit_array[shot]
                        if hasattr(bits, "__iter__") and not isinstance(bits, str):
                            bit_string = "".join(str(int(b)) for b in bits)
                        else:
                            bit_string = str(bits)
                        bit_strings.append(bit_string)
                    except Exception:
                        continue

            return dict(Counter(bit_strings))

        except Exception as e:
            print(f"    Warning: Could not extract counts: {e}")
            return {}

    def _convert_old_to_new_format(
        self, data: Dict[str, Any], old_jobs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert old 'jobs' format entries to new 'individual_results' format.

        Args:
            data: The full file data
            old_jobs: List of old-format job entries

        Returns:
            List of new-format individual result entries
        """
        config_id = data.get("config_id", "unknown")
        backend_name = data.get("backend_name", "unknown")

        # Try to get QWARD metrics from original circuit
        qward_metrics = {}
        try:
            config = self.get_config(config_id)
            circuit = self.create_circuit(config)
            qward_metrics = calculate_qward_metrics(circuit)
        except Exception as e:
            qward_metrics = {"error": str(e)}

        individual_results = []
        for job in old_jobs:
            opt_level = job.get("optimization_level", 0)
            counts = job.get("counts", {})
            total_shots = sum(counts.values()) if counts else self.shots

            new_result = {
                "experiment_id": f"{config_id}_IBM-QPU_{opt_level:03d}",
                "config_id": config_id,
                "noise_model": "IBM-QPU",
                "optimization_level": opt_level,
                "job_id": job.get("job_id", ""),
                "timestamp": data.get("saved_at", datetime.now().isoformat()),
                "num_qubits": data.get("config", {}).get("num_qubits", 0),
                "circuit_depth": 0,  # Original depth unknown in old format
                "total_gates": 0,
                "transpiled_depth": job.get("circuit_depth", 0),
                "qward_metrics": qward_metrics,
                "backend_type": "qpu",
                "backend_name": backend_name,
                "shots": total_shots,
                "counts": counts,
                "status": job.get("status", "unknown"),
                "error": job.get("error"),
            }

            # Evaluate results if counts are available
            if counts:
                try:
                    config = self.get_config(config_id)
                    evaluation = self.evaluate_result(counts, config, total_shots)
                    new_result.update(evaluation)
                except Exception:
                    new_result["success_rate"] = job.get("success_rate")
            else:
                new_result["success_rate"] = job.get("success_rate")

            individual_results.append(new_result)

        return individual_results

    def _recalculate_batch_summary(self, data: Dict[str, Any]) -> None:
        """Recalculate batch_summary from individual_results.

        Args:
            data: The full file data (modified in place)
        """
        results = data.get("individual_results", [])
        success_rates = [r["success_rate"] for r in results if r.get("success_rate") is not None]

        config_id = data.get("config_id", "unknown")
        backend_name = data.get("backend_name", "unknown")

        batch_summary = {
            "config_id": config_id,
            "noise_model": "IBM-QPU",
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

            # Calculate quantum advantage
            try:
                config = self.get_config(config_id)
                random_chance = self.get_random_chance(config)
                if random_chance > 0:
                    mean_advantage = batch_summary["mean_success_rate"] / random_chance
                    batch_summary["mean_quantum_advantage_ratio"] = mean_advantage
                    batch_summary["quantum_advantage_demonstrated"] = mean_advantage > 2.0
            except Exception:
                pass

            # Statistical analysis
            if len(success_rates) >= 2:
                analysis = calculate_statistical_analysis(success_rates, config_id, "IBM-QPU")
                batch_summary["analysis"] = analysis

        data["batch_summary"] = batch_summary

    # =========================================================================
    # Batch Recovery (for timed-out experiments without JSON files)
    # =========================================================================

    def recover_batch(
        self,
        batch_id: str,
        config_id: str,
        backend_name: str,
        service: "QiskitRuntimeService",
        job_ids: Optional[List[str]] = None,
        transpiled_depths: Optional[Dict[int, int]] = None,
        original_depth: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Recover a timed-out batch by fetching results from IBM Cloud.

        This creates a complete result structure from a batch that timed out
        during execution and never saved a JSON file.

        Args:
            batch_id: The IBM batch/session ID
            config_id: The experiment configuration ID (e.g., 'M3-1', 'SR8')
            backend_name: IBM backend name (e.g., 'ibm_fez')
            service: Connected QiskitRuntimeService
            job_ids: Optional list of known job IDs (opt0, opt1, opt2, opt3)
            transpiled_depths: Optional dict mapping opt_level -> transpiled depth
            original_depth: Optional original circuit depth

        Returns:
            Complete result dict (same structure as _build_rich_result), or None on failure
        """
        print(f"\n{'='*70}")
        print(f"RECOVERING BATCH: {config_id}")
        print(f"{'='*70}")
        print(f"Batch ID: {batch_id}")
        print(f"Backend: {backend_name}")

        # Get config and create circuit
        try:
            config = self.get_config(config_id)
            circuit = self.create_circuit(config)
            config_desc = self.get_config_description(config)
            actual_depth = original_depth or circuit.depth()
            actual_gates = circuit.size()
        except Exception as e:
            print(f"Error: Could not get config '{config_id}': {e}")
            return None

        # Calculate QWARD metrics
        print("Calculating QWARD metrics on original circuit...")
        qward_metrics = calculate_qward_metrics(circuit)
        if "error" in qward_metrics:
            print(f"  Warning: QWARD metrics error: {qward_metrics['error']}")
        else:
            print("  QWARD metrics calculated successfully")

        # Fetch jobs from IBM
        print(f"\nFetching jobs from batch {batch_id}...")
        ibm_jobs = []
        try:
            ibm_jobs = service.jobs(session_id=batch_id, limit=100)
            print(f"Found {len(ibm_jobs)} jobs in batch")
        except Exception as e:
            print(f"Warning: Could not fetch batch jobs: {e}")

        # If batch fetch failed but we have job_ids, try individually
        if not ibm_jobs and job_ids:
            print("Falling back to individual job retrieval...")
            for jid in job_ids:
                try:
                    job = service.job(jid)
                    ibm_jobs.append(job)
                except Exception as e:
                    print(f"  Could not retrieve job {jid}: {e}")

        if not ibm_jobs:
            print("ERROR: No jobs could be retrieved from IBM Cloud.")
            return None

        # Sort jobs by creation date to match optimization levels
        # IBM jobs from a batch are typically in order: opt0, opt1, opt2, opt3
        try:
            ibm_jobs_sorted = sorted(ibm_jobs, key=lambda j: j.creation_date or "")
        except Exception:
            ibm_jobs_sorted = ibm_jobs

        # Build individual results
        individual_results = []
        success_rates = []
        random_chance = self.get_random_chance(config)
        all_completed = True

        for i, ibm_job in enumerate(ibm_jobs_sorted):
            opt_level = i  # Assumes jobs were submitted in order 0, 1, 2, 3
            jid = ibm_job.job_id()
            status_str = str(ibm_job.status()).replace("JobStatus.", "")
            print(f"\n  Job {jid} (opt_level={opt_level}): {status_str}")

            t_depth = (transpiled_depths or {}).get(opt_level, 0)

            counts = {}
            total_shots = self.shots
            evaluation = {}

            if "DONE" in status_str:
                try:
                    primitive_result = ibm_job.result()
                    counts = self._extract_counts_from_primitive_result(primitive_result)
                    if counts:
                        total_shots = sum(counts.values())
                        evaluation = self.evaluate_result(counts, config, total_shots)
                        success_rates.append(evaluation.get("success_rate", 0.0))
                        print(
                            f"    Extracted {total_shots} shots, "
                            f"success_rate={evaluation.get('success_rate', 0):.4f}"
                        )
                    else:
                        print("    Warning: Could not extract counts from result")
                except Exception as e:
                    print(f"    Error extracting results: {e}")
            else:
                all_completed = False
                print(f"    Job not completed (status: {status_str})")

            run_result = {
                "experiment_id": f"{config_id}_IBM-QPU_{opt_level:03d}",
                "config_id": config_id,
                "noise_model": "IBM-QPU",
                "optimization_level": opt_level,
                "job_id": jid,
                "timestamp": datetime.now().isoformat(),
                "num_qubits": config_desc.get("num_qubits", circuit.num_qubits),
                "circuit_depth": actual_depth,
                "total_gates": actual_gates,
                "transpiled_depth": t_depth,
                "qward_metrics": qward_metrics,
                "backend_type": "qpu",
                "backend_name": backend_name,
                "shots": total_shots,
                "counts": counts,
                "status": status_str,
                "error": None,
            }
            run_result.update(evaluation)
            individual_results.append(run_result)

        # Build batch summary
        batch_summary = {
            "config_id": config_id,
            "noise_model": "IBM-QPU",
            "backend_name": backend_name,
            "num_runs": len(individual_results),
            "shots_per_run": self.shots,
            "backend_type": "qpu",
        }

        valid_rates = [r for r in success_rates if r is not None]
        if valid_rates:
            batch_summary.update(
                {
                    "mean_success_rate": statistics.mean(valid_rates),
                    "std_success_rate": (
                        statistics.stdev(valid_rates) if len(valid_rates) > 1 else 0.0
                    ),
                    "min_success_rate": min(valid_rates),
                    "max_success_rate": max(valid_rates),
                    "median_success_rate": statistics.median(valid_rates),
                }
            )

            if random_chance > 0:
                mean_advantage = batch_summary["mean_success_rate"] / random_chance
                batch_summary["mean_quantum_advantage_ratio"] = mean_advantage
                batch_summary["quantum_advantage_demonstrated"] = mean_advantage > 2.0

            if len(valid_rates) >= 2:
                analysis = calculate_statistical_analysis(valid_rates, config_id, "IBM-QPU")
                batch_summary["analysis"] = analysis

        # Build complete result
        overall_status = "completed" if all_completed else "partial"
        result = {
            "config_id": config_id,
            "noise_id": "IBM-QPU",
            "algorithm": self.algorithm_name,
            "execution_type": "IBM_QPU",
            "saved_at": datetime.now().isoformat(),
            "recovered_at": datetime.now().isoformat(),
            "batch_id": batch_id,
            "backend_name": backend_name,
            "status": overall_status,
            "config": config_desc,
            "batch_summary": batch_summary,
            "individual_results": individual_results,
        }

        # Print summary
        print(f"\n{'='*70}")
        print(f"RECOVERY SUMMARY for {config_id}")
        print(f"{'='*70}")
        print(f"Status: {overall_status}")
        print(f"Jobs recovered: {len(individual_results)}")
        if valid_rates:
            print(f"Mean success rate: {statistics.mean(valid_rates):.4f}")
        print(f"{'='*70}")

        return result

    def recover_batches(
        self,
        batches: List[Dict[str, Any]],
        channel: Optional[str] = None,
        token: Optional[str] = None,
        instance: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Recover multiple timed-out batches from IBM Cloud.

        Args:
            batches: List of dicts with keys: batch_id, config_id, backend_name,
                     and optionally: job_ids, transpiled_depths, original_depth
            channel: IBM Quantum channel
            token: IBM Quantum API token
            instance: IBM Quantum instance
            dry_run: If True, only report what would be recovered

        Returns:
            Summary dictionary with recovery results
        """
        if not IBM_QUANTUM_AVAILABLE:
            raise ImportError(
                "IBM Quantum Runtime is not available. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        if not batches:
            print("No batches to recover.")
            return {"status": "nothing_to_do", "recovered": 0}

        print(f"\n{'='*70}")
        print(f"BATCH RECOVERY: {len(batches)} batch(es) to recover")
        print(f"{'='*70}")

        for b in batches:
            print(f"  {b['config_id']}: batch={b['batch_id'][:16]}... ({b['backend_name']})")

        if dry_run:
            print("\n[DRY RUN] Would recover the above batches.")
            return {
                "status": "dry_run",
                "batches_to_recover": len(batches),
            }

        # Connect to IBM Quantum service
        print("\nConnecting to IBM Quantum service...")
        service_kwargs = {}
        if channel:
            service_kwargs["channel"] = channel
        if token:
            service_kwargs["token"] = token
        if instance:
            service_kwargs["instance"] = instance

        service = QiskitRuntimeService(**service_kwargs)
        print("Connected successfully.")

        recovered = 0
        errors = 0
        results_summary = []

        for batch_info in batches:
            config_id = batch_info["config_id"]
            batch_id = batch_info["batch_id"]
            backend_name = batch_info["backend_name"]

            try:
                result = self.recover_batch(
                    batch_id=batch_id,
                    config_id=config_id,
                    backend_name=backend_name,
                    service=service,
                    job_ids=batch_info.get("job_ids"),
                    transpiled_depths=batch_info.get("transpiled_depths"),
                    original_depth=batch_info.get("original_depth"),
                )

                if result:
                    # Save the recovered result
                    config = self.get_config(config_id)
                    save_path = self._save_results(result, config, backend_name)
                    print(f"Recovered and saved: {save_path}")
                    recovered += 1
                    results_summary.append(
                        {
                            "config_id": config_id,
                            "status": "recovered",
                            "file": str(save_path),
                        }
                    )
                else:
                    print(f"Could not recover batch for {config_id}")
                    results_summary.append(
                        {
                            "config_id": config_id,
                            "status": "failed",
                        }
                    )
                    errors += 1

            except Exception as e:
                print(f"ERROR recovering {config_id}: {e}")
                errors += 1
                results_summary.append(
                    {
                        "config_id": config_id,
                        "status": "error",
                        "error": str(e),
                    }
                )

        # Print final summary
        print(f"\n{'='*70}")
        print("BATCH RECOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Batches processed: {len(batches)}")
        print(f"Successfully recovered: {recovered}")
        print(f"Errors: {errors}")
        print(f"{'='*70}")

        return {
            "status": "completed",
            "batches_processed": len(batches),
            "recovered": recovered,
            "errors": errors,
            "details": results_summary,
        }

    # =========================================================================
    # CLI Support
    # =========================================================================

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI usage."""
        parser = argparse.ArgumentParser(
            description=f"Run {self.algorithm_name} on IBM Quantum hardware"
        )
        parser.add_argument("--config", "-c", default=None, help="Configuration ID to run")
        parser.add_argument(
            "--backend",
            "-b",
            default=None,
            help="IBM backend name (default: auto-select least busy)",
        )
        parser.add_argument(
            "--opt-levels",
            "-o",
            nargs="+",
            type=int,
            default=[0, 1, 2, 3],
            help="Optimization levels to test (default: 0 1 2 3)",
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
            "--channel", default=None, help="IBM Quantum channel ('ibm_quantum' or 'ibm_cloud')"
        )
        parser.add_argument("--token", default=None, help="IBM Quantum API token")
        parser.add_argument(
            "--instance", default=None, help="IBM Quantum instance (e.g., 'ibm-q/open/main')"
        )
        # Update & status commands
        parser.add_argument(
            "--status",
            action="store_true",
            help="Show status of all saved experiment data files",
        )
        parser.add_argument(
            "--update",
            action="store_true",
            help="Update experiment data files with missing results from IBM Cloud",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="With --update: show what would be updated without making changes",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="With --update: re-fetch all jobs from IBM even if data looks healthy",
        )
        # Recovery commands (for timed-out batches with no JSON file)
        parser.add_argument(
            "--recover",
            action="store_true",
            help="Recover timed-out batches from IBM Cloud (creates new JSON files)",
        )
        parser.add_argument(
            "--batch-id",
            default=None,
            help="With --recover: specific batch ID to recover",
        )
        parser.add_argument(
            "--recover-config",
            default=None,
            help="With --recover --batch-id: config ID for the batch being recovered",
        )
        parser.add_argument(
            "--recover-backend",
            default=None,
            help="With --recover --batch-id: backend name for the batch being recovered",
        )
        return parser

    def list_configs(self) -> None:
        """Print available configurations."""
        print("=" * 70)
        print(f"AVAILABLE {self.algorithm_name} CONFIGURATIONS FOR IBM QPU")
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
        print(f"Usage: python <script>.py --config <CONFIG_ID>")
        print("=" * 70)

    def run_cli(self, args: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Run from command line arguments."""
        parser = self.create_argument_parser()
        parsed = parser.parse_args(args)

        # Handle --status command
        if parsed.status:
            self.print_data_status()
            return None

        # Handle --update command
        if parsed.update:
            result = self.update_from_ibm(
                channel=parsed.channel,
                token=parsed.token,
                instance=parsed.instance,
                dry_run=parsed.dry_run,
                force=parsed.force,
            )
            return result

        # Handle --recover command
        if parsed.recover:
            if parsed.batch_id:
                # Recover a specific batch
                if not parsed.recover_config:
                    print("Error: --recover-config is required with --batch-id")
                    return None
                batches = [
                    {
                        "batch_id": parsed.batch_id,
                        "config_id": parsed.recover_config,
                        "backend_name": parsed.recover_backend or "unknown",
                    }
                ]
            else:
                # Use the subclass-defined timed-out batches
                batches = self.get_timed_out_batches()
                if not batches:
                    print("No timed-out batches defined for recovery.")
                    print(
                        "Use --batch-id, --recover-config, --recover-backend to recover a specific batch."
                    )
                    return None

            result = self.recover_batches(
                batches=batches,
                channel=parsed.channel,
                token=parsed.token,
                instance=parsed.instance,
                dry_run=parsed.dry_run,
            )
            return result

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
            backend_name=parsed.backend,
            optimization_levels=parsed.opt_levels,
            save_results=not parsed.no_save,
            channel=parsed.channel,
            token=parsed.token,
            instance=parsed.instance,
        )

        print("\n" + "=" * 70)
        if result.get("status") == "completed":
            print("EXECUTION COMPLETE")
            summary = result.get("batch_summary", {})
            if summary.get("mean_success_rate") is not None:
                print(f"Mean success rate: {summary['mean_success_rate']:.2%}")
                print(f"Quantum advantage: {summary.get('mean_quantum_advantage_ratio', 0):.2f}x")
        else:
            print(f"EXECUTION STATUS: {result.get('status', 'unknown')}")
        print("=" * 70)

        return result
