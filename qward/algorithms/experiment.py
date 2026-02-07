"""
Base Experiment Framework

This module provides a generalized experiment framework for running quantum algorithm
experiments. It abstracts the common patterns from QFT, Grover, and other experiments
into reusable base classes.

Usage:
    1. Subclass BaseExperimentRunner for your algorithm
    2. Implement the abstract methods for your specific algorithm
    3. Use the inherited campaign workflow for systematic experiments

Example:
    class MyAlgorithmRunner(BaseExperimentRunner):
        def create_circuit(self, config):
            return MyAlgorithmCircuitGenerator(config).circuit

        def calculate_success(self, counts, config, circuit_gen):
            return success_rate(counts, config.target_states)

        def create_result(self, ...):
            return MyExperimentResult(...)

    runner = MyAlgorithmRunner()
    results = runner.run_campaign(config_ids=['A', 'B'], noise_ids=['IDEAL', 'DEP-MED'])
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Tuple,
    Generic,
    TypeVar,
    Callable,
    Union,
    cast,
)

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .noise_generator import NoiseConfig, NoiseModelGenerator
from .experiment_utils import calculate_qward_metrics


# =============================================================================
# Type Variables for Generic Classes
# =============================================================================

ConfigT = TypeVar("ConfigT")  # Experiment configuration type
ResultT = TypeVar("ResultT", bound="BaseExperimentResult")  # Result type
BatchT = TypeVar("BatchT", bound="BaseBatchResult")  # Batch result type
AnalysisT = TypeVar("AnalysisT")  # Analysis type


# =============================================================================
# Base Data Classes
# =============================================================================


@dataclass
class BaseExperimentResult:
    """
    Base class for experiment results.

    All experiment-specific result classes should inherit from this and add
    their own fields. The common fields provide identification, timing, and
    QWARD metrics for correlation analysis.
    """

    # Identification
    experiment_id: str
    config_id: str
    noise_model: str
    run_number: int
    timestamp: str

    # Circuit properties (always available)
    num_qubits: int
    circuit_depth: int
    total_gates: int

    # QWARD Pre-runtime Metrics
    qward_metrics: Optional[Dict[str, Any]] = None

    # Backend metadata
    backend_type: str = "simulator"
    backend_name: str = "AerSimulator"

    # Execution
    shots: int = 1024
    execution_time_ms: float = 0.0

    # Results
    counts: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    success_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseExperimentResult":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class BaseBatchResult(Generic[ResultT, AnalysisT]):
    """
    Base class for batch results (multiple runs of the same configuration).

    Contains aggregate statistics and optional statistical analysis.
    """

    config_id: str
    noise_model: str
    num_runs: int
    shots_per_run: int

    # Aggregate statistics
    mean_success_rate: float
    std_success_rate: float
    min_success_rate: float
    max_success_rate: float
    median_success_rate: float

    # Backend metadata
    backend_type: str = "simulator"
    backend_name: str = "AerSimulator"

    # Statistical analysis (algorithm-specific)
    analysis: Optional[AnalysisT] = None

    # Individual results
    results: List[ResultT] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without individual results for summary)."""
        result = {
            "config_id": self.config_id,
            "noise_model": self.noise_model,
            "num_runs": self.num_runs,
            "shots_per_run": self.shots_per_run,
            "backend_type": self.backend_type,
            "backend_name": self.backend_name,
            "mean_success_rate": self.mean_success_rate,
            "std_success_rate": self.std_success_rate,
            "min_success_rate": self.min_success_rate,
            "max_success_rate": self.max_success_rate,
            "median_success_rate": self.median_success_rate,
        }
        if self.analysis is not None:
            if hasattr(self.analysis, "to_dict"):
                result["analysis"] = self.analysis.to_dict()
            else:
                result["analysis"] = self.analysis
        return result


# =============================================================================
# Campaign Progress Tracking
# =============================================================================


@dataclass
class CampaignProgress:
    """Tracks progress of an experiment campaign."""

    total_batches: int
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        elapsed = self.elapsed_seconds
        return f"{elapsed:.1f}s ({elapsed/60:.1f} min)"

    def __str__(self) -> str:
        return (
            f"Progress: {self.completed + self.skipped}/{self.total_batches} "
            f"(completed: {self.completed}, skipped: {self.skipped}, failed: {self.failed})"
        )


# =============================================================================
# Base Experiment Runner
# =============================================================================


# This abstract runner intentionally exposes broad constructor/method signatures
# to support multiple algorithm families through one generic interface.
# pylint: disable=too-many-positional-arguments
class BaseExperimentRunner(ABC, Generic[ConfigT, ResultT, BatchT, AnalysisT]):
    """
    Abstract base class for experiment runners.

    Provides the common workflow for running experiments across configurations
    and noise models, with incremental saving and resume support.

    Subclasses must implement:
        - algorithm_name: Property returning the algorithm name
        - create_circuit: Create quantum circuit from config
        - calculate_success: Calculate success metrics from counts
        - create_result: Create experiment result object
        - get_config: Get experiment config by ID
        - get_noise_config: Get noise config by ID
        - get_all_config_ids: Get all available config IDs
        - get_all_noise_ids: Get all available noise IDs

    Optional overrides:
        - analyze_batch: Statistical analysis for batch results
        - load_result_from_dict: Custom result deserialization
    """

    def __init__(
        self,
        shots: int = 1024,
        num_runs: int = 10,
        optimization_level: int = 0,
        output_dir: str = "data",
        backend_type: str = "simulator",
        backend_name: str = "AerSimulator",
    ):
        """
        Initialize the experiment runner.

        Args:
            shots: Number of shots per experiment
            num_runs: Default number of runs per batch
            optimization_level: Transpiler optimization level
            output_dir: Base output directory for results
        """
        self.shots = shots
        self.num_runs = num_runs
        self.optimization_level = optimization_level
        self.output_dir = output_dir
        self.backend_type = backend_type
        self.backend_name = backend_name

    # =========================================================================
    # Abstract Properties and Methods (must be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the algorithm name (e.g., 'GROVER', 'QFT')."""
        pass

    @abstractmethod
    def create_circuit(self, config: ConfigT) -> Tuple[QuantumCircuit, Any]:
        """
        Create the quantum circuit from configuration.

        Args:
            config: Experiment configuration

        Returns:
            Tuple of (circuit, circuit_generator_or_metadata)
            The second element can be used for success calculation
        """
        pass

    @abstractmethod
    def calculate_success(
        self,
        counts: Dict[str, int],
        config: ConfigT,
        circuit_metadata: Any,
    ) -> Tuple[float, int]:
        """
        Calculate success metrics from measurement counts.

        Args:
            counts: Measurement counts
            config: Experiment configuration
            circuit_metadata: Circuit generator or metadata from create_circuit

        Returns:
            Tuple of (success_rate, success_count)
        """
        pass

    @abstractmethod
    def create_result(
        self,
        config: ConfigT,
        noise_config: NoiseConfig,
        run_number: int,
        transpiled_circuit: QuantumCircuit,
        counts: Dict[str, int],
        execution_time_ms: float,
        success_rate: float,
        success_count: int,
        qward_metrics: Optional[Dict[str, Any]],
        circuit_metadata: Any,
        backend_type: str,
        backend_name: str,
    ) -> ResultT:
        """
        Create an experiment result object.

        Args:
            config: Experiment configuration
            noise_config: Noise model configuration
            run_number: Run number (1-indexed)
            transpiled_circuit: Transpiled quantum circuit
            counts: Measurement counts
            execution_time_ms: Execution time in milliseconds
            success_rate: Calculated success rate
            success_count: Number of successful shots
            qward_metrics: QWARD metrics dictionary
            circuit_metadata: Circuit generator or metadata

        Returns:
            Algorithm-specific experiment result
        """
        pass

    @abstractmethod
    def get_config(self, config_id: str) -> ConfigT:
        """Get experiment configuration by ID."""
        pass

    @abstractmethod
    def get_noise_config(self, noise_id: str) -> NoiseConfig:
        """Get noise configuration by ID."""
        pass

    @abstractmethod
    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        pass

    @abstractmethod
    def get_all_noise_ids(self) -> List[str]:
        """Get all available noise model IDs."""
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    def analyze_batch(
        self,
        success_rates: List[float],
        config_id: str,
        noise_model: str,
        ideal_rates: Optional[List[float]] = None,
    ) -> Optional[AnalysisT]:
        """
        Perform statistical analysis on batch results.

        Override this method to add algorithm-specific statistical analysis.

        Args:
            success_rates: List of success rates from runs
            config_id: Configuration ID
            noise_model: Noise model ID
            ideal_rates: Optional ideal rates for comparison

        Returns:
            Analysis object or None
        """
        return None

    def print_batch_analysis(self, analysis: Optional[AnalysisT]) -> None:
        """
        Print batch analysis summary.

        Override to customize analysis output.
        """
        if analysis is not None and hasattr(analysis, "__str__"):
            print(analysis)

    def load_result_from_dict(self, data: Dict[str, Any]) -> ResultT:
        """
        Reconstruct result from dictionary.

        Override for custom deserialization.
        """
        # Default implementation - subclasses should override
        raise NotImplementedError(
            "Subclass must implement load_result_from_dict for custom result types"
        )

    def load_analysis_from_dict(self, data: Optional[Dict[str, Any]]) -> Optional[AnalysisT]:
        """
        Reconstruct analysis from dictionary.

        Override for custom analysis deserialization.
        """
        return None  # Default: no analysis reconstruction

    def get_config_description(self, config: ConfigT) -> str:
        """
        Get a description string for the configuration.

        Override for custom config descriptions in verbose output.
        """
        if hasattr(config, "num_qubits"):
            return f"Qubits: {config.num_qubits}"
        return ""

    # =========================================================================
    # Core Experiment Execution
    # =========================================================================

    def run_single(
        self,
        config: ConfigT,
        noise_config: NoiseConfig,
        run_number: int,
        shots: Optional[int] = None,
        calculate_qward: bool = True,
    ) -> ResultT:
        """
        Run a single experiment with given configuration.

        Args:
            config: Experiment configuration
            noise_config: Noise model configuration
            run_number: Run number (1-indexed)
            shots: Number of shots (defaults to self.shots)
            calculate_qward: Whether to calculate QWARD metrics

        Returns:
            Experiment result
        """
        shots = shots or self.shots

        # Create circuit
        circuit, circuit_metadata = self.create_circuit(config)

        # Create noise model
        noise_model = NoiseModelGenerator.create_from_config(noise_config)

        # Create simulator
        simulator = AerSimulator(noise_model=noise_model)

        # Transpile circuit
        pm = generate_preset_pass_manager(
            target=simulator.target,
            optimization_level=self.optimization_level,
        )
        transpiled_circuit = pm.run(circuit)

        # Calculate QWARD metrics on transpiled circuit
        qward_metrics = None
        if calculate_qward:
            qward_metrics = calculate_qward_metrics(transpiled_circuit)

        # Run experiment
        start_time = time.time()
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        execution_time_ms = (time.time() - start_time) * 1000

        counts = result.get_counts()

        # Calculate success metrics
        success_rate, success_count = self.calculate_success(counts, config, circuit_metadata)

        # Create and return result
        return self.create_result(
            config=config,
            noise_config=noise_config,
            run_number=run_number,
            transpiled_circuit=transpiled_circuit,
            counts=counts,
            execution_time_ms=execution_time_ms,
            success_rate=success_rate,
            success_count=success_count,
            qward_metrics=qward_metrics,
            circuit_metadata=circuit_metadata,
            backend_type=self.backend_type,
            backend_name=self.backend_name,
        )

    def run_batch(
        self,
        config_id: str,
        noise_id: str,
        num_runs: Optional[int] = None,
        shots: Optional[int] = None,
        ideal_rates: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> BatchT:
        """
        Run multiple experiments with the same configuration.

        Args:
            config_id: Configuration ID
            noise_id: Noise model ID
            num_runs: Number of runs (defaults to self.num_runs)
            shots: Shots per run (defaults to self.shots)
            ideal_rates: Optional ideal rates for comparison
            verbose: Print progress

        Returns:
            Batch result with aggregate statistics
        """
        num_runs = num_runs or self.num_runs
        shots = shots or self.shots

        config = self.get_config(config_id)
        noise_config = self.get_noise_config(noise_id)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running {config_id} with {noise_id} ({num_runs} runs)")
            desc = self.get_config_description(config)
            if desc:
                print(f"  {desc}")
            print(f"{'=' * 60}")

        results = []
        for run in range(1, num_runs + 1):
            if verbose:
                print(f"  Run {run}/{num_runs}...", end=" ", flush=True)

            result = self.run_single(config, noise_config, run, shots)
            results.append(result)

            if verbose:
                print(f"Success rate: {result.success_rate:.4f}")

        # Calculate aggregate statistics
        rates = [r.success_rate for r in results]

        # Statistical analysis
        analysis = cast(
            Optional[AnalysisT],
            self.analyze_batch(
                success_rates=rates,
                config_id=config_id,
                noise_model=noise_id,
                ideal_rates=ideal_rates,
            ),
        )

        if verbose and analysis is not None:
            self.print_batch_analysis(analysis)
        elif verbose:
            print(f"\n  Summary: mean={np.mean(rates):.4f}, std={np.std(rates):.4f}")

        return self._create_batch_result(
            config_id=config_id,
            noise_model=noise_id,
            num_runs=num_runs,
            shots_per_run=shots,
            rates=rates,
            analysis=analysis,
            results=results,
            backend_type=self.backend_type,
            backend_name=self.backend_name,
        )

    def _create_batch_result(
        self,
        config_id: str,
        noise_model: str,
        num_runs: int,
        shots_per_run: int,
        rates: List[float],
        analysis: Optional[AnalysisT],
        results: List[ResultT],
        backend_type: str,
        backend_name: str,
    ) -> BatchT:
        """Create batch result. Override for custom batch result types."""
        return cast(
            BatchT,
            BaseBatchResult(
                config_id=config_id,
                noise_model=noise_model,
                num_runs=num_runs,
                shots_per_run=shots_per_run,
                backend_type=backend_type,
                backend_name=backend_name,
                mean_success_rate=float(np.mean(rates)),
                std_success_rate=float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                min_success_rate=float(np.min(rates)),
                max_success_rate=float(np.max(rates)),
                median_success_rate=float(np.median(rates)),
                analysis=analysis,
                results=results,
            ),
        )

    # =========================================================================
    # Campaign Execution
    # =========================================================================

    def run_campaign(
        self,
        config_ids: Optional[List[str]] = None,
        noise_ids: Optional[List[str]] = None,
        num_runs: Optional[int] = None,
        shots: Optional[int] = None,
        save_results: bool = True,
        output_dir: Optional[str] = None,
        verbose: bool = True,
        incremental_save: bool = True,
        session_id: Optional[str] = None,
        skip_existing: bool = True,
    ) -> Dict[Tuple[str, str], BatchT]:
        # pylint: disable=too-many-branches
        """
        Run a full experiment campaign across configurations and noise models.

        Args:
            config_ids: List of configuration IDs (None = all)
            noise_ids: List of noise model IDs (None = all)
            num_runs: Number of runs per configuration
            shots: Shots per run
            save_results: Whether to save results to disk
            output_dir: Output directory for results
            verbose: Print progress
            incremental_save: Save results after each batch
            session_id: Unique session identifier (auto-generated if None)
            skip_existing: Skip configurations that already have saved results

        Returns:
            Dictionary mapping (config_id, noise_id) to BatchResult
        """
        # Set defaults
        config_ids = config_ids or self.get_all_config_ids()
        noise_ids = noise_ids or self.get_all_noise_ids()
        num_runs = num_runs or self.num_runs
        shots = shots or self.shots
        output_dir = output_dir or self.output_dir

        # Ensure IDEAL is first for baseline comparison
        if "IDEAL" in noise_ids:
            noise_ids = ["IDEAL"] + [n for n in noise_ids if n != "IDEAL"]

        total_batches = len(config_ids) * len(noise_ids)

        # Generate session ID
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup paths
        base_path = Path(output_dir)
        raw_path = base_path / "raw" / session_id
        raw_path.mkdir(parents=True, exist_ok=True)

        if verbose:
            self._print_campaign_header(
                session_id,
                config_ids,
                noise_ids,
                num_runs,
                shots,
                total_batches,
                incremental_save,
                skip_existing,
                raw_path,
            )

        all_results: Dict[Tuple[str, str], BatchT] = {}
        ideal_rates_cache: Dict[str, List[float]] = {}

        # Check for existing results
        existing_files = set()
        if skip_existing:
            existing_files = {f.stem for f in raw_path.glob("*.json")}
            if existing_files and verbose:
                print(f"\nFound {len(existing_files)} existing results in this session")

        progress = CampaignProgress(total_batches=total_batches)

        for config_id in config_ids:
            for noise_id in noise_ids:
                batch_key = f"{config_id}_{noise_id}"
                batch_num = progress.completed + progress.skipped + progress.failed + 1

                # Skip if already exists
                if skip_existing and batch_key in existing_files:
                    if verbose:
                        print(f"\n[{batch_num}/{total_batches}] {batch_key} - SKIPPED (exists)")
                    progress.skipped += 1

                    # Load existing result
                    try:
                        existing_result = self._load_batch_from_file(raw_path / f"{batch_key}.json")
                        if existing_result:
                            all_results[(config_id, noise_id)] = existing_result
                            if noise_id == "IDEAL":
                                ideal_rates_cache[config_id] = [
                                    r.success_rate for r in existing_result.results
                                ]
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: Could not load existing result: {e}")
                    continue

                if verbose:
                    print(f"\n[{batch_num}/{total_batches}] ", end="")

                try:
                    # Get ideal rates for comparison
                    ideal_rates = ideal_rates_cache.get(config_id)

                    batch_result = self.run_batch(
                        config_id=config_id,
                        noise_id=noise_id,
                        num_runs=num_runs,
                        shots=shots,
                        ideal_rates=ideal_rates,
                        verbose=verbose,
                    )

                    # Cache ideal rates
                    if noise_id == "IDEAL":
                        ideal_rates_cache[config_id] = [
                            r.success_rate for r in batch_result.results
                        ]

                    all_results[(config_id, noise_id)] = batch_result
                    progress.completed += 1

                    # Incremental save
                    if incremental_save:
                        self._save_batch_to_file(batch_result, config_id, noise_id, raw_path)
                        if verbose:
                            print(f"    [Saved: {batch_key}.json]")

                except Exception as e:
                    progress.failed += 1
                    if verbose:
                        print(f"    ERROR: {e}")

        # Save campaign summary
        if save_results:
            self._save_campaign_summary(all_results, base_path, session_id, verbose)

        if verbose:
            self._print_campaign_footer(progress, session_id, base_path)

        return all_results

    # =========================================================================
    # Data Persistence
    # =========================================================================

    def _save_batch_to_file(
        self,
        batch: BatchT,
        config_id: str,
        noise_id: str,
        output_path: Path,
    ) -> None:
        """Save a single batch result to disk."""
        filename = f"{config_id}_{noise_id}.json"
        filepath = output_path / filename

        data = {
            "config_id": config_id,
            "noise_id": noise_id,
            "algorithm": self.algorithm_name,
            "saved_at": datetime.now().isoformat(),
            "batch_summary": batch.to_dict(),
            "individual_results": [r.to_dict() for r in batch.results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_batch_from_file(self, filepath: Path) -> Optional[BatchT]:
        """Load a batch result from disk."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            summary = data["batch_summary"]
            individual = data.get("individual_results", [])

            # Reconstruct results
            results = [self.load_result_from_dict(r) for r in individual]

            # Reconstruct analysis
            analysis = cast(
                Optional[AnalysisT], self.load_analysis_from_dict(summary.get("analysis"))
            )

            return self._create_batch_result(
                config_id=summary.get("config_id", ""),
                noise_model=summary.get("noise_model", ""),
                num_runs=summary.get("num_runs", 0),
                shots_per_run=summary.get("shots_per_run", 0),
                rates=[r.success_rate for r in results],
                analysis=analysis,
                results=results,
                backend_type=summary.get("backend_type", self.backend_type),
                backend_name=summary.get("backend_name", self.backend_name),
            )
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return None

    def _save_campaign_summary(
        self,
        results: Dict[Tuple[str, str], BatchT],
        base_path: Path,
        session_id: str,
        verbose: bool = True,
    ) -> None:
        """Save aggregated campaign summary."""
        agg_path = base_path / "aggregated"
        agg_path.mkdir(parents=True, exist_ok=True)

        summary = []
        for (config_id, noise_id), batch in results.items():
            summary.append(
                {
                    "config_id": config_id,
                    "noise_model": noise_id,
                    **batch.to_dict(),
                }
            )

        summary_file = agg_path / f"campaign_summary_{session_id}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        if verbose:
            print(f"\nAggregated summary saved to: {summary_file}")

    def aggregate_session(
        self,
        session_id: str,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[Tuple[str, str], BatchT]:
        """
        Aggregate all results from a session directory.

        Useful for resuming or analyzing a partially completed campaign.

        Args:
            session_id: The session ID to aggregate
            output_dir: Output directory
            verbose: Print progress

        Returns:
            Dictionary mapping (config_id, noise_id) to BatchResult
        """
        output_dir = output_dir or self.output_dir
        base_path = Path(output_dir)
        raw_path = base_path / "raw" / session_id

        if not raw_path.exists():
            raise ValueError(f"Session directory not found: {raw_path}")

        all_results: Dict[Tuple[str, str], BatchT] = {}
        files = list(raw_path.glob("*.json"))

        if verbose:
            print(f"Loading {len(files)} results from session {session_id}...")

        for filepath in files:
            # Parse filename to get config_id and noise_id
            parts = filepath.stem.rsplit("_", 1)
            if len(parts) >= 2:
                config_id = parts[0]
                noise_id = parts[1]

                batch = self._load_batch_from_file(filepath)
                if batch:
                    all_results[(config_id, noise_id)] = batch
                    if verbose:
                        print(f"  Loaded: {config_id}_{noise_id}")

        if verbose:
            print(f"Loaded {len(all_results)} batch results")

        # Generate aggregated summary
        self._save_campaign_summary(all_results, base_path, session_id, verbose)

        return all_results

    # =========================================================================
    # Output Helpers
    # =========================================================================

    def _print_campaign_header(
        self,
        session_id: str,
        config_ids: List[str],
        noise_ids: List[str],
        num_runs: int,
        shots: int,
        total_batches: int,
        incremental_save: bool,
        skip_existing: bool,
        raw_path: Path,
    ) -> None:
        """Print campaign header."""
        print(f"\n{'#' * 70}")
        print(f"{self.algorithm_name} EXPERIMENT CAMPAIGN")
        print(f"{'#' * 70}")
        print(f"Session ID: {session_id}")
        print(f"Configurations: {len(config_ids)}")
        print(f"Noise models: {len(noise_ids)}")
        print(f"Runs per batch: {num_runs}")
        print(f"Shots per run: {shots}")
        print(f"Total batches: {total_batches}")
        print(f"Incremental save: {incremental_save}")
        print(f"Skip existing: {skip_existing}")
        print(f"Output path: {raw_path}")
        print(f"{'#' * 70}")

    def _print_campaign_footer(
        self,
        progress: CampaignProgress,
        session_id: str,
        base_path: Path,
    ) -> None:
        """Print campaign footer."""
        print(f"\n{'#' * 70}")
        print("CAMPAIGN COMPLETE")
        print(f"{'#' * 70}")
        print(f"Total batches: {progress.total_batches}")
        print(f"Completed: {progress.completed}")
        print(f"Skipped (existing): {progress.skipped}")
        print(f"Failed: {progress.failed}")
        print(f"Total time: {progress.elapsed_formatted}")
        print(f"Session ID: {session_id}")
        print(f"Results saved to: {base_path}")
        print(f"{'#' * 70}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BaseExperimentResult",
    "BaseBatchResult",
    "BaseExperimentRunner",
    "CampaignProgress",
    "ConfigT",
    "ResultT",
    "BatchT",
    "AnalysisT",
]
