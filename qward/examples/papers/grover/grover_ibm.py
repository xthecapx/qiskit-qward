#!/usr/bin/env python3
"""
Grover IBM QPU Execution Script

This script runs Grover's algorithm on IBM Quantum hardware using the
IBMExperimentBase framework. It uses configurations from Region 1 (high expected
success) identified through simulator analysis.

Usage:
    python grover_ibm.py                     # Run default config (S2-1)
    python grover_ibm.py --config S3-1       # Run specific config
    python grover_ibm.py --list              # List available configs

Prerequisites:
    - IBM Quantum account configured (QiskitRuntimeService.save_account())
    - qiskit-ibm-runtime installed

Example:
    >>> from qward.examples.papers.grover.grover_ibm import GroverIBMExperiment
    >>> experiment = GroverIBMExperiment()
    >>> result = experiment.run("S2-1")
    >>> print(f"Success rate: {result['batch_summary']['mean_success_rate']:.2%}")
"""

from pathlib import Path
from typing import Dict, Any, List, Callable

from qiskit import QuantumCircuit

from qward.algorithms import Grover
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase
from qward.examples.papers.grover.grover_configs import (
    get_config,
    ExperimentConfig,
    CONFIGS_BY_ID,
)
from qward.examples.papers.grover.grover_success_metrics import evaluate_job


# =============================================================================
# Region 1 Configurations (Worth running on QPU)
# Prioritized by expected success rate from simulator analysis
# =============================================================================

REGION1_PRIORITY = [
    # Priority 1-5: Highest success (>90%)
    {"config_id": "S2-1", "expected_success": 0.967, "qubits": 2, "depth": 12, "description": "2q scalability"},
    {"config_id": "ASYM-1", "expected_success": 0.920, "qubits": 3, "depth": 46, "description": "asymmetric marked"},
    {"config_id": "ASYM-2", "expected_success": 0.905, "qubits": 3, "depth": 46, "description": "asymmetric marked"},
    {"config_id": "M3-2", "expected_success": 0.905, "qubits": 3, "depth": 44, "description": "2 marked states"},
    {"config_id": "SYM-1", "expected_success": 0.902, "qubits": 3, "depth": 44, "description": "symmetric marked"},
    # Priority 6-10: Good success (80-90%)
    {"config_id": "SYM-2", "expected_success": 0.903, "qubits": 3, "depth": 44, "description": "symmetric marked"},
    {"config_id": "H3-3", "expected_success": 0.837, "qubits": 3, "depth": 58, "description": "hamming weight 3"},
    {"config_id": "H3-2", "expected_success": 0.833, "qubits": 3, "depth": 62, "description": "hamming weight 2"},
    {"config_id": "S3-1", "expected_success": 0.826, "qubits": 3, "depth": 62, "description": "3q scalability"},
    {"config_id": "M3-1", "expected_success": 0.823, "qubits": 3, "depth": 62, "description": "1 marked state"},
    # Priority 11-15: 4 qubit configs (68-79%)
    {"config_id": "M4-4", "expected_success": 0.791, "qubits": 4, "depth": 142, "description": "4 marked states"},
    {"config_id": "M4-2", "expected_success": 0.712, "qubits": 4, "depth": 172, "description": "2 marked states"},
    {"config_id": "S4-1", "expected_success": 0.706, "qubits": 4, "depth": 178, "description": "4q scalability"},
    {"config_id": "H4-4", "expected_success": 0.700, "qubits": 4, "depth": 172, "description": "hamming weight 4"},
    {"config_id": "H4-0", "expected_success": 0.682, "qubits": 4, "depth": 178, "description": "hamming weight 0"},
    # =========================================================================
    # LARGE CONFIGS - Transpiled depth grows ~4x per qubit!
    # Note: 8+ qubits have VERY deep circuits (100k+ depth) - may not be 
    # practical on current NISQ hardware due to decoherence
    # =========================================================================
    {"config_id": "S5-1", "expected_success": 0.50, "qubits": 5, "depth": 1940, "description": "5q scalability (practical)"},
    {"config_id": "S6-1", "expected_success": 0.25, "qubits": 6, "depth": 8087, "description": "6q scalability (challenging)"},
    {"config_id": "S7-1", "expected_success": 0.10, "qubits": 7, "depth": 33000, "description": "7q scalability [DEEP]"},
    {"config_id": "S8-1", "expected_success": 0.02, "qubits": 8, "depth": 140000, "description": "8q scalability [VERY DEEP]"},
    {"config_id": "S10-1", "expected_success": 0.001, "qubits": 10, "depth": 2000000, "description": "10q [IMPRACTICAL - decoherence]"},
    {"config_id": "S12-1", "expected_success": 0.0001, "qubits": 12, "depth": 30000000, "description": "12q [THEORETICAL ONLY]"},
    {"config_id": "S14-1", "expected_success": 0.00001, "qubits": 14, "depth": 500000000, "description": "14q [THEORETICAL ONLY]"},
]


class GroverIBMExperiment(IBMExperimentBase[ExperimentConfig]):
    """Grover's algorithm experiment runner for IBM QPU."""
    
    @property
    def algorithm_name(self) -> str:
        return "GROVER"
    
    def get_config(self, config_id: str) -> ExperimentConfig:
        """Get Grover experiment configuration."""
        return get_config(config_id)
    
    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        return list(CONFIGS_BY_ID.keys())
    
    def create_circuit(self, config: ExperimentConfig) -> QuantumCircuit:
        """Create Grover circuit for the configuration."""
        grover = Grover(marked_states=config.marked_states, use_barriers=True)
        return grover.circuit
    
    def create_success_criteria(self, config: ExperimentConfig) -> Callable[[str], bool]:
        """Create success criteria for Grover's algorithm."""
        marked_states = config.marked_states
        
        def is_success(result: str) -> bool:
            clean_result = result.replace(" ", "").strip()
            return clean_result in marked_states
        
        return is_success
    
    def get_random_chance(self, config: ExperimentConfig) -> float:
        """Get classical random search probability."""
        return config.classical_random_prob
    
    def get_config_description(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Get configuration description for saving."""
        return {
            "config_id": config.config_id,
            "num_qubits": config.num_qubits,
            "marked_states": config.marked_states,
            "num_marked": config.num_marked,
            "theoretical_success": config.theoretical_success,
            "classical_random_prob": config.classical_random_prob,
            "theoretical_iterations": config.theoretical_iterations,
            "description": config.description,
        }
    
    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: ExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        """Evaluate Grover result with algorithm-specific metrics."""
        # Calculate basic success metrics
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0
        
        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0
        
        result = {
            "success_rate": s_rate,
            "success_count": s_count,
            "marked_states": config.marked_states,
            "num_marked": config.num_marked,
            "theoretical_success": config.theoretical_success,
            "grover_iterations": config.theoretical_iterations,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
        }
        
        # Try to use Grover-specific evaluation
        if counts:
            try:
                evaluation = evaluate_job(
                    counts=counts,
                    marked_states=config.marked_states,
                    num_qubits=config.num_qubits,
                    theoretical_prob=config.theoretical_success,
                    config_id=config.config_id,
                )
                result.update({
                    "threshold_30": evaluation.threshold_30,
                    "threshold_50": evaluation.threshold_50,
                    "threshold_70": evaluation.threshold_70,
                    "threshold_90": evaluation.threshold_90,
                    "statistical_success": str(evaluation.statistical_success),
                    "statistical_pvalue": evaluation.statistical_pvalue,
                    "quantum_advantage": evaluation.quantum_advantage_success,
                    "advantage_ratio": evaluation.advantage_ratio,
                })
            except Exception:
                # Fallback to basic thresholds
                result.update({
                    "threshold_30": s_rate >= 0.30,
                    "threshold_50": s_rate >= 0.50,
                    "threshold_70": s_rate >= 0.70,
                    "threshold_90": s_rate >= 0.90,
                })
        else:
            result.update({
                "threshold_30": False,
                "threshold_50": False,
                "threshold_70": False,
                "threshold_90": False,
            })
        
        return result
    
    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return REGION1_PRIORITY
    
    def get_output_dir(self) -> Path:
        """Get output directory for Grover results."""
        return Path(__file__).parent / "data" / "qpu" / "raw"


# =============================================================================
# Convenience Functions
# =============================================================================

def run_grover_on_ibm(
    config_id: str = "S2-1",
    backend_name: str = None,
    optimization_levels: List[int] = None,
    shots: int = 1024,
    timeout: int = 600,
    save_results: bool = True,
    channel: str = None,
    token: str = None,
    instance: str = None,
) -> Dict[str, Any]:
    """Run Grover's algorithm on IBM Quantum hardware.
    
    Convenience function that creates a GroverIBMExperiment and runs it.
    
    Args:
        config_id: Configuration ID (e.g., "S2-1", "S3-1")
        backend_name: Optional IBM backend name
        optimization_levels: Transpiler optimization levels
        shots: Number of shots
        timeout: Timeout in seconds
        save_results: Whether to save results
        channel: IBM Quantum channel
        token: IBM Quantum API token
        instance: IBM Quantum instance
        
    Returns:
        Rich result dict with batch_summary and individual_results
    """
    experiment = GroverIBMExperiment(shots=shots, timeout=timeout)
    return experiment.run(
        config_id=config_id,
        backend_name=backend_name,
        optimization_levels=optimization_levels,
        save_results=save_results,
        channel=channel,
        token=token,
        instance=instance,
    )


def list_configs():
    """List available Grover configurations."""
    experiment = GroverIBMExperiment()
    experiment.list_configs()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run Grover IBM experiment from command line."""
    experiment = GroverIBMExperiment()
    experiment.run_cli()


if __name__ == "__main__":
    main()
