#!/usr/bin/env python3
"""
QFT IBM QPU Execution Script

This script runs Quantum Fourier Transform on IBM Quantum hardware using the
IBMExperimentBase framework. It uses configurations from Region 1 (high expected
success) identified through simulator analysis.

Usage:
    python qft_ibm.py                        # Run default config (SR2)
    python qft_ibm.py --config SR4           # Run specific config
    python qft_ibm.py --list                 # List available configs

Prerequisites:
    - IBM Quantum account configured (QiskitRuntimeService.save_account())
    - qiskit-ibm-runtime installed

Example:
    >>> from qward.examples.papers.qft.qft_ibm import QFTIBMExperiment
    >>> experiment = QFTIBMExperiment()
    >>> result = experiment.run("SR2")
    >>> print(f"Success rate: {result['batch_summary']['mean_success_rate']:.2%}")
"""

from pathlib import Path
from typing import Dict, Any, List, Callable

from qiskit import QuantumCircuit

from qward.algorithms import QFTCircuitGenerator
from qward.examples.papers.ibm_experiment_base import IBMExperimentBase
from qward.examples.papers.qft.qft_configs import (
    get_config,
    QFTExperimentConfig,
    CONFIGS_BY_ID,
)


# =============================================================================
# Region 1 Configurations (Worth running on QPU)
# Prioritized by expected success rate from simulator analysis
# =============================================================================

REGION1_PRIORITY = [
    # Highest success period detection configs
    {"config_id": "PV4-P8", "expected_success": 1.000, "qubits": 4, "depth": 27, "description": "period=8, 4q"},
    {"config_id": "SP4-P4", "expected_success": 0.980, "qubits": 4, "depth": 27, "description": "period=4, 4q"},
    {"config_id": "PV4-P4", "expected_success": 0.979, "qubits": 4, "depth": 27, "description": "period=4, 4q"},
    {"config_id": "PV6-P16", "expected_success": 0.975, "qubits": 6, "depth": 39, "description": "period=16, 6q"},
    # Round-trip configs (simplest test)
    {"config_id": "SR2", "expected_success": 0.961, "qubits": 2, "depth": 10, "description": "roundtrip 2q"},
    {"config_id": "SR3", "expected_success": 0.944, "qubits": 3, "depth": 14, "description": "roundtrip 3q"},
    {"config_id": "SR4", "expected_success": 0.915, "qubits": 4, "depth": 18, "description": "roundtrip 4q"},
    # Period detection scalability
    {"config_id": "SP5-P4", "expected_success": 0.954, "qubits": 5, "depth": 33, "description": "period=4, 5q"},
    {"config_id": "PV6-P8", "expected_success": 0.954, "qubits": 6, "depth": 39, "description": "period=8, 6q"},
    {"config_id": "SP6-P8", "expected_success": 0.952, "qubits": 6, "depth": 39, "description": "period=8, 6q"},
    # Input variation configs
    {"config_id": "IV4-0000", "expected_success": 0.919, "qubits": 4, "depth": 17, "description": "input |0000⟩"},
    {"config_id": "IV4-0101", "expected_success": 0.916, "qubits": 4, "depth": 18, "description": "input |0101⟩"},
    # Larger roundtrip configs
    {"config_id": "SR5", "expected_success": 0.889, "qubits": 5, "depth": 22, "description": "roundtrip 5q"},
    {"config_id": "SR6", "expected_success": 0.856, "qubits": 6, "depth": 26, "description": "roundtrip 6q"},
    {"config_id": "SR7", "expected_success": 0.826, "qubits": 7, "depth": 30, "description": "roundtrip 7q"},
    # =========================================================================
    # LARGE CONFIGS (QPU ONLY - too slow for classical simulation)
    # QFT scales O(n²), so these are PRACTICAL on current NISQ hardware!
    # Depths are transpiled estimates (opt_level=2, IBM backend)
    # =========================================================================
    {"config_id": "SR8", "expected_success": 0.70, "qubits": 8, "depth": 628, "description": "roundtrip 8q [QPU ONLY]"},
    {"config_id": "SR10", "expected_success": 0.50, "qubits": 10, "depth": 800, "description": "roundtrip 10q [QPU ONLY]"},
    {"config_id": "SR12", "expected_success": 0.30, "qubits": 12, "depth": 1306, "description": "roundtrip 12q [QPU ONLY]"},
    {"config_id": "SR14", "expected_success": 0.15, "qubits": 14, "depth": 1540, "description": "roundtrip 14q [QPU ONLY]"},
    {"config_id": "SP8-P4", "expected_success": 0.60, "qubits": 8, "depth": 650, "description": "period=4, 8q [QPU ONLY]"},
    {"config_id": "SP8-P16", "expected_success": 0.65, "qubits": 8, "depth": 650, "description": "period=16, 8q [QPU ONLY]"},
    {"config_id": "SP10-P4", "expected_success": 0.40, "qubits": 10, "depth": 850, "description": "period=4, 10q [QPU ONLY]"},
    {"config_id": "SP10-P16", "expected_success": 0.45, "qubits": 10, "depth": 850, "description": "period=16, 10q [QPU ONLY]"},
    {"config_id": "SP12-P8", "expected_success": 0.25, "qubits": 12, "depth": 1350, "description": "period=8, 12q [QPU ONLY]"},
]


class QFTIBMExperiment(IBMExperimentBase[QFTExperimentConfig]):
    """QFT algorithm experiment runner for IBM QPU."""
    
    @property
    def algorithm_name(self) -> str:
        return "QFT"
    
    def get_config(self, config_id: str) -> QFTExperimentConfig:
        """Get QFT experiment configuration."""
        return get_config(config_id)
    
    def get_all_config_ids(self) -> List[str]:
        """Get all available configuration IDs."""
        return list(CONFIGS_BY_ID.keys())
    
    def create_circuit(self, config: QFTExperimentConfig) -> QuantumCircuit:
        """Create QFT circuit for the configuration."""
        if config.test_mode == "roundtrip":
            qft_gen = QFTCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="roundtrip",
                input_state=config.input_state,
                use_barriers=True,
            )
        else:  # period_detection
            qft_gen = QFTCircuitGenerator(
                num_qubits=config.num_qubits,
                test_mode="period_detection",
                period=config.period,
                use_barriers=True,
            )
        
        # Store generator for success criteria
        self._current_generator = qft_gen
        return qft_gen.circuit
    
    def create_success_criteria(self, config: QFTExperimentConfig) -> Callable[[str], bool]:
        """Create success criteria for QFT.
        
        For roundtrip: output should match input state
        For period detection: output should be at expected peak positions
        """
        # Use the circuit generator's built-in success criteria if available
        if hasattr(self, '_current_generator') and self._current_generator is not None:
            return self._current_generator.success_criteria
        
        # Fallback: create success criteria based on config
        if config.test_mode == "roundtrip":
            expected_state = config.input_state
            
            def roundtrip_success(result: str) -> bool:
                clean_result = result.replace(" ", "").strip()
                return clean_result == expected_state
            
            return roundtrip_success
        else:
            # Period detection: peaks at N/period intervals
            search_space = config.search_space
            period = config.period
            num_peaks = search_space // period
            
            # Expected peaks at k * (N/period) for k = 0, 1, ..., period-1
            expected_peaks = set()
            step = search_space // period
            for k in range(period):
                peak_value = k * step
                # Convert to binary string
                peak_str = format(peak_value, f'0{config.num_qubits}b')
                expected_peaks.add(peak_str)
            
            def period_success(result: str) -> bool:
                clean_result = result.replace(" ", "").strip()
                return clean_result in expected_peaks
            
            return period_success
    
    def get_random_chance(self, config: QFTExperimentConfig) -> float:
        """Get classical random chance for the configuration."""
        if config.test_mode == "roundtrip":
            # For roundtrip, random chance is 1/2^n (single correct output)
            return 1.0 / config.search_space
        else:
            # For period detection, random chance is num_peaks/2^n
            num_peaks = config.search_space // config.period
            return num_peaks / config.search_space
    
    def get_config_description(self, config: QFTExperimentConfig) -> Dict[str, Any]:
        """Get configuration description for saving."""
        desc = {
            "config_id": config.config_id,
            "num_qubits": config.num_qubits,
            "test_mode": config.test_mode,
            "search_space": config.search_space,
            "theoretical_gate_count": config.theoretical_gate_count,
            "description": config.description,
        }
        
        if config.test_mode == "roundtrip":
            desc["input_state"] = config.input_state
        else:
            desc["period"] = config.period
            desc["expected_num_peaks"] = config.expected_num_peaks
        
        return desc
    
    def evaluate_result(
        self,
        counts: Dict[str, int],
        config: QFTExperimentConfig,
        total_shots: int,
    ) -> Dict[str, Any]:
        """Evaluate QFT result with algorithm-specific metrics."""
        # Calculate basic success metrics
        success_criteria = self.create_success_criteria(config)
        s_count = sum(c for k, c in counts.items() if success_criteria(k))
        s_rate = s_count / total_shots if total_shots > 0 else 0.0
        
        random_chance = self.get_random_chance(config)
        advantage_ratio = s_rate / random_chance if random_chance > 0 else 0.0
        
        result = {
            "success_rate": s_rate,
            "success_count": s_count,
            "test_mode": config.test_mode,
            "random_chance": random_chance,
            "advantage_ratio": advantage_ratio,
            "quantum_advantage": advantage_ratio > 2.0,
            # QFT uses higher thresholds since it's typically very accurate
            "threshold_30": s_rate >= 0.30,
            "threshold_50": s_rate >= 0.50,
            "threshold_70": s_rate >= 0.70,
            "threshold_90": s_rate >= 0.90,
            "threshold_95": s_rate >= 0.95,
            "threshold_99": s_rate >= 0.99,
        }
        
        # Add mode-specific fields
        if config.test_mode == "roundtrip":
            result["input_state"] = config.input_state
        else:
            result["period"] = config.period
            result["expected_num_peaks"] = config.expected_num_peaks
        
        return result
    
    def get_priority_configs(self) -> List[Dict[str, Any]]:
        """Get prioritized configurations for QPU execution."""
        return REGION1_PRIORITY
    
    def get_output_dir(self) -> Path:
        """Get output directory for QFT results."""
        return Path(__file__).parent / "data" / "qpu" / "raw"


# =============================================================================
# Convenience Functions
# =============================================================================

def run_qft_on_ibm(
    config_id: str = "SR2",
    backend_name: str = None,
    optimization_levels: List[int] = None,
    shots: int = 1024,
    timeout: int = 600,
    save_results: bool = True,
    channel: str = None,
    token: str = None,
    instance: str = None,
) -> Dict[str, Any]:
    """Run QFT on IBM Quantum hardware.
    
    Convenience function that creates a QFTIBMExperiment and runs it.
    
    Args:
        config_id: Configuration ID (e.g., "SR2", "SP4-P4")
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
    experiment = QFTIBMExperiment(shots=shots, timeout=timeout)
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
    """List available QFT configurations."""
    experiment = QFTIBMExperiment()
    experiment.list_configs()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run QFT IBM experiment from command line."""
    experiment = QFTIBMExperiment()
    experiment.run_cli()


if __name__ == "__main__":
    main()
