# Noise Model Generator Refactoring Plan

## Overview

This document outlines the plan to refactor common code between the QFT and Grover experiment runners (`qft_experiment.py` and `grover_experiment.py`). The primary goal is to create a unified **Noise Model Generator** module in the `algorithms` folder to prevent code duplication and provide a reusable noise modeling infrastructure.

## Problem Statement

### Current Code Duplication

The following functions/code are duplicated or nearly identical across experiment files:

| Function/Code | `qft_experiment.py` | `grover_experiment.py` | `executor.py` |
|---------------|---------------------|------------------------|---------------|
| `create_noise_model()` | ✅ Lines 119-176 | ✅ Lines 120-200 | ✅ (class methods) |
| `calculate_qward_metrics()` | ✅ Lines 60-96 | ✅ Lines 61-101 | ❌ |
| `_serialize_value()` | ✅ Lines 99-111 | ✅ Lines 104-117 | ❌ |
| Constants (SHOTS, NUM_RUNS, etc.) | ✅ Lines 50-53 | ✅ Lines 52-54 | ✅ (in __init__) |
| `ExperimentResult` dataclass | ✅ | ✅ | ❌ |
| `BatchResult` dataclass | ✅ | ✅ | ❌ |

### Pain Points

1. **Maintenance Burden**: Changes to noise models must be made in multiple places
2. **Inconsistency Risk**: Different implementations may diverge over time
3. **Testing Overhead**: Same logic needs testing in multiple files
4. **Feature Parity**: New noise models added to one file might be forgotten in others

---

## Proposed Solution

### New Module Structure

Create a new module `noise_generator.py` in the `algorithms` folder with the following components:

```
qward/algorithms/
├── __init__.py                    # Updated exports
├── noise_generator.py             # NEW: Unified noise model generator
├── experiment_utils.py            # NEW: Shared experiment utilities
├── executor.py                    # Refactored to use noise_generator
├── grover.py
├── qft.py
└── ...
```

---

## Phase 1: Noise Model Generator (`noise_generator.py`)

### 1.1 NoiseConfig Dataclass

```python
@dataclass
class NoiseConfig:
    """Configuration for noise model generation."""
    noise_id: str
    noise_type: str  # "none", "depolarizing", "pauli", "readout", "combined", "thermal"
    parameters: Dict[str, float]
    description: str = ""
```

### 1.2 NoiseModelGenerator Class

```python
class NoiseModelGenerator:
    """
    Unified noise model generator for quantum circuit experiments.
    
    Supports:
    - Depolarizing noise (single and two-qubit)
    - Pauli noise (X, Y, Z errors)
    - Readout errors
    - Combined noise (depolarizing + readout)
    - Thermal noise (T1/T2 approximation)
    - Custom noise levels
    """
    
    @staticmethod
    def create_from_config(config: NoiseConfig) -> Optional[NoiseModel]:
        """Create noise model from configuration."""
        
    @staticmethod
    def create_depolarizing(p1: float = 0.01, p2: float = 0.05) -> NoiseModel:
        """Create depolarizing noise model."""
        
    @staticmethod
    def create_pauli(px: float = 0.01, py: float = 0.01, pz: float = 0.01) -> NoiseModel:
        """Create Pauli noise model."""
        
    @staticmethod
    def create_readout(p01: float = 0.02, p10: float = 0.02) -> NoiseModel:
        """Create readout error model."""
        
    @staticmethod
    def create_combined(
        p1: float = 0.01,
        p2: float = 0.05,
        p_readout: float = 0.02
    ) -> NoiseModel:
        """Create combined depolarizing + readout noise model."""
        
    @staticmethod
    def create_thermal(t1: float = 50e-6, t2: float = 70e-6, gate_time: float = 50e-9) -> NoiseModel:
        """Create thermal noise model (T1/T2 relaxation approximation)."""
```

### 1.3 Predefined Noise Configurations

```python
# Preset configurations matching current experiment needs
PRESET_NOISE_CONFIGS: Dict[str, NoiseConfig] = {
    "IDEAL": NoiseConfig("IDEAL", "none", {}, "No noise"),
    "DEP-LOW": NoiseConfig("DEP-LOW", "depolarizing", {"p1": 0.001, "p2": 0.005}, "Low depolarizing"),
    "DEP-MED": NoiseConfig("DEP-MED", "depolarizing", {"p1": 0.01, "p2": 0.05}, "Medium depolarizing"),
    "DEP-HIGH": NoiseConfig("DEP-HIGH", "depolarizing", {"p1": 0.05, "p2": 0.10}, "High depolarizing"),
    "READ-LOW": NoiseConfig("READ-LOW", "readout", {"p01": 0.01, "p10": 0.01}, "Low readout error"),
    "READ-MED": NoiseConfig("READ-MED", "readout", {"p01": 0.02, "p10": 0.02}, "Medium readout error"),
    "READ-HIGH": NoiseConfig("READ-HIGH", "readout", {"p01": 0.05, "p10": 0.05}, "High readout error"),
    "COMB-LOW": NoiseConfig("COMB-LOW", "combined", {"p1": 0.001, "p2": 0.005, "p_readout": 0.01}, "Low combined"),
    "COMB-MED": NoiseConfig("COMB-MED", "combined", {"p1": 0.01, "p2": 0.05, "p_readout": 0.02}, "Medium combined"),
    "COMB-HIGH": NoiseConfig("COMB-HIGH", "combined", {"p1": 0.05, "p2": 0.10, "p_readout": 0.05}, "High combined"),
}
```

---

## Phase 2: Experiment Utilities (`experiment_utils.py`)

### 2.1 QWARD Metrics Helper

```python
def calculate_qward_metrics(
    circuit: QuantumCircuit,
    strategies: Optional[List[Type]] = None
) -> Dict[str, Any]:
    """
    Calculate pre-runtime QWARD metrics for a circuit.
    
    Args:
        circuit: The quantum circuit to analyze
        strategies: Optional list of metric strategy classes (defaults to standard set)
        
    Returns:
        Dictionary with all QWARD metrics serialized for JSON
    """
```

### 2.2 Serialization Utilities

```python
def serialize_value(value: Any) -> Any:
    """Convert a value to JSON-serializable format."""

def serialize_metrics_dict(metrics_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Convert QWARD metrics DataFrames to JSON-serializable dictionaries."""
```

### 2.3 Experiment Constants

```python
@dataclass
class ExperimentDefaults:
    """Default experiment parameters."""
    shots: int = 1024
    num_runs: int = 10
    optimization_level: int = 0

DEFAULT_EXPERIMENT_PARAMS = ExperimentDefaults()
```

### 2.4 Base Result Classes (Optional - Phase 3)

Could consider creating base classes for experiment results:

```python
@dataclass
class BaseExperimentResult:
    """Base class for experiment results."""
    experiment_id: str
    config_id: str
    noise_model: str
    run_number: int
    timestamp: str
    num_qubits: int
    circuit_depth: int
    total_gates: int
    qward_metrics: Optional[Dict[str, Any]] = None
    shots: int = 1024
    execution_time_ms: float = 0.0
    counts: Dict[str, int] = None
    success_rate: float = 0.0
    success_count: int = 0

@dataclass
class BaseBatchResult:
    """Base class for batch results."""
    config_id: str
    noise_model: str
    num_runs: int
    shots_per_run: int
    mean_success_rate: float
    std_success_rate: float
    min_success_rate: float
    max_success_rate: float
    median_success_rate: float
```

---

## Phase 3: Refactor Existing Code

### 3.1 Update `executor.py`

Replace internal noise model methods with `NoiseModelGenerator`:

```python
from .noise_generator import NoiseModelGenerator, NoiseConfig

class QuantumCircuitExecutor:
    def _get_noise_model(self, noise_model_config, noise_level=0.05):
        if noise_model_config is None:
            return None
        elif isinstance(noise_model_config, NoiseModel):
            return noise_model_config
        elif isinstance(noise_model_config, str):
            # Use NoiseModelGenerator based on string type
            return NoiseModelGenerator.create_by_type(noise_model_config, noise_level)
        elif isinstance(noise_model_config, NoiseConfig):
            return NoiseModelGenerator.create_from_config(noise_model_config)
```

### 3.2 Update `qft_experiment.py`

```python
# Replace local imports with centralized imports
from qward.algorithms import NoiseModelGenerator, NoiseConfig
from qward.algorithms.experiment_utils import calculate_qward_metrics, serialize_value

# Remove local create_noise_model function
# Use NoiseModelGenerator.create_from_config(noise_config) instead
```

### 3.3 Update `grover_experiment.py`

Same pattern as QFT experiment - import from centralized module.

### 3.4 Update `__init__.py`

```python
from .noise_generator import (
    NoiseConfig,
    NoiseModelGenerator,
    PRESET_NOISE_CONFIGS,
    get_preset_noise_config,
)
from .experiment_utils import (
    calculate_qward_metrics,
    serialize_value,
    ExperimentDefaults,
    DEFAULT_EXPERIMENT_PARAMS,
)

__all__ = [
    # ... existing exports ...
    "NoiseConfig",
    "NoiseModelGenerator", 
    "PRESET_NOISE_CONFIGS",
    "get_preset_noise_config",
    "calculate_qward_metrics",
    "serialize_value",
    "ExperimentDefaults",
    "DEFAULT_EXPERIMENT_PARAMS",
]
```

---

## Implementation Order

### Step 1: Create `noise_generator.py` ⬅️ START HERE
- [ ] Create `NoiseConfig` dataclass
- [ ] Implement `NoiseModelGenerator` class with all noise types
- [ ] Add preset noise configurations
- [ ] Add unit tests

### Step 2: Create `experiment_utils.py`
- [ ] Move `calculate_qward_metrics` function
- [ ] Move `serialize_value` function
- [ ] Add experiment defaults
- [ ] Add unit tests

### Step 3: Refactor `executor.py`
- [ ] Import from `noise_generator`
- [ ] Replace internal noise methods with `NoiseModelGenerator`
- [ ] Ensure backward compatibility
- [ ] Update tests

### Step 4: Refactor `qft_experiment.py`
- [ ] Import from centralized modules
- [ ] Remove duplicated functions
- [ ] Update tests

### Step 5: Refactor `grover_experiment.py`
- [ ] Import from centralized modules
- [ ] Remove duplicated functions
- [ ] Update tests

### Step 6: Update `grover_configs.py` and `qft_configs.py`
- [ ] Import `NoiseConfig` from centralized location
- [ ] Remove local `NoiseConfig` definitions (if any)

---

## API Design Considerations

### Backward Compatibility

The refactoring should maintain backward compatibility:

```python
# Old way (still works via passthrough)
noise_model = create_noise_model(noise_config)

# New way (preferred)
noise_model = NoiseModelGenerator.create_from_config(noise_config)

# Direct creation (new capability)
noise_model = NoiseModelGenerator.create_depolarizing(p1=0.01, p2=0.05)
```

### Extensibility

Design allows easy addition of new noise types:

```python
class NoiseModelGenerator:
    @classmethod
    def register_noise_type(cls, name: str, factory: Callable) -> None:
        """Register a custom noise type factory."""
```

### Type Safety

Use proper type hints throughout:

```python
from typing import Optional, Dict, Any, Callable, Type
from qiskit_aer.noise import NoiseModel
```

---

## Testing Strategy

### Unit Tests for `noise_generator.py`

```python
class TestNoiseModelGenerator:
    def test_create_depolarizing_noise(self):
        """Test depolarizing noise model creation."""
        
    def test_create_readout_noise(self):
        """Test readout error model creation."""
        
    def test_create_combined_noise(self):
        """Test combined noise model creation."""
        
    def test_create_from_config(self):
        """Test config-based creation."""
        
    def test_preset_configs(self):
        """Test all preset configurations work."""
        
    def test_ideal_returns_none(self):
        """Test IDEAL config returns None."""
```

### Integration Tests

- Verify experiments still produce same results after refactoring
- Compare output distributions with old implementations

---

## File Sizes After Refactoring

| File | Before | After (Estimated) |
|------|--------|-------------------|
| `noise_generator.py` | N/A | ~200 lines |
| `experiment_utils.py` | N/A | ~100 lines |
| `executor.py` | 522 lines | ~350 lines |
| `qft_experiment.py` | 932 lines | ~850 lines |
| `grover_experiment.py` | 680 lines | ~600 lines |

---

## Benefits

1. **Single Source of Truth**: One implementation for noise models
2. **Easier Testing**: Test noise models once, use everywhere
3. **Better Maintainability**: Changes propagate automatically
4. **Clearer API**: Well-documented, typed interface
5. **Extensibility**: Easy to add new noise types
6. **Consistency**: All experiments use identical noise models
7. **Reduced Code**: ~150 lines removed from duplication

---

## Timeline Estimate

| Phase | Estimated Time |
|-------|---------------|
| Phase 1: noise_generator.py | 2-3 hours |
| Phase 2: experiment_utils.py | 1-2 hours |
| Phase 3: Refactor existing code | 2-3 hours |
| Testing & Validation | 2-3 hours |
| **Total** | **7-11 hours** |

---

## Next Steps

1. ✅ **Review this plan** and confirm approach
2. ✅ **Create `noise_generator.py`** with core functionality
3. ✅ **Create `experiment_utils.py`** with shared utilities
4. ✅ **Refactor experiment files** to use new modules
5. ✅ **Add research-justified noise parameters with references**
6. ⏳ **Update tests** and documentation

---

## Literature References for Noise Parameters

The noise model parameters are justified with the following empirical sources:

1. **IBM Research.** "Characterizing errors on qubit operations via iterative randomized benchmarking." (2016). Single-qubit gate fidelity ~99.9%.

2. **Arute et al.** "Quantum supremacy using a programmable superconducting processor." Nature 574, 505-510 (2019). Google Sycamore benchmarks.

3. **IBM Research.** "Benchmarking the noise sensitivity of different parametric two-qubit gates." CZ/iSWAP gates: 0.9%-1.3% error.

4. **Geller & Zhou.** "Logical error rate in the Pauli twirling approximation." Nature Scientific Reports 3, 14670 (2013). Pauli approximation validity.

5. **Swiadek et al.** "Enhancing Dispersive Readout of Superconducting Qubits." arXiv:2307.07765 (2023). Achieved 0.25% readout error.

6. **Lienhard et al.** "Model-based Optimization of Superconducting Qubit Readout." arXiv:2308.02079 (2023). ~1.5% error for 17-qubit simultaneous readout.

7. **Wang et al.** "High-fidelity gates on atomic qubits." Nature 620, 734-740 (2023). Trapped ion single-qubit error < 1e-4.

8. **IonQ.** "IonQ Aria System Specifications." (2024). ~99.4% two-qubit fidelity.

9. **Escofet et al.** "An Accurate Efficient Analytic Model of Fidelity under Depolarizing Noise." arXiv:2503.06693 (2025).

10. **Tomita & Svore.** "Low-distance surface codes under realistic quantum noise." Phys. Rev. A 90, 062320 (2014).

---

*Plan created: 2026-01-20*
*Implementation completed: 2026-01-21*
*Author: QWARD Development Team*
