# GROVER QPU Configuration Report

**Data source:** qward/examples/papers/grover/data/simulator/raw
**Noise models:** IBM-HERON-R1, IBM-HERON-R2, IBM-HERON-R3, RIGETTI-ANKAA3
**Reference:** IBM-HERON-R2
**Configs analyzed:** 24


## Configuration Analysis

| Config | Qubits | Depth | Gates | IBM-HERON-R1 | IBM-HERON-R2 | IBM-HERON-R3 | RIGETTI-ANKAA3 | Random | Region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S2-1** | 2 | 12 | 21 | **93.0%** | **96.7%** | **98.6%** | **93.2%** | 25.0% | Region 1 ✅ |
| **ASYM-1** | 3 | 46 | 82 | **88.6%** | **92.0%** | **96.0%** | **84.8%** | 25.0% | Region 1 ✅ |
| **ASYM-2** | 3 | 46 | 80 | **86.3%** | **90.5%** | **95.8%** | **83.7%** | 25.0% | Region 1 ✅ |
| **H3-0** | 3 | 62 | 113 | **77.9%** | **82.6%** | **89.1%** | **74.1%** | 12.5% | Region 1 ✅ |
| **H3-1** | 3 | 62 | 109 | **78.6%** | **82.7%** | **89.3%** | **73.9%** | 12.5% | Region 1 ✅ |
| **H3-2** | 3 | 62 | 105 | **79.4%** | **83.3%** | **89.7%** | **75.3%** | 12.5% | Region 1 ✅ |
| **H3-3** | 3 | 58 | 101 | **79.7%** | **83.7%** | **89.1%** | **74.9%** | 12.5% | Region 1 ✅ |
| **M3-1** | 3 | 62 | 113 | **78.6%** | **82.3%** | **88.9%** | **74.3%** | 12.5% | Region 1 ✅ |
| **M3-2** | 3 | 44 | 78 | **86.3%** | **90.5%** | **95.4%** | **84.4%** | 25.0% | Region 1 ✅ |
| **M3-4** | 3 | 2 | 9 | **51.2%** | **50.4%** | **48.9%** | **49.6%** | 50.0% | Region 3 ❌ |
| **S3-1** | 3 | 62 | 105 | **78.7%** | **82.6%** | **89.6%** | **74.6%** | 12.5% | Region 1 ✅ |
| **SYM-1** | 3 | 44 | 78 | **86.5%** | **90.2%** | **95.9%** | **83.9%** | 25.0% | Region 1 ✅ |
| **SYM-2** | 3 | 44 | 78 | **86.4%** | **90.3%** | **95.6%** | **83.2%** | 25.0% | Region 1 ✅ |
| **H4-0** | 4 | 178 | 281 | **64.2%** | **68.2%** | **83.1%** | **51.1%** | 6.2% | Region 1 ✅ |
| **H4-2** | 4 | 172 | 269 | **65.3%** | **69.1%** | **82.7%** | **52.4%** | 6.2% | Region 1 ✅ |
| **H4-4** | 4 | 172 | 257 | **65.6%** | **70.0%** | **82.9%** | **53.1%** | 6.2% | Region 1 ✅ |
| **M4-1** | 4 | 178 | 281 | **65.2%** | **69.2%** | **82.2%** | **52.3%** | 6.2% | Region 1 ✅ |
| **M4-2** | 4 | 172 | 257 | **66.1%** | **71.2%** | **82.8%** | **56.2%** | 12.5% | Region 1 ✅ |
| **M4-4** | 4 | 142 | 208 | **74.7%** | **79.1%** | **89.9%** | **66.2%** | 25.0% | Region 1 ✅ |
| **S4-1** | 4 | 178 | 269 | **65.3%** | **70.6%** | **82.9%** | **52.9%** | 6.2% | Region 1 ✅ |
| **S5-1** | 5 | 474 | 685 | **43.0%** | **44.8%** | **69.3%** | **23.3%** | 3.1% | Region 2 ⚠️ |
| **S6-1** | 6 | 1827 | 2751 | **5.3%** | **5.1%** | **23.4%** | **1.8%** | 1.6% | Region 2 ⚠️ |
| **S7-1** | 7 | 3586 | 5745 | **0.9%** | **1.0%** | **5.0%** | **0.8%** | 0.8% | Region 3 ❌ |
| **S8-1** | 8 | 8946 | 13531 | **0.4%** | **0.4%** | **0.6%** | **0.4%** | 0.4% | Region 3 ❌ |

## Region Summary (Reference: IBM-HERON-R2)

### Region 1 ✅ (19 configs)
**Worth running on QPU (algorithm works)**

Configs: S2-1, ASYM-1, ASYM-2, M3-2, SYM-2, SYM-1, H3-3, H3-2, H3-1, H3-0, S3-1, M3-1, M4-4, M4-2, S4-1, H4-4, M4-1, H4-2, H4-0

Success range: 68.2% - 96.7%

### Region 2 ⚠️ (2 configs)
**Marginal - boundary exploration**

Configs: S5-1, S6-1

Success range: 5.1% - 44.8%

### Region 3 ❌ (3 configs)
**Algorithm fails - document limits only**

Configs: M3-4, S7-1, S8-1

Success range: 0.4% - 50.4%


## QPU Configurations (Python)

```python
# Grover QPU Experiment Configurations
# Generated from simulator data analysis
# Reference noise model: IBM-HERON-R2

from qward.examples.papers.grover.grover_configs import get_config, get_noise_config

# Configurations to run on QPU (prioritized by expected success)
QPU_CONFIGS = [
    # Priority 1: S2-1 - 96.7% expected (Region 1)
    {
        "config_id": "S2-1",
        "num_qubits": 2,
        "marked_states": ['01'],
        "expected_success": 0.9666,
        "circuit_depth": 12,
        "total_gates": 21,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 2: ASYM-1 - 92.0% expected (Region 1)
    {
        "config_id": "ASYM-1",
        "num_qubits": 3,
        "marked_states": ['000', '001'],
        "expected_success": 0.9203,
        "circuit_depth": 46,
        "total_gates": 82,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 3: ASYM-2 - 90.5% expected (Region 1)
    {
        "config_id": "ASYM-2",
        "num_qubits": 3,
        "marked_states": ['000', '011'],
        "expected_success": 0.9054,
        "circuit_depth": 46,
        "total_gates": 80,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 4: M3-2 - 90.5% expected (Region 1)
    {
        "config_id": "M3-2",
        "num_qubits": 3,
        "marked_states": ['000', '111'],
        "expected_success": 0.9046,
        "circuit_depth": 44,
        "total_gates": 78,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 5: SYM-2 - 90.3% expected (Region 1)
    {
        "config_id": "SYM-2",
        "num_qubits": 3,
        "marked_states": ['001', '110'],
        "expected_success": 0.9026,
        "circuit_depth": 44,
        "total_gates": 78,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 6: SYM-1 - 90.2% expected (Region 1)
    {
        "config_id": "SYM-1",
        "num_qubits": 3,
        "marked_states": ['000', '111'],
        "expected_success": 0.9021,
        "circuit_depth": 44,
        "total_gates": 78,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 7: H3-3 - 83.7% expected (Region 1)
    {
        "config_id": "H3-3",
        "num_qubits": 3,
        "marked_states": ['111'],
        "expected_success": 0.8371,
        "circuit_depth": 58,
        "total_gates": 101,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 8: H3-2 - 83.3% expected (Region 1)
    {
        "config_id": "H3-2",
        "num_qubits": 3,
        "marked_states": ['011'],
        "expected_success": 0.8333,
        "circuit_depth": 62,
        "total_gates": 105,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 9: H3-1 - 82.7% expected (Region 1)
    {
        "config_id": "H3-1",
        "num_qubits": 3,
        "marked_states": ['001'],
        "expected_success": 0.8266,
        "circuit_depth": 62,
        "total_gates": 109,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 10: H3-0 - 82.6% expected (Region 1)
    {
        "config_id": "H3-0",
        "num_qubits": 3,
        "marked_states": ['000'],
        "expected_success": 0.8262,
        "circuit_depth": 62,
        "total_gates": 113,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 11: S3-1 - 82.6% expected (Region 1)
    {
        "config_id": "S3-1",
        "num_qubits": 3,
        "marked_states": ['011'],
        "expected_success": 0.8257,
        "circuit_depth": 62,
        "total_gates": 105,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 12: M3-1 - 82.3% expected (Region 1)
    {
        "config_id": "M3-1",
        "num_qubits": 3,
        "marked_states": ['000'],
        "expected_success": 0.8234,
        "circuit_depth": 62,
        "total_gates": 113,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 13: M4-4 - 79.1% expected (Region 1)
    {
        "config_id": "M4-4",
        "num_qubits": 4,
        "marked_states": ['0000', '0011', '1100', '1111'],
        "expected_success": 0.7913,
        "circuit_depth": 142,
        "total_gates": 208,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 14: M4-2 - 71.2% expected (Region 1)
    {
        "config_id": "M4-2",
        "num_qubits": 4,
        "marked_states": ['0000', '1111'],
        "expected_success": 0.7118,
        "circuit_depth": 172,
        "total_gates": 257,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 15: S4-1 - 70.6% expected (Region 1)
    {
        "config_id": "S4-1",
        "num_qubits": 4,
        "marked_states": ['0110'],
        "expected_success": 0.7061,
        "circuit_depth": 178,
        "total_gates": 269,
        "runs": 5,
        "region": "Region 1",
    },
]

# Quick access to configurations
def get_qpu_config(config_id: str):
    """Get full ExperimentConfig for a QPU experiment."""
    return get_config(config_id)

# Total experiments
# Total runs: 75
# Estimated QPU time: ~375 seconds
```