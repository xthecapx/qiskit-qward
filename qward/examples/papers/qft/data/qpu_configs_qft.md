# QFT QPU Configuration Report

**Data source:** qward/examples/papers/qft/data/raw
**Noise models:** IBM-HERON-R1, IBM-HERON-R2, IBM-HERON-R3, RIGETTI-ANKAA3
**Reference:** IBM-HERON-R2
**Configs analyzed:** 25


## Configuration Analysis

| Config | Qubits | Depth | Gates | IBM-HERON-R1 | IBM-HERON-R2 | IBM-HERON-R3 | RIGETTI-ANKAA3 | Random | Region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SR2** | 2 | 10 | 15 | **93.2%** | **96.1%** | **98.6%** | **94.3%** | 25.0% | Region 1 ✅ |
| **SP3-P2** | 3 | 21 | 32 | **96.7%** | **98.1%** | **99.3%** | **96.7%** | 50.0% | Region 3 ❌ |
| **SR3** | 3 | 14 | 23 | **89.9%** | **94.4%** | **97.7%** | **90.9%** | 12.5% | Region 1 ✅ |
| **IV4-0000** | 4 | 17 | 32 | **85.9%** | **91.9%** | **96.5%** | **86.3%** | 6.2% | Region 1 ✅ |
| **IV4-0001** | 4 | 18 | 33 | **85.5%** | **91.8%** | **97.1%** | **86.5%** | 6.2% | Region 1 ✅ |
| **IV4-0101** | 4 | 18 | 34 | **85.3%** | **91.6%** | **96.3%** | **86.5%** | 6.2% | Region 1 ✅ |
| **IV4-1010** | 4 | 18 | 34 | **85.5%** | **91.5%** | **96.3%** | **86.2%** | 6.2% | Region 1 ✅ |
| **IV4-1111** | 4 | 18 | 36 | **85.7%** | **91.5%** | **96.5%** | **85.6%** | 6.2% | Region 1 ✅ |
| **PV4-P2** | 4 | 27 | 44 | **92.5%** | **95.8%** | **98.1%** | **92.5%** | 50.0% | Region 3 ❌ |
| **PV4-P4** | 4 | 27 | 44 | **96.0%** | **97.9%** | **99.2%** | **96.3%** | 25.0% | Region 1 ✅ |
| **PV4-P8** | 4 | 27 | 44 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 12.5% | Region 1 ✅ |
| **SP4-P2** | 4 | 27 | 44 | **92.3%** | **95.7%** | **98.4%** | **92.3%** | 50.0% | Region 3 ❌ |
| **SP4-P4** | 4 | 27 | 44 | **96.4%** | **98.0%** | **99.1%** | **96.7%** | 25.0% | Region 1 ✅ |
| **SR4** | 4 | 18 | 34 | **86.0%** | **91.5%** | **96.2%** | **86.1%** | 6.2% | Region 1 ✅ |
| **SP5-P4** | 5 | 33 | 56 | **92.5%** | **95.4%** | **98.2%** | **92.2%** | 25.0% | Region 1 ✅ |
| **SR5** | 5 | 22 | 46 | **82.1%** | **88.9%** | **95.2%** | **81.8%** | 3.1% | Region 1 ✅ |
| **PV6-P16** | 6 | 39 | 70 | **96.1%** | **97.5%** | **99.1%** | **95.8%** | 6.2% | Region 1 ✅ |
| **PV6-P2** | 6 | 39 | 70 | **84.5%** | **90.9%** | **96.5%** | **84.6%** | 50.0% | Region 3 ❌ |
| **PV6-P4** | 6 | 39 | 70 | **88.7%** | **93.2%** | **97.0%** | **88.1%** | 25.0% | Region 1 ✅ |
| **PV6-P8** | 6 | 39 | 70 | **91.9%** | **95.4%** | **98.1%** | **91.9%** | 12.5% | Region 1 ✅ |
| **SP6-P4** | 6 | 39 | 70 | **88.7%** | **93.2%** | **97.0%** | **88.2%** | 25.0% | Region 1 ✅ |
| **SP6-P8** | 6 | 39 | 70 | **92.4%** | **95.2%** | **97.7%** | **92.1%** | 12.5% | Region 1 ✅ |
| **SR6** | 6 | 26 | 61 | **77.9%** | **85.6%** | **93.4%** | **75.9%** | 1.6% | Region 1 ✅ |
| **SR7** | 7 | 30 | 77 | **73.5%** | **82.6%** | **92.2%** | **71.5%** | 0.8% | Region 1 ✅ |
| **SR8** | 8 | 34 | 96 | **69.3%** | **79.6%** | **90.5%** | **65.4%** | 0.4% | Region 1 ✅ |

## Region Summary (Reference: IBM-HERON-R2)

### Region 1 ✅ (21 configs)
**Worth running on QPU (algorithm works)**

Configs: PV4-P8, SP4-P4, PV4-P4, PV6-P16, SR2, SP5-P4, PV6-P8, SP6-P8, SR3, SP6-P4, PV6-P4, IV4-0000, IV4-0001, IV4-0101, IV4-1111, SR4, IV4-1010, SR5, SR6, SR7, SR8

Success range: 79.6% - 100.0%

### Region 2 ⚠️ (0 configs)
**Marginal - boundary exploration**

### Region 3 ❌ (4 configs)
**Algorithm fails - document limits only**

Configs: SP3-P2, PV4-P2, SP4-P2, PV6-P2

Success range: 90.9% - 98.1%


## QPU Configurations (Python)

```python
# QFT QPU Experiment Configurations
# Generated from simulator data analysis
# Reference noise model: IBM-HERON-R2

from qward.examples.papers.qft.qft_configs import get_config, get_noise_config

# Configurations to run on QPU (prioritized by expected success)
QPU_CONFIGS = [
    # Priority 1: PV4-P8 - 100.0% expected (Region 1)
    {
        "config_id": "PV4-P8",
        "num_qubits": 4,
        "test_mode": "period_detection",
        "period": 8,
        "expected_success": 1.0000,
        "circuit_depth": 27,
        "total_gates": 44,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 2: SP4-P4 - 98.0% expected (Region 1)
    {
        "config_id": "SP4-P4",
        "num_qubits": 4,
        "test_mode": "period_detection",
        "period": 4,
        "expected_success": 0.9802,
        "circuit_depth": 27,
        "total_gates": 44,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 3: PV4-P4 - 97.9% expected (Region 1)
    {
        "config_id": "PV4-P4",
        "num_qubits": 4,
        "test_mode": "period_detection",
        "period": 4,
        "expected_success": 0.9794,
        "circuit_depth": 27,
        "total_gates": 44,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 4: PV6-P16 - 97.5% expected (Region 1)
    {
        "config_id": "PV6-P16",
        "num_qubits": 6,
        "test_mode": "period_detection",
        "period": 16,
        "expected_success": 0.9753,
        "circuit_depth": 39,
        "total_gates": 70,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 5: SR2 - 96.1% expected (Region 1)
    {
        "config_id": "SR2",
        "num_qubits": 2,
        "test_mode": "roundtrip",
        "input_state": "01",
        "expected_success": 0.9610,
        "circuit_depth": 10,
        "total_gates": 15,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 6: SP5-P4 - 95.4% expected (Region 1)
    {
        "config_id": "SP5-P4",
        "num_qubits": 5,
        "test_mode": "period_detection",
        "period": 4,
        "expected_success": 0.9543,
        "circuit_depth": 33,
        "total_gates": 56,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 7: PV6-P8 - 95.4% expected (Region 1)
    {
        "config_id": "PV6-P8",
        "num_qubits": 6,
        "test_mode": "period_detection",
        "period": 8,
        "expected_success": 0.9538,
        "circuit_depth": 39,
        "total_gates": 70,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 8: SP6-P8 - 95.2% expected (Region 1)
    {
        "config_id": "SP6-P8",
        "num_qubits": 6,
        "test_mode": "period_detection",
        "period": 8,
        "expected_success": 0.9519,
        "circuit_depth": 39,
        "total_gates": 70,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 9: SR3 - 94.4% expected (Region 1)
    {
        "config_id": "SR3",
        "num_qubits": 3,
        "test_mode": "roundtrip",
        "input_state": "101",
        "expected_success": 0.9441,
        "circuit_depth": 14,
        "total_gates": 23,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 10: SP6-P4 - 93.2% expected (Region 1)
    {
        "config_id": "SP6-P4",
        "num_qubits": 6,
        "test_mode": "period_detection",
        "period": 4,
        "expected_success": 0.9319,
        "circuit_depth": 39,
        "total_gates": 70,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 11: PV6-P4 - 93.2% expected (Region 1)
    {
        "config_id": "PV6-P4",
        "num_qubits": 6,
        "test_mode": "period_detection",
        "period": 4,
        "expected_success": 0.9315,
        "circuit_depth": 39,
        "total_gates": 70,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 12: IV4-0000 - 91.9% expected (Region 1)
    {
        "config_id": "IV4-0000",
        "num_qubits": 4,
        "test_mode": "roundtrip",
        "input_state": "0000",
        "expected_success": 0.9188,
        "circuit_depth": 17,
        "total_gates": 32,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 13: IV4-0001 - 91.8% expected (Region 1)
    {
        "config_id": "IV4-0001",
        "num_qubits": 4,
        "test_mode": "roundtrip",
        "input_state": "0001",
        "expected_success": 0.9179,
        "circuit_depth": 18,
        "total_gates": 33,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 14: IV4-0101 - 91.6% expected (Region 1)
    {
        "config_id": "IV4-0101",
        "num_qubits": 4,
        "test_mode": "roundtrip",
        "input_state": "0101",
        "expected_success": 0.9161,
        "circuit_depth": 18,
        "total_gates": 34,
        "runs": 5,
        "region": "Region 1",
    },
    # Priority 15: IV4-1111 - 91.5% expected (Region 1)
    {
        "config_id": "IV4-1111",
        "num_qubits": 4,
        "test_mode": "roundtrip",
        "input_state": "1111",
        "expected_success": 0.9153,
        "circuit_depth": 18,
        "total_gates": 36,
        "runs": 5,
        "region": "Region 1",
    },
]

# Quick access to configurations
def get_qpu_config(config_id: str):
    """Get full QFTExperimentConfig for a QPU experiment."""
    return get_config(config_id)

# Total runs: 75
# Estimated QPU time: ~375 seconds
```