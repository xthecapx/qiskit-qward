# GROVER Region Analysis Report

**Input file:** campaign_summary_qpu_missing_20260121_092108.json
**Reference noise model:** IBM-HERON-R2
**Region 1 threshold:** 0.5
**Advantage multiplier:** 2.0x random chance


## GROVER Configuration Summary

| Config | Qubits | Marked | IBM-HERON-R3 | IBM-HERON-R2 | IBM-HERON-R1 | RIGETTI-ANKAA3 | Random | Region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S2-1** | 2 | 1 | **98.6%** | **96.7%** | **93.0%** | **93.2%** | 25.0% | Region 1 ✅ |
| **ASYM-1** | 3 | 2 | **96.0%** | **92.0%** | **88.6%** | **84.8%** | 25.0% | Region 1 ✅ |
| **ASYM-2** | 3 | 2 | **95.8%** | **90.5%** | **86.3%** | **83.7%** | 25.0% | Region 1 ✅ |
| **H3-0** | 3 | 1 | **89.1%** | **82.6%** | **77.9%** | **74.1%** | 12.5% | Region 1 ✅ |
| **H3-1** | 3 | 1 | **89.3%** | **82.7%** | **78.6%** | **73.9%** | 12.5% | Region 1 ✅ |
| **H3-2** | 3 | 1 | **89.7%** | **83.3%** | **79.4%** | **75.3%** | 12.5% | Region 1 ✅ |
| **H3-3** | 3 | 1 | **89.1%** | **83.7%** | **79.7%** | **74.9%** | 12.5% | Region 1 ✅ |
| **M3-1** | 3 | 1 | **88.9%** | **82.3%** | **78.6%** | **74.3%** | 12.5% | Region 1 ✅ |
| **M3-2** | 3 | 2 | **95.4%** | **90.5%** | **86.3%** | **84.4%** | 25.0% | Region 1 ✅ |
| **M3-4** | 3 | 4 | **48.9%** | **50.4%** | **51.2%** | **49.6%** | 50.0% | Region 3 ❌ |
| **S3-1** | 3 | 1 | **89.6%** | **82.6%** | **78.7%** | **74.6%** | 12.5% | Region 1 ✅ |
| **SYM-1** | 3 | 2 | **95.9%** | **90.2%** | **86.5%** | **83.9%** | 25.0% | Region 1 ✅ |
| **SYM-2** | 3 | 2 | **95.6%** | **90.3%** | **86.4%** | **83.2%** | 25.0% | Region 1 ✅ |
| **H4-0** | 4 | 1 | **83.1%** | **68.2%** | **64.2%** | **51.1%** | 6.2% | Region 1 ✅ |
| **H4-2** | 4 | 1 | **82.7%** | **69.1%** | **65.3%** | **52.4%** | 6.2% | Region 1 ✅ |
| **H4-4** | 4 | 1 | **82.9%** | **70.0%** | **65.6%** | **53.1%** | 6.2% | Region 1 ✅ |
| **M4-1** | 4 | 1 | **82.2%** | **69.2%** | **65.2%** | **52.3%** | 6.2% | Region 1 ✅ |
| **M4-2** | 4 | 2 | **82.8%** | **71.2%** | **66.1%** | **56.2%** | 12.5% | Region 1 ✅ |
| **M4-4** | 4 | 4 | **89.9%** | **79.1%** | **74.7%** | **66.2%** | 25.0% | Region 1 ✅ |
| **S4-1** | 4 | 1 | **82.9%** | **70.6%** | **65.3%** | **52.9%** | 6.2% | Region 1 ✅ |
| **S5-1** | 5 | 1 | **69.3%** | **44.8%** | **43.0%** | **23.3%** | 3.1% | Region 2 ⚠️ |
| **S6-1** | 6 | 1 | **23.4%** | **5.1%** | **5.3%** | **1.8%** | 1.6% | Region 2 ⚠️ |
| **S7-1** | 7 | 1 | **5.0%** | **1.0%** | **0.9%** | **0.8%** | 0.8% | Region 3 ❌ |
| **S8-1** | 8 | 1 | **0.6%** | **0.4%** | **0.4%** | **0.4%** | 0.4% | Region 3 ❌ |

## Region Summary (using IBM-HERON-R2)

### Region 1 ✅ (19 configs)
**Signal Dominant - Worth running on QPU**

Configs: ASYM-1, ASYM-2, H3-0, H3-1, H3-2, H3-3, H4-0, H4-2, H4-4, M3-1, M3-2, M4-1, M4-2, M4-4, S2-1, S3-1, S4-1, SYM-1, SYM-2

Success range: 68.2% - 96.7%

### Region 2 ⚠️ (2 configs)
**Transition Zone - Marginal quantum advantage**

Configs: S5-1, S6-1

Success range: 5.1% - 44.8%

### Region 3 ❌ (3 configs)
**Noise Dominant - Algorithm fails, use for documenting limits**

Configs: M3-4, S7-1, S8-1

Success range: 0.4% - 50.4%


## Noise Model Comparison

| Noise Model | Avg Success | Min | Max | Region 1 Count | Region 3 Count |
| --- | --- | --- | --- | --- | --- |
| IBM-HERON-R3 | 76.9% | 0.6% | 98.6% | 20 | 2 |
| IBM-HERON-R2 | 68.6% | 0.4% | 96.7% | 19 | 3 |
| IBM-HERON-R1 | 65.3% | 0.4% | 93.0% | 19 | 3 |
| RIGETTI-ANKAA3 | 59.2% | 0.4% | 93.2% | 19 | 4 |

## QPU Experiment Recommendations

Prioritized list of configurations to run on real QPU:

| Priority | Config | Qubits | Expected Success | Region | Runs Suggested |
| --- | --- | --- | --- | --- | --- |
| 1 | S2-1 | 2 | 96.7% | Region 1 ✅ | 5 |
| 2 | ASYM-1 | 3 | 92.0% | Region 1 ✅ | 5 |
| 3 | ASYM-2 | 3 | 90.5% | Region 1 ✅ | 5 |
| 4 | M3-2 | 3 | 90.5% | Region 1 ✅ | 5 |
| 5 | SYM-2 | 3 | 90.3% | Region 1 ✅ | 5 |
| 6 | SYM-1 | 3 | 90.2% | Region 1 ✅ | 5 |
| 7 | H3-3 | 3 | 83.7% | Region 1 ✅ | 5 |
| 8 | H3-2 | 3 | 83.3% | Region 1 ✅ | 5 |
| 9 | H3-1 | 3 | 82.7% | Region 1 ✅ | 5 |
| 10 | H3-0 | 3 | 82.6% | Region 1 ✅ | 5 |
| 11 | S3-1 | 3 | 82.6% | Region 1 ✅ | 5 |
| 12 | M3-1 | 3 | 82.3% | Region 1 ✅ | 5 |
| 13 | M4-4 | 4 | 79.1% | Region 1 ✅ | 5 |
| 14 | M4-2 | 4 | 71.2% | Region 1 ✅ | 5 |
| 15 | S4-1 | 4 | 70.6% | Region 1 ✅ | 5 |