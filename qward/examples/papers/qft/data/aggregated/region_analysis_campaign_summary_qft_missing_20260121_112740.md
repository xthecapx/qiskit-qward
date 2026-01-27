# QFT Region Analysis Report

**Input file:** campaign_summary_qft_missing_20260121_112740.json
**Reference noise model:** RIGETTI-ANKAA3
**Region 1 threshold:** 0.5
**Advantage multiplier:** 2.0x random chance


## QFT Configuration Summary

| Config | Qubits | Marked | RIGETTI-ANKAA3 | Random | Region |
| --- | --- | --- | --- | --- | --- |
| **SR2** | 2 | 1 | **94.3%** | 25.0% | Region 1 ✅ |
| **SP3-P2** | 3 | 4 | **96.7%** | 50.0% | Region 1 ✅ |
| **SR3** | 3 | 1 | **90.9%** | 12.5% | Region 1 ✅ |
| **IV4-0000** | 4 | 1 | **86.3%** | 6.2% | Region 1 ✅ |
| **IV4-0001** | 4 | 1 | **86.5%** | 6.2% | Region 1 ✅ |
| **IV4-0101** | 4 | 1 | **86.5%** | 6.2% | Region 1 ✅ |
| **IV4-1010** | 4 | 1 | **86.2%** | 6.2% | Region 1 ✅ |
| **IV4-1111** | 4 | 1 | **85.6%** | 6.2% | Region 1 ✅ |
| **PV4-P2** | 4 | 8 | **92.5%** | 50.0% | Region 1 ✅ |
| **PV4-P4** | 4 | 4 | **96.3%** | 25.0% | Region 1 ✅ |
| **PV4-P8** | 4 | 2 | **100.0%** | 12.5% | Region 1 ✅ |
| **SP4-P2** | 4 | 8 | **92.3%** | 50.0% | Region 1 ✅ |
| **SP4-P4** | 4 | 4 | **96.7%** | 25.0% | Region 1 ✅ |
| **SR4** | 4 | 1 | **86.1%** | 6.2% | Region 1 ✅ |
| **SP5-P4** | 5 | 8 | **92.2%** | 25.0% | Region 1 ✅ |
| **SR5** | 5 | 1 | **81.8%** | 3.1% | Region 1 ✅ |
| **PV6-P16** | 6 | 4 | **95.8%** | 6.2% | Region 1 ✅ |
| **PV6-P2** | 6 | 32 | **84.6%** | 50.0% | Region 1 ✅ |
| **PV6-P4** | 6 | 16 | **88.1%** | 25.0% | Region 1 ✅ |
| **PV6-P8** | 6 | 8 | **91.9%** | 12.5% | Region 1 ✅ |
| **SP6-P4** | 6 | 16 | **88.2%** | 25.0% | Region 1 ✅ |
| **SP6-P8** | 6 | 8 | **92.1%** | 12.5% | Region 1 ✅ |
| **SR6** | 6 | 1 | **75.9%** | 1.6% | Region 2 ⚠️ |
| **SR7** | 7 | 1 | **71.5%** | 0.8% | Region 2 ⚠️ |
| **SR8** | 8 | 1 | **65.4%** | 0.4% | Region 2 ⚠️ |

## Region Summary (using RIGETTI-ANKAA3)

### Region 1 ✅ (22 configs)
**Signal Dominant - Worth running on QPU**

Configs: IV4-0000, IV4-0001, IV4-0101, IV4-1010, IV4-1111, PV4-P2, PV4-P4, PV4-P8, PV6-P16, PV6-P2, PV6-P4, PV6-P8, SP3-P2, SP4-P2, SP4-P4, SP5-P4, SP6-P4, SP6-P8, SR2, SR3, SR4, SR5

Success range: 81.8% - 100.0%

### Region 2 ⚠️ (3 configs)
**Transition Zone - Marginal quantum advantage**

Configs: SR6, SR7, SR8

Success range: 65.4% - 75.9%

### Region 3 ❌ (0 configs)
**Noise Dominant - Algorithm fails, use for documenting limits**


## Noise Model Comparison

| Noise Model | Avg Success | Min | Max | Region 1 Count | Region 3 Count |
| --- | --- | --- | --- | --- | --- |
| RIGETTI-ANKAA3 | 88.2% | 65.4% | 100.0% | 22 | 0 |

## QPU Experiment Recommendations

Prioritized list of configurations to run on real QPU:

| Priority | Config | Qubits | Expected Success | Region | Runs Suggested |
| --- | --- | --- | --- | --- | --- |
| 1 | PV4-P8 | 4 | 100.0% | Region 1 ✅ | 5 |
| 2 | SP4-P4 | 4 | 96.7% | Region 1 ✅ | 5 |
| 3 | SP3-P2 | 3 | 96.7% | Region 1 ✅ | 5 |
| 4 | PV4-P4 | 4 | 96.3% | Region 1 ✅ | 5 |
| 5 | PV6-P16 | 6 | 95.8% | Region 1 ✅ | 5 |
| 6 | SR2 | 2 | 94.3% | Region 1 ✅ | 5 |
| 7 | PV4-P2 | 4 | 92.5% | Region 1 ✅ | 5 |
| 8 | SP4-P2 | 4 | 92.3% | Region 1 ✅ | 5 |
| 9 | SP5-P4 | 5 | 92.2% | Region 1 ✅ | 5 |
| 10 | SP6-P8 | 6 | 92.1% | Region 1 ✅ | 5 |
| 11 | PV6-P8 | 6 | 91.9% | Region 1 ✅ | 5 |
| 12 | SR3 | 3 | 90.9% | Region 1 ✅ | 5 |
| 13 | SP6-P4 | 6 | 88.2% | Region 1 ✅ | 5 |
| 14 | PV6-P4 | 6 | 88.1% | Region 1 ✅ | 5 |
| 15 | IV4-0101 | 4 | 86.5% | Region 1 ✅ | 5 |