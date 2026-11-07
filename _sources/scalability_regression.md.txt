# Which algorithms lose success at similar rates?

On IBM Fez, **QFT and phase estimation grouped together as logical two-qubit gate count grew. At increasing circuit width, the pair was Bernstein–Vazirani and vTP.** The resource used to measure growth changes which algorithms have similar decay. This baseline covers 1,168 jobs, 184 configurations and all 28 pairs of eight algorithms.

## What circuit structure leads us to expect

| Relationship | Why compare them? | Fez success-rate clusters |
| --- | --- | --- |
| Fourier family: QFT and phase estimation (PE) | PE uses an inverse QFT. | Together on 2 of 7 supported resource metrics. |
| Oracle family: Grover and Bernstein–Vazirani (BV) | Both query an oracle. | Together on 0 of 2 supported metrics. |
| Entanglement family: vTP and GHZ | Both distribute entanglement across qubits. | Together on 1 of 7 supported metrics. |
| Fourier extension: Shor with QFT or PE | Shor order finding uses phase estimation and an inverse QFT. | QFT and Shor grouped by width in one period. Both Shor comparisons lack a second eligible period. |
| Mirror-circuit comparison: vTP and random volumetric (RV) | Both use inverse operations for validation. Fez vTP used fixed payload gates around teleportation. RV used random two-qubit layers. | Together on 0 of 3 supported metrics. At equal width their fitted losses differed by 23.35 percentage points. |

The first three rows are the theory-based families in the thesis. Shor tests a composed circuit. RV provides a hardware control.

## What deterministic clustering found

These are all the multi-algorithm groups for **success rate** at the default Fez settings. The gap measures how far apart the two fitted losses were across their shared resource range, in percentage points (pp).

| Group found | Resource used to measure growth | Median gap |
| --- | --- | ---: |
| QFT and PE | Logical two-qubit gate count / logical entanglement ratio | 4.34 / 4.39 pp |
| vTP and GHZ | Logical depth | 1.09 pp |
| BV and vTP | Total logical qubits | 3.87 pp |
| GHZ and QFT | Logical operation count | 3.13 pp |
| BV and GHZ | Compiled depth / compiled two-qubit depth | 3.14 / 4.57 pp |

**The target matters too.** At equal width, switching from success rate to Hellinger fidelity retained BV and vTP and added QFT and PE. Each target therefore needs its own grouping.

**How to read a group.** Within each collection period, fit a line to each algorithm on at least three shared resource values. Compare the fitted changes from the first shared value to the last. Deterministic complete-linkage clustering uses the median gap across at least two periods and joins a group only when every pair is within the exploratory 5-point tolerance. Short shared ranges can produce small gaps. The export also includes 2.5- and 10-point tolerances.

## Dashboard handoff

Filter by **backend, resource metric and outcome**. Show the theoretical relationship beside the observed group, median gap and number of eligible periods. Distinguish a singleton (compared, no group formed) from insufficient overlap. Mark one-period results as provisional.

Run `python -m qward.examples.papers.scalability_regression --decay-only`. Under `report/scalability_regression/decay/`, use `groups/decay_groups.json` for groups and pairwise gaps, `matched_levels.csv` for all supported pairs’ observed points and fitted lines, and `groups/pair_recurrence.csv` for the grouped/supported counts above.
