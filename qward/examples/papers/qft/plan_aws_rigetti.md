# QFT on AWS Rigetti (Ankaa-3): Execution Plan

This plan applies lessons from the Grover AWS Rigetti campaign: start small, use optimization level 3, iterate by qubit count, and collect statistically relevant data.

---

## Findings from Grover on Rigetti (to reuse)

- **Optimization level 3** (Qiskit transpiler before submission) is used; it improved 2-qubit results.
- **2 qubits**: Strong success (~90–96% DSR); multiple runs per config for statistics.
- **3 qubits**: Marginal or below random for many configs; still worth characterizing.
- **Depth**: Useful signal was in shallow circuits (transpiled depth &lt; 150 for Grover); scale up only if smaller sizes look viable.
- **Statistics**: At least **3 runs per config** for mean/std; **5 runs** preferred for confidence intervals.

---

## QFT configs (existing)

From `qft_configs.py`:

- **Roundtrip (SR)**: SR2 (2q), SR3 (3q), SR4 (4q), SR5–SR14. Single expected outcome; simplest to interpret.
- **Period detection (SP/PV)**: Multiple peaks; 3q–12q, various periods. More complex success criteria.
- **Input variation (IV)**: 4q with different input states.

IBM Region 1 depths (transpiled, opt 2): SR2 ~10, SR3 ~14, SR4 ~18, SR5 ~22, SR6 ~26, SR7 ~30; period configs ~27–39 for 4–6q.

---

## Phased execution plan

### Phase 1: 2 qubits only (baseline)

- **Config**: **SR2** (roundtrip, 2q, input |01⟩).
- **Runs**: **5** (for mean, std, and basic confidence).
- **Goal**: Confirm QFT roundtrip works on Rigetti at 2q and establish baseline DSR/success rate.
- **Success criterion**: Mean success rate &gt; 70% and DSR (Michelson) clearly above random (e.g. &gt; 0.5).

**Commands (example):**

```bash
# 5× SR2
for i in 1 2 3 4 5; do
  uv run python qward/examples/papers/qft/qft_aws.py --config SR2
done
```

---

### Phase 2: 3 qubits

- **Config**: **SR3** (roundtrip, 3q, input |101⟩).
- **Runs**: **3** (minimum for a rough mean/std).
- **Goal**: See if 3q QFT roundtrip is viable on Rigetti (Grover 3q was marginal).
- **Success criterion**: If mean success &gt; 50%, proceed to 4q; else document and optionally try one period config at 3q (SP3-P2).

---

### Phase 3: 4 qubits (roundtrip + one period)

- **Configs**: **SR4** (roundtrip 4q), **SP4-P2** (period=2, 4q, fewer peaks).
- **Runs**: **3 per config**.
- **Goal**: Characterize 4q; compare roundtrip vs period detection.
- **Success criterion**: If SR4 or SP4-P2 shows mean success &gt; 40%, consider adding more 4q configs (e.g. PV4-P2, IV4-0000) or a single 5q probe.

---

### Phase 4: 5–6 qubits (only if Phase 3 is promising)

- **Configs**: **SR5**, optionally **SP5-P4** or **PV6-P2**.
- **Runs**: **2 per config** (exploratory).
- **Goal**: Find practical qubit limit for QFT on Rigetti; avoid deep circuits (e.g. transpiled depth &gt; 300) until 2–4q are understood.

---

## Config set for automation (Rigetti small-scale)

Use these for a “characterization” batch (e.g. `--characterize` or `--batch --batch-configs ...`):

| Phase | Config IDs | Qubits | Runs (suggested) |
|-------|------------|--------|------------------|
| 1     | SR2        | 2      | 5                |
| 2     | SR3        | 3      | 3                |
| 3     | SR4, SP4-P2| 4      | 3 each           |
| 4     | SR5        | 5      | 2 (optional)    |

**Order**: Run in increasing qubit count so early phases inform whether to continue.

---

## Technical setup (already or to add)

1. **Optimization level 3** for QFT on AWS (same as Grover): pass `optimization_level=3` in `run_aws()` (default in experiment `run()` for Rigetti).
2. **Output**: JSON per run under `qft/data/qpu/aws/`; then regenerate `DSR_result_aws.csv` (include QFT) and plots with `--max-qubits` / `--max-depth` as for Grover.
3. **DSR**: QFT uses same DSR pipeline; expected outcomes are either a single state (roundtrip) or a list of peak states (period detection).

---

## Summary

- **Start**: SR2 × 5 runs.
- **Then**: SR3 × 3 runs.
- **Then**: SR4 × 3, SP4-P2 × 3.
- **Optional**: SR5 × 2 (and optionally one more 5q/6q config) if 4q looks viable.
- **Always**: Use optimization_level=3; keep depth in mind (useful range &lt; ~150 for Grover; QFT may differ but start shallow).
- **Statistics**: Report mean ± std success rate and DSR (Michelson) per config; 3–5 runs per config for relevance.
