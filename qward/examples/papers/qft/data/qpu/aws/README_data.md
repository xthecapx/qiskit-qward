# QFT AWS (Rigetti Ankaa-3) data — 8 runs per config

## Target

- **8 runs each** for: SR2, SR3, SR4, SP4-P2, SR5 (40 jobs total).
- Results: one JSON per job in this directory.

## Regenerate DSR CSV for analysis

From repo root, include QFT AWS JSONs in the AWS DSR CSV:

```bash
cd /Users/cristianmarquezbarrios/Documents/code/qiskit-qward
PYTHONPATH=. uv run python qward/examples/papers/differential_success_rate_experiment.py \
  --provider aws \
  --qft-dir qward/examples/papers/qft/data/qpu/aws \
  --output qward/examples/papers/DSR_result_aws.csv
```

Then run the DSR analysis (e.g. with filters for QFT / qubits / depth):

```bash
PYTHONPATH=. uv run python qward/examples/papers/differential_success_rate_analysis.py \
  --input qward/examples/papers/DSR_result_aws.csv \
  --out-dir qward/examples/papers/plots/aws_rigetti
```

## Per-config counts (check after batch completes)

- **SR2**: 8 runs from batch (+ 2 earlier = 10 total)
- **SR3**: 8 runs from batch (+ 1 earlier = 9 total)
- **SR4**: 8 runs from batch (+ 1 earlier = 9 total)
- **SP4-P2**: 8 runs
- **SR5**: 8 runs

For “8 of each” from this batch only, count files by config prefix (e.g. `SR2_AWS_*`, `SR3_AWS_*`, …) and re-run missing configs if needed.

## Quick count (shell)

```bash
for c in SR2 SR3 SR4 "SP4-P2" SR5; do echo -n "$c: "; ls -1 "${c}"_AWS_*.json 2>/dev/null | wc -l; done
```
