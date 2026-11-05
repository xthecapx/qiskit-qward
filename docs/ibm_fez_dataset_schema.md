# ibm_fez dataset: columns and plan files

`dataset-*.csv` contains one row per completed QPU job, split into files of at most 20 MB. `dataset_manifest.json` lists the ordered parts and their row counts/byte sizes. Each part repeats the same header; concatenate them for analysis. They are produced by

```bash
qward/examples/papers/run_campaign.sh export OUT_DIR
# or: uv run -m qward.examples.papers.dataset_export --backend ibm_fez --out OUT_DIR
```

which also writes `schema.json` (every column with its group, dtype and
non-null count) and `export_report.json` (skipped records, errors, warnings).
Only `<algorithm>/data/qpu/v2/` records are read; legacy files are never touched.

## Column groups

| Prefix | Group | Source in the record |
|---|---|---|
| `meta.*` | identity: algorithm, config, job/run/batch/campaign IDs, window, backend, shots, optimization level | journal |
| `config.*` | scalar fields of the config description (qubits, secret, period, ...) | `prepared.json` → `config` |
| `a.*` | logical circuit: counts, depth, QWARD metrics, QuantumSpecific | `payload.features.logical` |
| `b.*` | executed (ISA) circuit: active qubits, 2Q gates/depth, `count_ops.*`, QWARD metrics | `payload.features.isa` |
| `c.*` | hardware at submission: gate/readout/T1/T2/idle statistics for the qubits used | `payload.features.hardware_at_submission` |
| `d.*` | hardware at execution (`job.properties()`), `d.changed_since_prepare` | `submission_journal[i].hardware_at_execution` |
| `e.*` | baseline predictions at submission: `m1_error`, `m1_mixture_error`, `m2_error`, `*_tvd`, ... | `payload.features.baselines` |
| `e_exec.*` | the same re-predicted from the execution-time calibration | `baselines_at_execution` |
| `f.*` | targets: `f.fidelity.*` (success, DSR, Hellinger, TVD), `f.histogram_assessment.*`, `f.counts_json`, `f.observed_error`, `f.residual_vs_*` | `individual_results[i].targets` |
| `g.*` | provenance: seed, scheduling, Runtime options, timestamps, queue/run seconds, package versions | record |

Nested keys are joined with dots; lists are stored as JSON text. Each group has
`<prefix>.available`; when false, `<prefix>.reason` says why and the other
columns of that group are empty. Empty means *not computed*, never zero.

Documented reasons:
- `dynamic_circuit` (`e.*`, `e_exec.*`, `a.quantum_specific.*`): mid-circuit
  control flow (vTP S configs).
- `size_limit>8_qubits` (`a.quantum_specific.*`): QuantumSpecific is skipped
  above 8 qubits.
- `no_reference` (`e.*`, `f.*`) means the runner has no tractable ideal
  distribution. GHZ and standard BV supply their analytic references and
  complete success sets above the generic 12-bit enumeration limit.

Gate counts `b.count_ops.<gate>` are filled with 0 when a gate is absent from
the executed circuit.

## Targets and predictions

- `f.fidelity.success_rate` uses the runner's success set; Hellinger and TVD
  use the exact ideal distribution (`g.reference_exact` says whether it was
  sampled instead, as for dynamic circuits).
- `f.observed_error` = 1 − success rate. Compare with `e.m1_error`,
  `e.m1_mixture_error`, `e.m2_error` (predicted 1 − P(success set)), and
  `f.fidelity.tvd` with `e.m2_tvd` / `e.m1_mixture_tvd` / `e.m1_tvd`.
- `f.residual_vs_<model>` = observed error − predicted error.
- For GHZ above 12 measured bits, the histogram TVD interval uses simultaneous
  binomial intervals on the two ideal labels. The interval method is recorded
  in `f.histogram_assessment.interval_method`. BV retains the single-label
  Clopper–Pearson interval at every width.

## Validation (the export fails unless `--allow-incomplete`)

Errors: no rows, duplicate job IDs, mixed or unexpected record schema,
simulated backends (unless `--allow-fake`), an unavailable group a/b/c/d/f
(or e without a documented reason), counts not summing to shots, calibration
coverage below 1. Warnings (in the report only): QuantumSpecific unavailable
for an undocumented reason, calibration changed between prepare and
execution, sampled references.

## Plan files

One plan per algorithm per window, `qward/examples/papers/<algo>/plans/<algo>-w<N>.json`:

```json
{
  "backend": "ibm_fez",
  "campaign_id": "fez-dataset-v2",
  "window": 1,
  "shots": 1024,
  "optimization_level": 3,
  "seed_transpiler": 11,
  "order_seed": 7,
  "runs": 2,
  "planned_executions": 80,
  "items": [{"config": "GHZ4"}, {"config": "GHZ8"}, {"config": "GHZ12", "runs": 2}]
}
```

`runs` is per config unless an item overrides it; `planned_executions` is the
window's run budget (defaults to the total). A plan is submitted as one Batch;
jobs are ordered by a seeded shuffle of configs, run 1 of every config before
run 2.
