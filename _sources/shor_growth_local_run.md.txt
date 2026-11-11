> Historical research note: current collection scripts, datasets, and dashboard hosting are maintained in [estimating-quantum-circuit](https://github.com/xthecapx/estimating-quantum-circuit). Commands below describe the pre-migration QWARD layout. Retained with the report pending its later migration.

# Run the Shor growth campaign locally

Branch: `codex/shor-growth-2026-11-05`, created from `main`.
Tonight's independent campaign is `fez-shor-growth-v1-2026-11-05`:
13 configurations, three independent jobs each, 1,024 shots/job (39 jobs).
The three families keep N=15 and use a=4/M2–M6, a=2/M3–M6, and a=7/M3–M6.
Total logical width spans 6–10 qubits. Historical a=7 configuration IDs remain valid.
The preset uses optimization level 3, seed 11, gate twirling and XpXm dynamical
decoupling, matching the previous Shor settings.

## Prepare and run tonight

From the repository root, activate the existing environment. Keep the same IBM
credentials used for the completed campaign; do not commit credentials.
All commands below use the profile explicitly to avoid selecting the 344-job
legacy profile. Run only one submission process for this collection at a time.
`CAMPAIGN_CHECKPOINT=1` is for Actions Git pushes; leave it unset locally.

```sh
source .venv/bin/activate
unset CAMPAIGN_CHECKPOINT
export CAMPAIGN_PROGRESS=1

# Offline: validate the selected workload (no IBM calls).
python scripts/ibm_campaign_ci.py validate --date 2026-11-05 --profile shor-growth --backend ibm_fez

# Refresh live calibration, transpile, compute/reuse logical metrics, save circuits.
# Creates five batches containing 9, 9, 9, 9, and 3 jobs. Submits no jobs.
python scripts/ibm_campaign_ci.py prepare --date 2026-11-05 --profile shor-growth --backend ibm_fez

# Submit the 39 planned jobs, at most two active batches, and monitor until terminal.
python scripts/ibm_campaign_ci.py submit --date 2026-11-05 --profile shor-growth --backend ibm_fez --max-batches 2 --batch-max-time 3600

# Retrieve results; retries reads only. Reports success only when all 39 jobs complete.
python scripts/ibm_campaign_ci.py recover --date 2026-11-05 --profile shor-growth --backend ibm_fez

# Refresh the complete multi-file dataset, including prior campaigns.
python -m qward.examples.papers.dataset_export --out qward/examples/papers/dataset_out
python dashboard/build.py
```

The committed plan is
`qward/examples/papers/shor/plans/shor-fez-shor-growth-d2026-11-05.json`.
Its parent manifest adds `_manifest.json`; individual batch files add
`_manifest_part01.json` through `_manifest_part05.json`.
Use the CI script above to drive these partitions, not `--submit-plan` on the parent.

If preparation is interrupted, rerun `prepare`: complete manifests and their frozen
circuits are retained. If submission stops, rerun `submit` to resume only untouched
batches. Already submitted batches keep their job IDs. Ambiguous partial batches
stop submission and need inspection. `recover` never creates replacement jobs;
failed or pending IBM jobs leave the campaign incomplete. Do not delete or reset
journals to bypass those checks.

No QPU preparation or submission is performed by the implementation's offline
validation. The first live operation is the `prepare` command you run above.
Save the code/plan now, then save records, prepared artifacts, manifests, and
updated export tables after recovery. Merge the branch later as usual.

## Tomorrow's third collection

The Actions workflow now defaults to profile `full-shor-growth`: 184 configurations,
381 jobs (342 existing non-Shor jobs plus 39 Shor jobs). It uses a distinct profile
identity and retains the `full` legacy profile for previous collections.

To use it on November 6, merge these changes into `main` before 11:45 a.m. Bogotá
and set repository variables:

- `IBM_CAMPAIGN_DATE=2026-11-06`
- `IBM_CAMPAIGN_PROFILE=full-shor-growth`
- `IBM_CAMPAIGN_BACKEND=ibm_fez`

A manually dispatched run also offers the profile selector. For recovery, choose
exactly the original profile/backend/date. Existing November 5 full runs use `full`.
The same date can have different profiles, so do not dispatch both `full` and
`full-shor-growth` unless you deliberately want both workloads.
If this branch is merged after the schedule, manually dispatch the expanded profile.

Dashboard family grouping distinguishes N/base/strategy. Its qubit axis continues
to show total logical qubits, and its depth axis shows executed ISA depth. Existing
metric definitions are preserved. The [evaluation](shor_growth_campaign_evaluation.md)
explains the ideal 50% task-success ceiling, changing noise baseline, and why
success-labelled DSR must not be used alone for Shor decay regression.
