# Grover on AWS Braket — IonQ Forte execution plan

Execute Grover on **IonQ Forte** via AWS Braket with a cost-conscious, phased plan.

**For a full “real experiment” budget (Grover + QFT + IBM comparison):** see [plan_ionq_real_experiment.md](../plan_ionq_real_experiment.md).

---

## Device and region

- **Device:** IonQ Forte (`Forte-1`)
- **ARN:** `arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1`
- **Region:** `us-east-1` (IonQ is only in us-east-1)
- **Backend name for code:** `device_id="Forte-1"`. Region is not passed; the script sets us-east-1 when the device is Forte-1.

Region is inferred from the device; you do not pass `--region`. For IonQ the script sets us-east-1 automatically. (You do not need to pass `--device "Forte-1"` and no `--region`, you get “No backend matches the criteria”. The Grover script now **auto-sets region to us-east-1** when the device is Forte-1, so you can omit `--region us-east-1`; passing it explicitly is still fine.

---

## Pricing (AWS Braket, per task + per shot)

| Provider | QPU           | Per task | Per shot  | 128 shots/job | 256 shots/job | 1024 shots/job |
|----------|---------------|----------|-----------|---------------|---------------|----------------|
| IonQ     | **Forte**     | $0.30    | **$0.08** | **$10.54**    | **$20.78**    | **$82.22**     |
| Rigetti  | Ankaa         | $0.30    | $0.00090  | $0.42         | $0.53         | $1.22          |

IonQ Forte is **~67× more expensive per shot** than Rigetti Ankaa. Use **128 or 256 shots** to control cost; see budget table below for totals.

---

## Recommended shot count for IonQ

- **Validation / first run:** **128 shots** (~\$10.55 per job)
- **Minimal statistics:** **256 shots** (~\$20.78 per job)
- **Avoid** 1024 shots per task on Forte unless necessary (~\$82 per job)

Use a dedicated experiment instance with lower shots, e.g.  
`GroverAWSExperiment(shots=256)` for IonQ runs.

---

## Phased execution plan

### Phase 0 — Validate pipeline (one job)

- **Config:** S2-1 (2 qubits, single marked state).
- **Shots:** 128.
- **Goal:** Confirm submission, run, and result parsing on IonQ Forte.
- **Cost:** ~\$10.55.

**Command (from repo root):**

```bash
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py \
  --config S2-1 \
  --device "Forte-1" \
  --shots 128
```
(Region is set from the device; no need to pass `--region`.)

---

### Phase 1 — Minimal Grover (2 qubits only)

- **Configs:** S2-1 only (simplest).
- **Runs:** 2–3 jobs.
- **Shots:** 256 per job.
- **Goal:** Get a small but meaningful DSR/success-rate snapshot on IonQ.
- **Cost:** 2 × \$20.78 ≈ **\$42** or 3 × \$20.78 ≈ **\$62**.

**Command (example, 2 runs):**

```bash
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py \
  --config S2-1 --device "Forte-1" --shots 256
# Run again for a second data point (or loop)
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py \
  --config S2-1 --device "Forte-1" --shots 256
```

---

### Phase 2 (optional) — One 3-qubit run

- **Config:** e.g. S3-1 (3 qubits, single marked).
- **Runs:** 1 job.
- **Shots:** 256.
- **Goal:** Check if 3q Grover is viable on Forte before scaling.
- **Cost:** ~\$21.

Only run after Phase 1 looks good.

---

## Budget summary (example)

| Phase   | Jobs      | Shots/job | 128 shots | 256 shots | 1024 shots |
|---------|-----------|-----------|-----------|-----------|------------|
| Phase 0 | 1 (S2-1)  | —         | ~\$11     | ~\$21     | ~\$82      |
| Phase 1 | 2–3 (S2-1)| —         | ~\$21–32  | ~\$42–62  | ~\$164–247 |
| Phase 2 | 1 (S3-1)  | —         | ~\$11     | ~\$21     | ~\$82      |
| **Total (4–5 jobs)** | — | **~\$42–54** | **~\$84–104** | **~\$328–411** |

To stay under ~\$50: do Phase 0 + 2 runs of S2-1 at **128 shots** (~\$11 + 2×\$10.54 ≈ \$32). For full experiment budgets (Grover + QFT at 128/256/1024), see [plan_ionq_real_experiment.md](../plan_ionq_real_experiment.md).

---

## Code / CLI

- **Device:** Use `--device "Forte-1"` for IonQ; region is set automatically (us-east-1), so you do not pass `--region`.
- **Shots:** The base CLI supports `--shots`; use `--shots 128` or `--shots 256` for IonQ to control cost.
- **Output:** Results are saved under `grover/data/qpu/aws/` with device name in the filename (e.g. `S2-1_AWS_Forte-1_*.json`), so DSR/analysis can include or filter by device.
- **Optimization level:** Same as Rigetti (level 3) is used by default in the Grover AWS experiment.

---

## Summary

- **Start:** 1× S2-1 on IonQ Forte with **128 shots** (~\$10.55) to validate.
- **Then:** 2–3× S2-1 with **256 shots** (~\$42–62) for minimal Grover-on-IonQ data.
- **Optionally:** 1× S3-1 with 256 shots (~\$21) to probe 3 qubits.
- **Always:** `device_id="Forte-1"`, `region="us-east-1"`, and **reduced shots** to control cost.
