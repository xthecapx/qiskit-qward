# Real experiment on IonQ Forte — budget and options

This document defines **experiment options** (per algorithm) aligned with what was run on **Rigetti Ankaa-3** and **IBM**, and gives **cost estimates** for running an equivalent “real experiment” on **IonQ Forte** via AWS Braket. Use it to report to your professor the cost per algorithm and to choose a scope (minimal / standard / full).

---

## 1. Provider pricing summary

### 1.1 AWS Braket (Rigetti and IonQ)

| Provider | QPU        | Per task | Per shot   | 128 shots/job | 256 shots/job | 1024 shots/job |
|----------|------------|----------|------------|---------------|---------------|----------------|
| **IonQ** | **Forte**  | $0.30    | **$0.08**  | **$10.54**    | **$20.78**    | **$82.22**     |
| Rigetti  | Ankaa      | $0.30    | $0.00090   | $0.42         | $0.53         | $1.22          |

- **IonQ Forte:** `arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1` — use `device_id="Forte-1"`, `region="us-east-1"`.
- **Rigetti Ankaa:** default in our scripts (`Ankaa-3`, `us-west-1`).

IonQ is ~67× more expensive per shot than Rigetti. You can **choose 128, 256, or 1024 shots per job**; section 4 gives full budgets at all three.

### 1.2 IBM Quantum

| Model        | Pricing |
|-------------|---------|
| **Pay-as-you-go** | **$96 USD per minute** of quantum execution (billed per second) |
| **Open Plan**     | Free, up to **10 minutes per month** (28-day window) |
| **Flex**          | $72/min (min. 400 min/year) |
| **Premium**       | $48/min (min. 5200 min/year) |

Cost per “experiment” on IBM depends on **total runtime** (sum of all job runtimes), not on shot count in the same way as Braket. For comparison with Braket we assume typical short jobs (e.g. 1–2 min per job) so a few dozen jobs can stay within free-tier or low pay-as-you-go cost.

---

## 2. What was run on Rigetti (reference scope)

- **Grover:** 13 configs (4× 2q: S2-00, S2-1, S2-10, S2-11; 9× 3q: S3-1, M3-1, M3-2, H3-0, H3-2, H3-3, SYM-1, ASYM-1, ASYM-2). Multiple runs per config (e.g. 8 runs each for statistical relevance).
- **QFT:** 4 configs (SR2, SR3, SR4, SP4-P2), 8 runs each; plus SR5 in some runs.
- **Teleportation:** Separate pipeline (CSV); included in DSR analysis but not submitted via the same AWS scripts.

---

## 3. Experiment options per algorithm (IonQ Forte)

For each algorithm we define **option A (minimal)**, **option B (standard)**, and **option C (full, Rigetti-equivalent)**. You can run with **128, 256, or 1024 shots per job**; **section 4** gives the full budget for all three shot counts.

---

### 3.1 Grover

| Option | Configs | Runs per config | Total jobs | Est. cost (IonQ, 256 shots) |
|--------|---------|------------------|------------|-----------------------------|
| **A — Minimal**   | 4 (2q only: S2-00, S2-1, S2-10, S2-11) | 1 | 4  | **~$83**  |
| **B — Standard** | 13 (4× 2q + 9× 3q, full characterization) | 2 | 26 | **~$540** |
| **C — Full**     | 13 (same as Rigetti) | 5 | 65 | **~$1,351** |

- **Recommendation for professor:** Start with **Option A** to validate and report 2q Grover on IonQ (~\$83). If budget allows, **Option B** gives a direct comparison to Rigetti characterization at ~\$540.

**Commands (Option A, 4 jobs):**

```bash
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py --config S2-00 --device "Forte-1" --region us-east-1 --shots 256
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py --config S2-1  --device "Forte-1" --region us-east-1 --shots 256
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py --config S2-10 --device "Forte-1" --region us-east-1 --shots 256
PYTHONPATH=. uv run python qward/examples/papers/grover/grover_aws.py --config S2-11 --device "Forte-1" --region us-east-1 --shots 256
```

---

### 3.2 QFT

| Option | Configs | Runs per config | Total jobs | Est. cost (IonQ, 256 shots) |
|--------|---------|------------------|------------|-----------------------------|
| **A — Minimal**   | 2 (SR2, SR3) | 2 | 4  | **~$83**  |
| **B — Standard** | 4 (SR2, SR3, SR4, SP4-P2) | 3 | 12 | **~$249** |
| **C — Full**     | 5 (add SR5) | 5 | 25 | **~$520** |

- **Recommendation:** **Option A** for a quick 2q/3q QFT snapshot (~\$83); **Option B** to mirror the Rigetti QFT batch (~\$249).

**Commands (Option A, 4 jobs):**

```bash
for c in SR2 SR3; do
  for i in 1 2; do
    PYTHONPATH=. uv run python qward/examples/papers/qft/qft_aws.py --config "$c" --device "Forte-1" --region us-east-1 --shots 256
  done
done
```

(QFT AWS script uses same `--device` / `--region` / `--shots` from the base CLI.)

---

### 3.3 Teleportation

Teleportation in this project is analyzed from **CSV** (separate from the Braket Grover/QFT scripts). There is no “run on IonQ” script in the repo for Teleportation. If you add one later:

- Count **tasks × shots** and use the same IonQ formula: **$0.30 + 256 × $0.08 ≈ $20.78 per job** (or 128 shots ≈ $10.54).
- For now, **Teleportation on IonQ:** not in scope; budget **$0** unless you define a new experiment.

---

## 4. Budget by shot count (128 / 256 / 1024) — for professor

All figures below are **IonQ Forte**. Per-job cost: **128 shots = $10.54**, **256 shots = $20.78**, **1024 shots = $82.22**.

### 4.1 Grover (IonQ Forte)

| Option | Jobs | 128 shots total | 256 shots total | 1024 shots total |
|--------|------|-----------------|-----------------|------------------|
| **A — Minimal**   | 4  | **~$42**  | **~$83**  | **~$329**  |
| **B — Standard** | 26 | **~$274** | **~$540** | **~$2,138** |
| **C — Full**     | 65 | **~$685** | **~$1,351** | **~$5,344** |

### 4.2 QFT (IonQ Forte)

| Option | Jobs | 128 shots total | 256 shots total | 1024 shots total |
|--------|------|-----------------|-----------------|------------------|
| **A — Minimal**   | 4  | **~$42**  | **~$83**  | **~$329**  |
| **B — Standard** | 12 | **~$126** | **~$249** | **~$987**  |
| **C — Full**     | 25 | **~$264** | **~$520** | **~$2,056** |

### 4.3 Combined (Grover + QFT)

| Option | 128 shots total | 256 shots total | 1024 shots total |
|--------|-----------------|-----------------|------------------|
| **A — Minimal**   | **~$84**  | **~$166** | **~$658**  |
| **B — Standard** | **~$400** | **~$789** | **~$3,125** |
| **C — Full**     | **~$949** | **~$1,871** | **~$7,400** |

Use **128 shots** for the lowest budget (e.g. **~\$84** for both algorithms, minimal scope); **256 shots** for a balance of cost and statistics; **1024 shots** only if you need full Rigetti-equivalent statistics and have the budget.

- **Rigetti (reference):** same option-C scope at 1024 shots/job: Grover 65×\$1.22 ≈ \$79, QFT 25×\$1.22 ≈ \$31 → **~\$110** total.
- **IBM:** runtime-based; similar number of jobs can often fit in **free tier (10 min/month)** or small pay-as-you-go usage (**$96/min**).

---

## 5. Suggested wording for professor

You can summarize as:

- **“Real experiment” on IonQ Forte** means running the same algorithms and configs we use on Rigetti (Grover characterization, QFT roundtrip/period).
- **We can choose 128, 256, or 1024 shots per job**; cost scales with shots (128 ≈ \$10.54/job, 256 ≈ \$20.78/job, 1024 ≈ \$82.22/job on IonQ).
- **Budget options (Grover + QFT combined):**
  - **Minimal (A):** 8 jobs total → **~\$84** (128 shots), **~\$166** (256), **~\$658** (1024).
  - **Standard (B):** 38 jobs total → **~\$400** (128), **~\$789** (256), **~\$3,125** (1024).
  - **Full (C):** 90 jobs total → **~\$949** (128), **~\$1,871** (256), **~\$7,400** (1024).
- **Recommendation:** 128 or 256 shots keeps cost manageable (e.g. **~\$84–\$166** for minimal, **~\$400–\$789** for standard). 1024 shots matches Rigetti statistics but is much more expensive on IonQ.
- **Comparison:** The same full scope on Rigetti (Ankaa) at 1024 shots is **~\$110** total; on IBM it is runtime-based (free tier or **$96/min** pay-as-you-go).

---

## 6. References in repo

- **Grover IonQ (cost-conscious):** `grover/plan_aws_ionq.md`
- **Grover Rigetti characterization:** `grover/grover_aws.py` — `RIGETTI_CHARACTERIZATION_CONFIGS`, `--characterize`
- **QFT Rigetti plan:** `qft/plan_aws_rigetti.md`
- **QFT configs:** `qft/qft_aws.py` — `RIGETTI_QFT_CONFIGS`, `--characterize`
- **Device/region for IonQ:** `grover/grover_aws.py` — `IONQ_FORTE_DEVICE`, `IONQ_FORTE_REGION` (same for QFT: `--device "Forte-1"` `--region us-east-1`).
