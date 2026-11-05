# Automated IBM campaigns

`ibm-campaign.yml` collects the reviewed full research profile on one selected IBM
chip: **172 configurations, 344 jobs, 1,024 shots per job**. It saves checkpoints
and results to a branch. **It does not create, update, or merge a pull request.**
Review the branch and merge the results manually to publish the dashboard.
The reduced monthly dashboard profile is a separate follow-up; this workflow
must not be scheduled monthly unchanged.

## November 6 test and setup

Merge the workflow and orchestration changes into `main` before the scheduled
slot. The date guard defaults to **November 6, 2026**, at **11:45 a.m. Bogotá**
(`45 16 * * *`, 16:45 UTC). GitHub may delay scheduled execution; this is the
requested trigger time, not a guaranteed QPU start time.

Under **Settings → Secrets and variables → Actions** configure:

| Type | Name | Value |
| --- | --- | --- |
| Secret | `IBM_QUANTUM_TOKEN` | Your IBM token |
| Secret | `IBM_QUANTUM_INSTANCE` | The instance/CRN used locally |
| Variable | `IBM_QUANTUM_CHANNEL` | `ibm_quantum_platform` |
| Variable | `IBM_CAMPAIGN_DATE` | `2026-11-06` for tomorrow's test |
| Variable | `IBM_CAMPAIGN_BACKEND` | `ibm_fez` (default), `ibm_kingston`, or `ibm_marrakesh` |

**Update an existing `IBM_CAMPAIGN_DATE=2026-11-05` variable**: repository variables
override the new default. Scheduled runs on other dates exit without IBM calls.
Do not put IBM credentials in plain variables or commit `.env`. The workflow
needs `contents: write`; permission to create PRs is no longer required.

In **Actions → Collect IBM campaign results → Run workflow**, select `main`,
the backend and collection date, and one of these modes:

| Mode | Behavior |
| --- | --- |
| `validate` | Checks the full profile and limits offline. No IBM calls or branch writes. |
| `run` | Claims a new backend/date, prepares circuits, submits jobs, recovers and exports. Refuses an existing claim. |
| `resume` | Checks out the existing results branch, preserves prepared circuits and existing jobs, prepares missing algorithms, and submits only wholly untouched batches. |
| `recover` | Checks out the existing branch and retrieves saved job IDs only. Does not prepare or submit. |

The token/instance must have access to the selected chip. Offline validation does
not verify access or backend availability. Each selected chip runs the same full
profile; selecting Kingston or Marrakesh does not run all three chips.

## Separate stages and smaller batches

The Actions page exposes independent **Prepare**, **Submit**, **Recover**, and
**Export** steps, with separate logs and timeouts:

1. **Prepare** (120 minutes): refresh calibration, transpile all circuits, reuse
   compatible logical metrics, and partition fresh manifests. Push a checkpoint
   after each algorithm. Existing prepared/submitted manifests are preserved.
2. **Submit** (150 minutes): submit untouched batches and poll IBM until slots
   become available. The default is three concurrent batches with **at most ten
   jobs per batch**, keeping both repetitions of a configuration together.
   This produces **37 batches overall, including five for PE**. Submission does
   not download results. Push a checkpoint after each submitted batch.
3. **Recover** (45 minutes): retrieve results across every submitted batch, with
   up to three retrieval passes and five seconds between incomplete passes.
   Completed records are skipped; pending jobs are left for a later recovery.
   A failed batch does not prevent collecting the remaining batches. Recovery
   also runs after a submission-step failure, when dependencies are available.
4. **Export**: only after all 344 unique planned jobs are complete, refresh the
   full dataset across periods and chips and validate the static dashboard.

You can select **6, 10, or 14 jobs per batch**, **1–3 concurrent batches**, and a
requested batch lifetime of **600, 1800, or 3600 seconds**. The default lifetime
is **3600 seconds**. IBM enforces the instance's limits; a rejected batch opening
sends no jobs. Smaller batches reduce exposure to lifetime expiry but do not
promise completion under every queue or backend condition. Batch limits are
fixed when preparation creates the partition; resume preserves that partition
rather than changing an existing experiment's job allocation.

The November 5 original PE batch had a 600-second lifetime and closed ten minutes
after starting, with only 46 seconds of billed QPU usage. Batch lifetime and billed
QPU time are different. IBM documents that queued jobs fail when the lifetime is
reached, while running jobs finish. See [IBM batch lifetime documentation](https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-batch).

Child manifests (`*_manifest_part01.json`, etc.) reference the original prepared
records; no extra configurations or repetitions are added. Parent manifests keep
all job IDs and summarize child completion and batch IDs. Within each partition,
submission preserves the reviewed seeded order. The two repetitions now occur
within smaller batches, so their timing differs from earlier one-batch-per-algorithm
collections; use recorded timestamps when comparing variation across periods.
For a partitioned collection, use the CI entry point to submit/recover; the legacy
single-manifest CLI is not a driver for parent manifests with `parts`.

## Retry and credit protection

- One account-wide GitHub concurrency group, with cancellation disabled, covers
  all chips. Do not run the same backend/date locally while Actions owns it.
- A permanent marker and result branch claim the backend/date **before** QPU work.
  A fresh `run` refuses an existing claim. Keep markers after merging results.
- Resume counts already submitted batches before opening new ones. Completed,
  failed, and cancelled jobs keep their original IDs. **Failed jobs are never
  replaced automatically.** Retrying retrieval does not spend another job's credits.
- A partially submitted batch, a missing ID, or an ambiguous submission blocks
  new submissions. Inspect IBM and the journal before reconciling it manually.
  Resume only sends a whole batch with no previous submission intent or IDs;
  it does not fill gaps in a partially submitted batch.
- Recovery reads all known submissions even after submission fails. Remaining
  missing, failed, pending, or unsubmitted jobs keep the collection incomplete;
  it cannot be exported as a completed campaign.
- A checkpoint push failure stops further submissions. The final cleanup step
  attempts to save remaining records to the branch even after failure. A timeout
  does not cancel IBM jobs. A forcibly terminated runner may lose writes since
  its last durable checkpoint; inspect IBM before resolving ambiguous state.
- Plans, parent and child manifests, records, circuit artifacts, logical cache,
  and export tables are uploaded as an artifact with 90-day retention, including
  interrupted runs. Branch and artifact backups do not depend on PR permissions.

The workflow summary links the result branch and reports completion or the need
for resume/recovery. Use the **same backend and original collection date** for
both, even when recovering on a later day. Do not delete a claim to bypass the
submission guards. Review the updated export report before merging results into
`main`; the normal docs/dashboard workflow publishes the static site afterward.

## Cache and environment

Compatible logical-metric values from prior campaigns are reused. Plain and
`+cpu` PyTorch wheels of the same release share a key because these metrics run
on CPU. Different releases, other build suffixes, circuit hashes, seeds, or metric
source invalidate the key. Legacy cloud keys remain readable. This does not claim
that recomputing on different machines is bit-for-bit identical. Transpilation and
calibration are always refreshed for a new backend/date.

Actions restores/saves the logical cache independently of branch merging, including
failed collections. Committed entries remain the fallback; new entries also go to
checkpoints and artifacts. Python 3.11 and CPU PyTorch 2.8.0 remain in use, with pip
download caching. Qiskit 2.6.2, Aer 0.18.2, and Runtime 0.51.0 are pinned to match the
local QPY-compatible environment. Installation does not yet consume `uv.lock`.

`CAMPAIGN_PROGRESS=1` enables UTC timestamps, per-configuration progress, cache
hits/misses, job IDs, and 30-second heartbeats for long operations. A heartbeat
shows that logging is alive, not that the remote operation is advancing.
`CAMPAIGN_CHECKPOINT=1` enables Git checkpoints and is set by Actions; leave it
unset for local inspection unless you deliberately want Git pushes.

## Identities, billing, and local commands

Fez uses `codex/ibm-fez-YYYY-MM-DD`, `<algorithm>-dYYYY-MM-DD.json`, and campaign ID
`fez-dataset-v2-YYYY-MM-DD`. Other chips include their name in the plan/marker paths
as well: for example `pe-kingston-d2026-11-06.json`. The legacy numeric `window`
field remains `YYYYMMDD`. Old numbered plans remain untouched; the reviewed
window-2 configuration set is the full-profile template.

Offline validation and targeted tests:

```sh
python scripts/ibm_campaign_ci.py validate --date 2026-11-06 --backend ibm_fez
python -m pytest tests/test_ibm_campaign_ci.py tests/test_campaign_submit.py tests/test_ibm_campaign.py -q
```

Local equivalents for a deliberately managed new collection (these contact IBM):

```sh
python scripts/ibm_campaign_ci.py prepare --date 2026-11-06 --backend ibm_fez --max-jobs-per-batch 10
python scripts/ibm_campaign_ci.py submit --date 2026-11-06 --backend ibm_fez --max-batches 3 --batch-max-time 3600
python scripts/ibm_campaign_ci.py recover --date 2026-11-06 --backend ibm_fez
```

For billing, pass **child manifests** for split algorithms and the **parent** for
an unsplit algorithm (Shor by default) to the existing `--plan-usage` command.
A split parent has `batch_ids` rather than one billable `batch_id`; do not count
both parent and children. Saved per-job usage remains available in raw journals.

A deliberate replacement of failed jobs requires manual review and extra credits.
Preserve the old attempt outside `data/qpu/v2` before creating a replacement. The
November 5 original PE attempt is retained at
`phase_estimation/archive/2026-11-05-pe-attempt-1/`, including 23 successes and 19
failures; the active dataset contains its full 42-job replacement. The algorithm
CLI supports `--submit-plan MANIFEST --batch-max-time 3600` for an explicitly
prepared unsplit replacement.

Change `IBM_CAMPAIGN_DATE` for the next date and merge each completed collection
before starting another so exports include the latest results. The date does not
automatically advance. A future monthly profile should select easy, near-boundary,
and failing configurations, retain repetitions, and define a smaller job budget
before enabling recurrence.

References: [GitHub scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
