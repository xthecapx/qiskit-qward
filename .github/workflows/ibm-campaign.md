# Automated IBM Fez campaign

`ibm-campaign.yml` runs the reviewed full experiment profile for a collection date and opens a results PR against `main`. It does not merge the PR. Each full collection is 172 configurations, 344 jobs, and 1,024 shots per job. The default is at most three active IBM batches. This first test is the full research campaign; the smaller monthly dashboard campaign is a follow-up design.

## Activate the November 5 test

1. Commit the workflow, `scripts/ibm_campaign_ci.py`, the checkpoint change in `campaign_submit.py`, tests, and documentation; push/merge them into `main` **before November 5, 2026, 11:45 a.m. America/Bogota**. Scheduled workflows must exist on the default branch.
2. Under repository **Settings → Secrets and variables → Actions → Secrets**, add:
   - `IBM_QUANTUM_TOKEN`: the IBM API token for this campaign.
   - `IBM_QUANTUM_INSTANCE`: the same IBM instance/CRN used for the local windows.
   These are secrets, not plain repository variables. Do not commit `.env`.
3. Under **Variables**, set (or keep the workflow defaults):
   - `IBM_CAMPAIGN_DATE`: `2026-11-05` (Bogotá calendar date).
   - `IBM_QUANTUM_CHANNEL`: `ibm_quantum_platform`.
4. Under **Settings → Actions → General → Workflow permissions**, enable **Allow GitHub Actions to create and approve pull requests**. The workflow requests `contents: write` and `pull-requests: write`.
5. In Actions, choose **Collect IBM campaign results → Run workflow**, branch `main`, mode `validate`, campaign date `2026-11-05`. This checks the plan and job count without IBM calls, branch creation, or spending QPU credits. It does not verify secret validity or IBM connectivity.

The cron is `45 16 * * *`: **16:45 UTC = 11:45 a.m. Bogotá**. Its date guard allows scheduled execution only on `IBM_CAMPAIGN_DATE`, defaulting to November 5. Later daily triggers exit without IBM calls. GitHub can delay a scheduled run; 11:45 is the requested workflow trigger, not a guaranteed QPU start. Dependency installation, preparation, and the IBM queue follow it.

If the files reach `main` after the scheduled slot, use **Run workflow → mode `run` → campaign date `2026-11-05`**, or choose a future date. The workflow runs only on `main`; opening the setup as a PR alone does not activate the schedule.

## Lifecycle and credit protection

- All cloud collections share one GitHub concurrency group with cancellation disabled.
- The workflow validates the backend, campaign, window, shots, plan budgets, and 344-job total before contacting IBM.
- Before preparing or submitting, it pushes a permanent claim on `codex/ibm-fez-2026-11-05` (using the chosen ISO date). Existing claims or result branches block fresh `run` attempts, including reruns of a workflow.
- Fresh runs also refuse existing manifests. Do not run the same campaign date locally while the cloud workflow owns it; a local runner cannot participate in GitHub's lock.
- Preparation uses the same local plans and metrics cache. CPU PyTorch is installed explicitly. A different PyTorch build/version can cause a cache miss; the first cloud preparation can therefore take longer. Newly computed cache entries are checkpointed.
- The runner pushes records after preparation, after each batch submission, and after each batch recovery. A failed checkpoint stops further submissions.
- The collection step has a four-hour timeout, leaving time in the job for backup and PR creation. A timeout does not cancel already submitted IBM jobs.
- Available manifests, prepared circuits, journals and tables are uploaded as an Actions artifact with 90-day retention, including on failure. The final step also commits remaining local state to the results branch. A forcibly terminated runner can still lose writes since its last pushed checkpoint; inspect IBM before resolving an incomplete batch.
- Successful collections require every manifest to be completed with the expected unique job IDs. They export `dataset.csv`, `references.csv`, and `calibrations.csv`, validate the static dashboard build, and open a ready-for-review PR. The tables include earlier periods as well.
- Failures open or update a draft results PR containing the available state. Read the workflow logs before merging anything. Missing PR permission can prevent PR creation even when the branch and artifact were saved.

The marker remains after merging the results. Keep it: it prevents rerunning a completed collection even if its results branch is deleted. These checks bound job count; IBM's actual billed seconds still depend on execution. Review per-job usage in the journals/export and use the existing `python -m qward.examples.papers.ghz.ghz_ibm --plan-usage qward/examples/papers/*/plans/*-d2026-11-05_manifest.json` command after checking out the results branch for the provider's billing report.

## Recover an interrupted run

In Actions, manually dispatch the same campaign date with mode **`recover`**. It checks out that date's results branch, waits for and retrieves existing submissions, then exports and updates the same PR if the collection is complete. Recovery never prepares a new circuit or submits another job. Missing or merely prepared batches remain incomplete, so they cannot silently trigger fresh spending.

A partial batch, a missing submission ID, or a preparation failure needs manual inspection. Download the latest artifact and compare its journals against the results branch and IBM's jobs before deciding how to resume. Do not delete a claim to force a retry while any submission may already exist. If the workflow failed before any IBM call, the claim can be removed only after verifying that no jobs were sent.

After recovery, approve any requested PR workflow runs, review the results, and merge manually into `main`. The normal docs workflow then publishes the dashboard. GitHub documents that PRs created with `GITHUB_TOKEN` may require a user with write access to approve their workflow runs. No extra personal access token is needed for this workflow.

## Next collection at another time

Set `IBM_CAMPAIGN_DATE` to the desired Bogotá date. No window counter is needed. Change the cron for the desired time and merge it before that slot. Bogotá is UTC−5: for example, 14:30 Bogotá is `30 19 * * *`. Or manually dispatch `campaign_date=2026-11-06`, `mode=run` at your chosen time.

The ISO date identifies the collection, independently of when a delayed workflow or recovery finishes. Plans are named `<algorithm>-dYYYY-MM-DD.json`, the campaign ID is `fez-dataset-v2-YYYY-MM-DD`, and the branch and claim also use that date. The old numeric `window` field is retained as `YYYYMMDD` for schema compatibility. Existing window 0/1/2/3 plans and results remain unchanged; the dated workflow uses the reviewed window-2 configuration set as its full-profile template.

Merge each completed results PR before starting the next date so the next export includes the latest data. Recovery always uses the original collection date, even on a later day. The date is an identity, not automatic advancement: a new date requests another full collection. `IBM_CAMPAIGN_WINDOW` is no longer used. Do not run the legacy numbered equivalent locally in parallel with a dated collection.

## Monthly dashboard follow-up

Do not change this full-profile workflow to monthly execution as-is. Define a dedicated reduced profile first, using the observed algorithm boundaries to select representative easy, near-boundary, and failing configurations. Preserve repeated runs so variation remains visible. Agree on a maximum configurations/jobs budget, shots, and a QPU-seconds monitoring policy.

That profile should have its own dated campaign identity and checkpoint branch per month, a monthly schedule, and the same export → validation → results PR process. The next two dated research collections will help choose the reduced set; this workflow intentionally does not automatically repeat the 344-job research campaign every month.

## Offline checks

```sh
python scripts/ibm_campaign_ci.py validate --date 2026-11-05
python scripts/ibm_campaign_ci.py validate --date 2026-11-06
python -m pytest tests/test_ibm_campaign_ci.py tests/test_campaign_submit.py -q
```

The `run` and `recover` modes contact IBM. Use them locally only when deliberately managing the corresponding collection. `CAMPAIGN_CHECKPOINT=1` enables Git pushes and is set only by the workflow; the ordinary local campaign driver remains unchanged unless a checkpoint callback is explicitly supplied.

References: [scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule), [repository Actions settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository), [GITHUB_TOKEN workflow triggers](https://docs.github.com/en/actions/concepts/security/github_token).
