#!/usr/bin/env bash
# Driver for the ibm_fez dataset campaign (docs/ibm_fez_dataset_plan.md).
#
#   run_campaign.sh prepare <algo|all> <window>   freeze records for one plan (no jobs)
#   run_campaign.sh submit  <algo|all> <window>   submit that manifest in ONE Batch
#   run_campaign.sh recover <algo|all> <window>   fetch results by saved job IDs
#   run_campaign.sh wait    <algo> <window>       block until that Batch's jobs finish
#   run_campaign.sh status  [algo] [window]   offline: planned vs submitted vs done
#   run_campaign.sh usage   [window]          QPU seconds billed by IBM vs the estimate
#   run_campaign.sh export  [out_dir]         build dataset.csv from completed records
#
# <algo>: ghz bv vtp qft pe grover rv shor (or "all" for the 8 of them)
# <window>: 0 = pilot (30 jobs), 1-3 = campaign windows
# Plans live in  <algo>/plans/<algo>-w<window>.json  (format: docs/ibm_fez_dataset_plan.md,
# "Plan files"); manifests are written beside them as <algo>-w<window>_manifest.json.
#
# Credentials come from IBM_QUANTUM_* in the repo-root .env or the environment.
# Nothing is passed on the command line, so tokens never appear in `ps`.
# Window 0 is the pilot (30 jobs, run it first); windows 1-3 are the campaign.
# `submit all` runs ONE Batch at a time (open, wait until it finishes, recover,
# then the next algorithm): IBM limits how many Batches can be active, and a
# queued Batch stays active. If a Batch cannot be opened, nothing was sent and
# the manifest stays prepared; rerun `submit` for that algorithm later.
# Only `submit` sends jobs. Prepare each window just before submitting it so its
# calibration is current, and keep windows on different days.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="${HERE}/$(basename "$0")"  # Absolute: the script cd's to the repo root below.
REPO="$(cd "${HERE}/../../.." && pwd)"
cd "${REPO}"

usage() { sed -n '2,25p' "${SELF}" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

module_for() {
    case "$1" in
        ghz)        echo "qward.examples.papers.ghz.ghz_ibm" ;;
        bv)         echo "qward.examples.papers.bv.bv_ibm" ;;
        vtp)        echo "qward.examples.papers.vtp.vtp_ibm" ;;
        qft)        echo "qward.examples.papers.qft.qft_ibm" ;;
        pe)         echo "qward.examples.papers.phase_estimation.pe_ibm" ;;
        grover)     echo "qward.examples.papers.grover.grover_ibm" ;;
        rv)         echo "qward.examples.papers.random_volumetric.rv_ibm" ;;
        shor)       echo "qward.examples.papers.shor.shor_ibm" ;;
        *) echo "unknown algorithm: $1" >&2; exit 2 ;;
    esac
}

ALGOS="ghz bv vtp qft pe grover rv shor"
dir_of() {
    case "$1" in
        pe) echo "phase_estimation" ;;
        rv) echo "random_volumetric" ;;
        *) echo "$1" ;;
    esac
}
plan_file() { echo "${HERE}/$(dir_of "$1")/plans/$1-w$2.json"; }
manifest_file() { echo "${HERE}/$(dir_of "$1")/plans/$1-w$2_manifest.json"; }

require_window() {
    [[ "$1" =~ ^[0-9]+$ ]] || { echo "window must be 0 (pilot) or 1-3" >&2; exit 2; }
}

ACTION="${1:-}"
[[ -n "${ACTION}" ]] || usage 1
shift || true

case "${ACTION}" in
    prepare|submit|recover|wait)
        [[ $# -eq 2 ]] || usage 2
        ALGO="$1"; WINDOW="$2"; require_window "${WINDOW}"
        if [[ "${ALGO}" == "all" ]]; then
            # One Batch per algorithm; `submit all` asks once for the whole window.
            if [[ "${ACTION}" == "submit" ]]; then
                for A in ${ALGOS}; do
                    [[ -f "$(manifest_file "${A}" "${WINDOW}")" ]] \
                        || { echo "prepare ${A} ${WINDOW} first" >&2; exit 2; }
                done
                "${SELF}" status
                read -r -p "Submit ALL ${ALGOS} (window ${WINDOW}) to IBM, one Batch at a time? [y/N] " ANSWER
                [[ "${ANSWER}" == "y" || "${ANSWER}" == "Y" ]] || { echo "aborted"; exit 1; }
                for A in ${ALGOS}; do
                    M="$(manifest_file "${A}" "${WINDOW}")"
                    # Skip what is already in IBM so a rerun resumes where it stopped.
                    if grep -q '"status": "prepared"' "${M}"; then
                        echo "=== ${A}: opening Batch"
                        uv run -m "$(module_for "${A}")" --submit-plan "${M}" > /dev/null \
                            || { echo "!!! ${A}: Batch not opened or submit failed; stopping. Rerun 'submit all ${WINDOW}' to resume." >&2; exit 1; }
                    else
                        echo "=== ${A}: already submitted"
                    fi
                    echo "=== ${A}: waiting for its jobs (Ctrl-C is safe; rerun to resume)"
                    uv run -m "$(module_for "${A}")" --wait-plan "${M}" > /dev/null
                    uv run -m "$(module_for "${A}")" --recover-plan "${M}" > /dev/null
                    echo "=== ${A}: done"
                done
            else
                for A in ${ALGOS}; do "${SELF}" "${ACTION}" "${A}" "${WINDOW}"; done
            fi
            exit 0
        fi
        MODULE="$(module_for "${ALGO}")"
        case "${ACTION}" in
            prepare)
                PLAN="$(plan_file "${ALGO}" "${WINDOW}")"
                [[ -f "${PLAN}" ]] || { echo "missing plan: ${PLAN}" >&2; exit 2; }
                echo "Preparing ${ALGO} window ${WINDOW} (no jobs are submitted)"
                uv run -m "${MODULE}" --prepare-plan "${PLAN}" --shots 1024
                ;;
            submit)
                MANIFEST="$(manifest_file "${ALGO}" "${WINDOW}")"
                [[ -f "${MANIFEST}" ]] || { echo "prepare first: ${MANIFEST}" >&2; exit 2; }
                uv run -m "${MODULE}" --plan-status "${MANIFEST}"
                read -r -p "Submit ${ALGO} window ${WINDOW} to IBM as one Batch? [y/N] " ANSWER
                [[ "${ANSWER}" == "y" || "${ANSWER}" == "Y" ]] || { echo "aborted"; exit 1; }
                uv run -m "${MODULE}" --submit-plan "${MANIFEST}"
                ;;
            recover)
                MANIFEST="$(manifest_file "${ALGO}" "${WINDOW}")"
                [[ -f "${MANIFEST}" ]] || { echo "no manifest: ${MANIFEST}" >&2; exit 2; }
                uv run -m "${MODULE}" --recover-plan "${MANIFEST}"
                ;;
            wait)
                MANIFEST="$(manifest_file "${ALGO}" "${WINDOW}")"
                [[ -f "${MANIFEST}" ]] || { echo "no manifest: ${MANIFEST}" >&2; exit 2; }
                uv run -m "${MODULE}" --wait-plan "${MANIFEST}"
                ;;
        esac
        ;;
    status)
        FILTER_ALGO="${1:-}"; FILTER_WINDOW="${2:-}"
        printf '%-10s %-7s %-10s %-10s %-10s %s\n' ALGO WINDOW STATUS PLANNED SUBMITTED RECORDS
        for ALGO in ${ALGOS}; do
            [[ -z "${FILTER_ALGO}" || "${FILTER_ALGO}" == "${ALGO}" ]] || continue
            for MANIFEST in "${HERE}/$(dir_of "${ALGO}")"/plans/${ALGO}-w*_manifest.json; do
                [[ -f "${MANIFEST}" ]] || continue
                WINDOW="$(basename "${MANIFEST}" | sed -E "s/^${ALGO}-w([0-9]+)_manifest.json$/\1/")"
                [[ -z "${FILTER_WINDOW}" || "${FILTER_WINDOW}" == "${WINDOW}" ]] || continue
                uv run -m "$(module_for "${ALGO}")" --plan-status "${MANIFEST}" 2>/dev/null \
                    | uv run python -c "
import json, sys
s = json.load(sys.stdin)
print('%-10s %-7s %-10s %-10s %-10s %s' % ('${ALGO}', '${WINDOW}', s['status'], s['jobs_planned'], s['jobs_submitted'], s['records_by_status']))"
            done
        done
        ;;
    usage)
        FILTER_WINDOW="${1:-}"
        MANIFESTS=()
        for ALGO in ${ALGOS}; do
            for MANIFEST in "${HERE}/$(dir_of "${ALGO}")"/plans/${ALGO}-w*_manifest.json; do
                [[ -f "${MANIFEST}" ]] || continue
                W="$(basename "${MANIFEST}" | sed -E "s/^${ALGO}-w([0-9]+)_manifest.json$/\1/")"
                [[ -z "${FILTER_WINDOW}" || "${FILTER_WINDOW}" == "${W}" ]] && MANIFESTS+=("${MANIFEST}")
            done
        done
        [[ ${#MANIFESTS[@]} -gt 0 ]] || { echo "no manifests" >&2; exit 2; }
        uv run -m qward.examples.papers.ghz.ghz_ibm --plan-usage "${MANIFESTS[@]}" \
            | uv run python -c "
import json, sys
u = json.load(sys.stdin)
print('%-20s %-7s %-6s %-14s %s' % ('ALGO', 'WINDOW', 'JOBS', 'IBM BATCH (s)', 'SAVED PER-JOB (s)'))
for b in u['batches']:
    print('%-20s %-7s %-6s %-14s %s' % (b['algorithm'], b['window'], b['jobs'], b['batch_usage_seconds'], b['saved_job_quantum_seconds']))
a = u['account']
print('total billed: %s s | account: %s of %s s used, %s s left' % (u['total_batch_usage_seconds'], a['usage_consumed_seconds'], a['usage_limit_seconds'], a['usage_remaining_seconds']))"
        ;;
    export)
        OUT="${1:-${HERE}/dataset_out}"
        uv run -m qward.examples.papers.dataset_export --backend ibm_fez --out "${OUT}"
        ;;
    -h|--help|help)
        usage 0
        ;;
    *)
        usage 2
        ;;
esac
