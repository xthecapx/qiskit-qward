#!/usr/bin/env bash
set -euo pipefail

# Two separate IBM batches, each containing 10 jobs at optimization level 3.
# If IBM_BACKEND is omitted, select the least-busy operational backend with at
# least 29 qubits and dynamic-circuit support. The selected backend is then
# pinned for both batches so their results remain comparable.

MODULE="qward.examples.papers.bv.bv_signal_background_ibm"

if [[ -z "${IBM_BACKEND:-}" ]]; then
    echo "Selecting least-busy dynamic-circuit backend..." >&2
    IBM_BACKEND="$(uv run -m "${MODULE}" --select-backend-only)"
fi

echo "Using IBM backend: ${IBM_BACKEND}"

for CONFIG in BVSB28 BVSB29; do
    echo "Preflight for ${CONFIG}"
    uv run -m "${MODULE}" \
        --config "${CONFIG}" \
        --preflight-only \
        --opt-levels 3 \
        --shots 1024 \
        --runs 10

    echo "Submitting ${CONFIG}: 10 jobs, 1024 shots, optimization level 3"
    uv run -m "${MODULE}" \
        --config "${CONFIG}" \
        --backend "${IBM_BACKEND}" \
        --opt-levels 3 \
        --shots 1024 \
        --runs 10 \
        --timeout 7200
done
