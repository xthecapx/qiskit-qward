#!/usr/bin/env bash
set -euo pipefail

# One IBM batch containing 10 BVSB27 jobs at optimization level 3.
# Default to ibm_marrakesh so the result is directly comparable with BVSB28
# and BVSB29. Set IBM_BACKEND to explicitly select another dynamic backend.

MODULE="qward.examples.papers.bv.bv_signal_background_ibm"
CONFIG="BVSB27"
IBM_BACKEND="${IBM_BACKEND:-ibm_marrakesh}"

echo "Using IBM backend: ${IBM_BACKEND}"

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
