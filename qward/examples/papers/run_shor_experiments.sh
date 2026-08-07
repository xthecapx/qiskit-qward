#!/bin/bash
# Run Shor QPU configs SHOR-N15-M3/M4/M6/M8 at optimization_level=3.
# Each config runs an AerSimulator baseline first (same shots), then submits
# the IBM batch. Baseline is stored in the QPU JSON as simulator_baseline.
#
# Env overrides:
#   SHOR_RUNS=5 SHOR_SHOTS=4096 SHOR_TIMEOUT=3600
#   SHOR_SIM_ONLY=1          # Aer baselines only (no QPU)
#   SHOR_SKIP_SIM=1          # skip Aer (not recommended)
#   SHOR_CONFIGS="SHOR-N15-M8"  # subset
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# papers/ -> examples/ -> qward/ -> repo root
ENV_FILE="$(cd "$SCRIPT_DIR/../../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|"") continue ;;
            IBM_QUANTUM_*=*)
                key="${line%%=*}"
                val="${line#*=}"
                val="${val%\"}"; val="${val#\"}"
                val="${val%\'}"; val="${val#\'}"
                export "$key=$val"
                ;;
        esac
    done < "$ENV_FILE"
    echo "Loaded IBM credentials from $ENV_FILE"
else
    echo "No .env at $ENV_FILE (will rely on exported env or saved Qiskit account)"
fi

AUTH_ARGS=""
[ -n "${IBM_QUANTUM_TOKEN:-}" ] && AUTH_ARGS="$AUTH_ARGS --token $IBM_QUANTUM_TOKEN"
[ -n "${IBM_QUANTUM_CHANNEL:-}" ] && AUTH_ARGS="$AUTH_ARGS --channel $IBM_QUANTUM_CHANNEL"
[ -n "${IBM_QUANTUM_INSTANCE:-}" ] && AUTH_ARGS="$AUTH_ARGS --instance $IBM_QUANTUM_INSTANCE"

if [ -n "${SHOR_CONFIGS:-}" ]; then
  # shellcheck disable=SC2206
  CONFIGS=($SHOR_CONFIGS)
else
  CONFIGS=(SHOR-N15-M3 SHOR-N15-M4 SHOR-N15-M6 SHOR-N15-M8)
fi
RUNS="${SHOR_RUNS:-5}"
SHOTS="${SHOR_SHOTS:-4096}"
TIMEOUT="${SHOR_TIMEOUT:-3600}"

EXTRA_ARGS=""
if [ "${SHOR_SIM_ONLY:-0}" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --simulator-only"
  echo "Mode: AerSimulator baseline only (no QPU submit)"
elif [ "${SHOR_SKIP_SIM:-0}" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --skip-simulator"
  echo "Mode: QPU only (Aer baseline skipped)"
else
  echo "Mode: Aer baseline then QPU (opt-level 3, DD=XpXm, gate twirling)"
fi

echo "Configs: ${CONFIGS[*]}"
echo "runs=$RUNS shots=$SHOTS timeout=$TIMEOUT"
echo

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running $cfg ==="
  # shellcheck disable=SC2086
  uv run python shor/shor_ibm.py \
    --config "$cfg" \
    --opt-levels 3 \
    --runs "$RUNS" \
    --shots "$SHOTS" \
    --timeout "$TIMEOUT" \
    $EXTRA_ARGS \
    $AUTH_ARGS
done

echo
echo "QPU results:     shor/data/qpu/raw/"
echo "Aer-only saves:  shor/data/simulator/baseline/"
echo "Paste batch IDs into rsa/.env as IBM_BATCH_IDS=m3=...,m4=...,m6=...,m8=..."
