#!/bin/bash
# =============================================================================
# Resume BV IBM multi-run campaign (skip already-finished BV2-ALT on marrakesh)
# =============================================================================
# Default: BV3-ALT .. BV14-ALT, opt=3, 1024 shots, 5 runs/batch.
# Omitting --backend lets Qiskit pick the least-busy operational QPU.
#
# Usage (from repo root or from this directory):
#   BV_RUNS=9 ./qward/examples/papers/run_bv_resume.sh
#   BV_RUNS=9 ./qward/examples/papers/run_bv_resume.sh bv
#   ./run_bv_resume.sh wall
#   ./run_bv_resume.sh all
#
# Modes: ladder (default) | wall | all | bv (= all)
#
# Env overrides:
#   BV_RUNS=9                       # match BV2's 9 repeats (default: 5)
#   BV_START=3 BV_END=14            # ladder range
#   BV_BACKEND=ibm_fez              # pin backend (default: auto least-busy)
#   BV_TIMEOUT=3600
#
# Examples:
#   BV_RUNS=9 ./qward/examples/papers/run_bv_resume.sh bv
#   BV_BACKEND=ibm_fez ./run_bv_resume.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    # IBM-only: do not source full .env (AWS lines with spaces around '=' break bash)
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|"") continue ;;
            IBM_QUANTUM_*=*)
                key="${line%%=*}"
                val="${line#*=}"
                # strip optional surrounding quotes
                val="${val%\"}"
                val="${val#\"}"
                val="${val%\'}"
                val="${val#\'}"
                export "$key=$val"
                ;;
        esac
    done < "$ENV_FILE"
    echo "Loaded IBM credentials from $ENV_FILE (AWS vars ignored)"
else
    echo "No .env at $ENV_FILE (will rely on exported env or saved Qiskit account)"
fi

AUTH_ARGS=""
if [ -n "${IBM_QUANTUM_TOKEN:-}" ]; then
    AUTH_ARGS="$AUTH_ARGS --token $IBM_QUANTUM_TOKEN"
fi
if [ -n "${IBM_QUANTUM_CHANNEL:-}" ]; then
    AUTH_ARGS="$AUTH_ARGS --channel $IBM_QUANTUM_CHANNEL"
fi
if [ -n "${IBM_QUANTUM_INSTANCE:-}" ]; then
    AUTH_ARGS="$AUTH_ARGS --instance $IBM_QUANTUM_INSTANCE"
fi

BV_RUNS="${BV_RUNS:-5}"
BV_START="${BV_START:-3}"
BV_END="${BV_END:-14}"
BV_TIMEOUT="${BV_TIMEOUT:-3600}"
BV_BACKEND="${BV_BACKEND:-}"

BACKEND_ARGS=""
if [ -n "$BV_BACKEND" ]; then
    BACKEND_ARGS="--backend $BV_BACKEND"
fi

BV_ARGS="--opt-levels 3 --shots 1024 --runs ${BV_RUNS} --timeout ${BV_TIMEOUT} ${BACKEND_ARGS}"

run_config() {
    local config=$1
    echo -e "${YELLOW}>>> Running BV: ${config} ${BV_ARGS}${NC}"
    # shellcheck disable=SC2086
    if uv run python bv/bv_ibm.py --config "$config" $AUTH_ARGS $BV_ARGS; then
        echo -e "${GREEN}>>> ${config} completed successfully${NC}"
    else
        echo -e "${RED}>>> ${config} FAILED${NC}"
        exit 1
    fi
    echo ""
}

run_ladder() {
    echo "=============================================="
    echo "BV LADDER RESUME (BV${BV_START}-ALT .. BV${BV_END}-ALT)"
    echo "  runs=${BV_RUNS}  shots=1024  opt=3"
    echo "  backend=${BV_BACKEND:-auto (least busy)}"
    echo "=============================================="
    local n
    for n in $(seq "$BV_START" "$BV_END"); do
        run_config "BV${n}-ALT"
    done
}

run_wall() {
    echo "=============================================="
    echo "BV BEYOND-WALL (BV29/30/31-ALT)"
    echo "  runs=${BV_RUNS}  shots=1024  opt=3"
    echo "  backend=${BV_BACKEND:-auto (least busy)}"
    echo "=============================================="
    local n
    for n in 29 30 31; do
        run_config "BV${n}-ALT"
    done
}

MODE="${1:-ladder}"
case "$MODE" in
    ladder|"")
        run_ladder
        ;;
    wall)
        run_wall
        ;;
    all|bv)
        run_ladder
        run_wall
        ;;
    *)
        echo "Usage: $0 [ladder|wall|all|bv]"
        exit 1
        ;;
esac

echo "=============================================="
echo "BV RESUME COMPLETE"
echo "Results: bv/data/qpu/raw/"
echo "=============================================="
