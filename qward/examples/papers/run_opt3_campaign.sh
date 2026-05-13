#!/bin/bash
# =============================================================================
# IBM QPU - Optimization Level 3 Only Campaign
# =============================================================================
# Runs 6 targeted configs (≤6 qubits) with --opt-levels 3 only.
# One batch at a time, sequential execution.
#
# Usage:
#   IBM_QUANTUM_TOKEN="..." IBM_QUANTUM_CHANNEL="ibm_cloud" IBM_QUANTUM_INSTANCE="thecap" \
#     ./run_opt3_campaign.sh
# =============================================================================

# Go to project root so qward is importable
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$(pwd)"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

AUTH_ARGS=""
[ -n "$IBM_QUANTUM_TOKEN" ]    && AUTH_ARGS="$AUTH_ARGS --token $IBM_QUANTUM_TOKEN"
[ -n "$IBM_QUANTUM_CHANNEL" ]  && AUTH_ARGS="$AUTH_ARGS --channel $IBM_QUANTUM_CHANNEL"
[ -n "$IBM_QUANTUM_INSTANCE" ] && AUTH_ARGS="$AUTH_ARGS --instance $IBM_QUANTUM_INSTANCE"

echo "=============================================="
echo "OPT-LEVEL 3 CAMPAIGN (6 configs, ≤6 qubits)"
echo "=============================================="

run_one() {
    local algo=$1
    local config=$2
    local script=$3

    echo -e "${YELLOW}>>> [$algo] $config (opt-level 3 only)${NC}"
    uv run python "$script" --config "$config" --opt-levels 3 $AUTH_ARGS
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}>>> $algo $config completed${NC}"
    else
        echo -e "${RED}>>> $algo $config FAILED${NC}"
    fi
    echo ""
}

GROVER_SCRIPT="qward/examples/papers/grover/grover_ibm.py"
QFT_SCRIPT="qward/examples/papers/qft/qft_ibm.py"

# 1. Grover 2q — critical gap (zero opt3 data)
run_one "GROVER" "S2-1" "$GROVER_SCRIPT"

# 2. Grover 5q — only 2 data points
run_one "GROVER" "S5-1" "$GROVER_SCRIPT"

# 3. Grover 6q — only 1 data point
run_one "GROVER" "S6-1" "$GROVER_SCRIPT"

# 4. QFT 2q — only 2 vs AWS's 10
run_one "QFT" "SR2" "$QFT_SCRIPT"

# 5. QFT 3q — only 2 vs AWS's 9
run_one "QFT" "SR3" "$QFT_SCRIPT"

# 6. QFT 5q — only 3 vs AWS's 12
run_one "QFT" "SR5" "$QFT_SCRIPT"

echo "=============================================="
echo "OPT-LEVEL 3 CAMPAIGN COMPLETE"
echo "=============================================="
echo "Results saved in:"
echo "  - qward/examples/papers/grover/data/qpu/raw/"
echo "  - qward/examples/papers/qft/data/qpu/raw/"
echo "=============================================="
