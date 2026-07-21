#!/bin/bash
# =============================================================================
# IBM QPU Experiment Runner
# =============================================================================
# Run this script to execute all missing experiments on IBM Quantum hardware.
# Each experiment runs with optimization levels 0, 1, 2, 3 by default.
#
# Usage:
#   ./run_ibm_experiments.sh [grover|qft|bv|bv-wall|all]
#
# Environment Variables (optional - will use saved credentials if not set):
#   IBM_QUANTUM_TOKEN    - Your IBM Quantum API token
#   IBM_QUANTUM_CHANNEL  - Channel: 'ibm_quantum' or 'ibm_cloud'
#   IBM_QUANTUM_INSTANCE - Instance: e.g., 'ibm-q/open/main'
#
# Examples:
#   # Use saved credentials
#   ./run_ibm_experiments.sh
#
#   # BV beyond-wall section only (29/30/31 ALT, opt=3, 1024 shots, 5 runs)
#   ./run_ibm_experiments.sh bv-wall
#
#   # BV ladder with 10 repeats per config
#   BV_RUNS=10 ./run_ibm_experiments.sh bv-ladder
#
#   # Use environment variables
#   export IBM_QUANTUM_TOKEN="your_token_here"
#   export IBM_QUANTUM_CHANNEL="ibm_quantum"
#   ./run_ibm_experiments.sh grover
#
#   # Or inline
#   IBM_QUANTUM_TOKEN="xxx" IBM_QUANTUM_CHANNEL="ibm_quantum" ./run_ibm_experiments.sh qft
#
# =============================================================================

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load IBM vars only from repo-root .env (skip AWS / invalid bash assignments)
ENV_FILE="$(cd "$(dirname "$0")/../../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|"") continue ;;
            IBM_QUANTUM_*=*)
                key="${line%%=*}"
                val="${line#*=}"
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

# Build authentication arguments (from .env or already-exported env)
AUTH_ARGS=""
if [ -n "$IBM_QUANTUM_TOKEN" ]; then
    AUTH_ARGS="$AUTH_ARGS --token $IBM_QUANTUM_TOKEN"
fi
if [ -n "$IBM_QUANTUM_CHANNEL" ]; then
    AUTH_ARGS="$AUTH_ARGS --channel $IBM_QUANTUM_CHANNEL"
fi
if [ -n "$IBM_QUANTUM_INSTANCE" ]; then
    AUTH_ARGS="$AUTH_ARGS --instance $IBM_QUANTUM_INSTANCE"
fi

echo "=============================================="
echo "IBM QPU EXPERIMENT RUNNER"
echo "=============================================="
if [ -n "$AUTH_ARGS" ]; then
    echo "Using authentication from environment / .env"
    echo "  channel=${IBM_QUANTUM_CHANNEL:-"(unset)"}"
    echo "  instance=${IBM_QUANTUM_INSTANCE:-"(unset)"}"
else
    echo "Using saved IBM Quantum credentials (no IBM_QUANTUM_* found)"
fi
echo ""

# Function to run a single experiment
run_experiment() {
    local algo=$1
    local config=$2
    local script=$3
    shift 3
    local extra_args="$*"

    echo -e "${YELLOW}>>> Running $algo: $config ${extra_args}${NC}"
    uv run python "$script" --config "$config" $AUTH_ARGS $extra_args
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}>>> $algo $config completed successfully${NC}"
    else
        echo -e "${RED}>>> $algo $config FAILED${NC}"
    fi
    echo ""
}

# =============================================================================
# GROVER EXPERIMENTS
# =============================================================================
run_grover() {
    echo "=============================================="
    echo "GROVER EXPERIMENTS"
    echo "=============================================="
    
    # Priority 1: Small circuits (highest success probability)
    # Already done: S2-1, H4-0, S6-1
    
    # 3-qubit configs (very fast, high success)
    run_experiment "GROVER" "ASYM-1" "grover/grover_ibm.py"
    run_experiment "GROVER" "ASYM-2" "grover/grover_ibm.py"
    run_experiment "GROVER" "M3-2" "grover/grover_ibm.py"
    run_experiment "GROVER" "SYM-1" "grover/grover_ibm.py"
    run_experiment "GROVER" "SYM-2" "grover/grover_ibm.py"
    run_experiment "GROVER" "H3-3" "grover/grover_ibm.py"
    run_experiment "GROVER" "H3-2" "grover/grover_ibm.py"
    run_experiment "GROVER" "S3-1" "grover/grover_ibm.py"
    run_experiment "GROVER" "M3-1" "grover/grover_ibm.py"
    
    # 4-qubit configs (medium depth)
    run_experiment "GROVER" "M4-4" "grover/grover_ibm.py"
    run_experiment "GROVER" "M4-2" "grover/grover_ibm.py"
    run_experiment "GROVER" "S4-1" "grover/grover_ibm.py"
    run_experiment "GROVER" "H4-4" "grover/grover_ibm.py"
    
    # 5-qubit config
    run_experiment "GROVER" "S5-1" "grover/grover_ibm.py"
    
    # 7-qubit config (deep circuit)
    run_experiment "GROVER" "S7-1" "grover/grover_ibm.py"
    
    # 8-qubit config (very deep - may fail due to decoherence)
    run_experiment "GROVER" "S8-1" "grover/grover_ibm.py"
}

# =============================================================================
# QFT EXPERIMENTS
# =============================================================================
run_qft() {
    echo "=============================================="
    echo "QFT EXPERIMENTS"
    echo "=============================================="
    
    # Already done: SR7
    
    # Small roundtrip configs (highest success)
    run_experiment "QFT" "SR2" "qft/qft_ibm.py"
    run_experiment "QFT" "SR3" "qft/qft_ibm.py"
    run_experiment "QFT" "SR4" "qft/qft_ibm.py"
    run_experiment "QFT" "SR5" "qft/qft_ibm.py"
    run_experiment "QFT" "SR6" "qft/qft_ibm.py"
    
    # Period detection configs (4-6 qubits)
    run_experiment "QFT" "PV4-P8" "qft/qft_ibm.py"
    run_experiment "QFT" "SP4-P4" "qft/qft_ibm.py"
    run_experiment "QFT" "PV4-P4" "qft/qft_ibm.py"
    run_experiment "QFT" "PV6-P16" "qft/qft_ibm.py"
    run_experiment "QFT" "SP5-P4" "qft/qft_ibm.py"
    run_experiment "QFT" "PV6-P8" "qft/qft_ibm.py"
    run_experiment "QFT" "SP6-P8" "qft/qft_ibm.py"
    
    # Input variation configs
    run_experiment "QFT" "IV4-0000" "qft/qft_ibm.py"
    run_experiment "QFT" "IV4-0101" "qft/qft_ibm.py"
    
    # Large configs (8-10 qubits) - QFT scales well!
    run_experiment "QFT" "SR8" "qft/qft_ibm.py"
    run_experiment "QFT" "SR10" "qft/qft_ibm.py"
    run_experiment "QFT" "SP8-P4" "qft/qft_ibm.py"
    run_experiment "QFT" "SP10-P4" "qft/qft_ibm.py"
}

# =============================================================================
# BV EXPERIMENTS
# =============================================================================
# Beyond-wall section: n_secret = 29/30/31 (total qubits 30/31/32).
# Ideal HF/TVDF require a statevector past this machine's wall; DSR does not.
# Waits via executor poll_interval=10 until DONE or --timeout.
# Override repeats with: BV_RUNS=10 ./run_ibm_experiments.sh bv-ladder
BV_RUNS="${BV_RUNS:-5}"

run_bv_wall() {
    echo "=============================================="
    echo "BV BEYOND-WALL EXPERIMENTS (HF/TVDF section)"
    echo "=============================================="
    local BV_ARGS="--opt-levels 3 --shots 1024 --runs ${BV_RUNS} --timeout 3600"
    run_experiment "BV" "BV29-ALT" "bv/bv_ibm.py" $BV_ARGS
    run_experiment "BV" "BV30-ALT" "bv/bv_ibm.py" $BV_ARGS
    run_experiment "BV" "BV31-ALT" "bv/bv_ibm.py" $BV_ARGS
}

# Combined-plot ladder (ALT, n=2..14) for 1_combined_dsr_comparison_ibm.
run_bv_ladder() {
    echo "=============================================="
    echo "BV SCALABILITY LADDER (combined DSR figure)"
    echo "=============================================="
    local BV_ARGS="--opt-levels 3 --shots 1024 --runs ${BV_RUNS} --timeout 3600"
    for n in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
        run_experiment "BV" "BV${n}-ALT" "bv/bv_ibm.py" $BV_ARGS
    done
}

run_bv() {
    run_bv_ladder
    run_bv_wall
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-all}" in
    grover)
        run_grover
        ;;
    qft)
        run_qft
        ;;
    bv)
        run_bv
        ;;
    bv-wall)
        run_bv_wall
        ;;
    bv-ladder)
        run_bv_ladder
        ;;
    all)
        run_grover
        run_qft
        run_bv
        ;;
    *)
        echo "Usage: $0 [grover|qft|bv|bv-wall|bv-ladder|all]"
        exit 1
        ;;
esac

echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results saved in:"
echo "  - grover/data/qpu/raw/"
echo "  - qft/data/qpu/raw/"
echo "  - bv/data/qpu/raw/"
echo "=============================================="
