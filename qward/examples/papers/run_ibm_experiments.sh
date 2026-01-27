#!/bin/bash
# =============================================================================
# IBM QPU Experiment Runner
# =============================================================================
# Run this script to execute all missing experiments on IBM Quantum hardware.
# Each experiment runs with optimization levels 0, 1, 2, 3 by default.
#
# Usage:
#   ./run_ibm_experiments.sh [grover|qft|all]
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

# Build authentication arguments
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
    echo "Using authentication from environment variables"
else
    echo "Using saved IBM Quantum credentials"
fi
echo ""

# Function to run a single experiment
run_experiment() {
    local algo=$1
    local config=$2
    local script=$3
    
    echo -e "${YELLOW}>>> Running $algo: $config${NC}"
    python "$script" --config "$config" $AUTH_ARGS
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
# MAIN
# =============================================================================
case "${1:-all}" in
    grover)
        run_grover
        ;;
    qft)
        run_qft
        ;;
    all|*)
        run_grover
        run_qft
        ;;
esac

echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Results saved in:"
echo "  - grover/data/qpu/raw/"
echo "  - qft/data/qpu/raw/"
echo "=============================================="
