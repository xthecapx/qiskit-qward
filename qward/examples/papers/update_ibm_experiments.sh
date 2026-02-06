#!/bin/bash
# =============================================================================
# IBM QPU Experiment Data Updater & Recovery Tool
# =============================================================================
# This script connects to IBM Quantum Cloud to:
#   1. Update experiment data files with missing histogram results
#   2. Recover timed-out batches that never saved JSON files
#
# It scans all saved JSON files, identifies ones with issues, and fetches
# the actual job results from IBM Cloud using the stored batch_id and job_ids.
#
# Usage:
#   ./update_ibm_experiments.sh [grover|qft|all] [action] [options]
#
# Actions:
#   --status      Only show status of data files (no updates)
#   --update      Update existing JSON files with missing data (default)
#   --recover     Recover timed-out batches (creates new JSON files)
#   --full        Run both --update and --recover
#
# Options:
#   --dry-run     Show what would be updated/recovered without making changes
#   --force       Re-fetch all jobs from IBM even if data looks healthy
#
# Environment Variables (optional - will use saved credentials if not set):
#   IBM_QUANTUM_TOKEN    - Your IBM Quantum API token
#   IBM_QUANTUM_CHANNEL  - Channel: 'ibm_quantum' or 'ibm_cloud'
#   IBM_QUANTUM_INSTANCE - Instance: e.g., 'ibm-q/open/main'
#
# Examples:
#   # Check status of all data files
#   ./update_ibm_experiments.sh all --status
#
#   # Dry run - see what would be updated
#   ./update_ibm_experiments.sh all --dry-run
#
#   # Update existing files + recover timed-out batches
#   ./update_ibm_experiments.sh all --full
#
#   # Only recover timed-out batches (creates new JSON files)
#   ./update_ibm_experiments.sh all --recover
#
#   # Recover only QFT timed-out batches (dry run)
#   ./update_ibm_experiments.sh qft --recover --dry-run
#
#   # Update only Grover experiments
#   ./update_ibm_experiments.sh grover
#
#   # Force re-fetch all QFT jobs from IBM
#   ./update_ibm_experiments.sh qft --force
#
#   # Use environment variables for authentication
#   IBM_QUANTUM_TOKEN="xxx" ./update_ibm_experiments.sh all --full
#
# =============================================================================

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Parse arguments
ALGO="${1:-all}"
shift || true  # Shift past the algorithm argument

# Collect flags
DO_UPDATE=true
DO_RECOVER=false
DO_STATUS=false
EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --status)
            DO_STATUS=true
            DO_UPDATE=false
            DO_RECOVER=false
            ;;
        --update)
            DO_UPDATE=true
            DO_RECOVER=false
            ;;
        --recover)
            DO_UPDATE=false
            DO_RECOVER=true
            ;;
        --full)
            DO_UPDATE=true
            DO_RECOVER=true
            ;;
        --dry-run|--force)
            EXTRA_FLAGS="$EXTRA_FLAGS $arg"
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Usage: $0 [grover|qft|all] [--status|--update|--recover|--full] [--dry-run|--force]"
            exit 1
            ;;
    esac
done

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

echo -e "${CYAN}${BOLD}=============================================="
echo "IBM QPU EXPERIMENT DATA UPDATER & RECOVERY"
echo "==============================================${NC}"
if [ -n "$AUTH_ARGS" ]; then
    echo "Auth: Using environment variables"
else
    echo "Auth: Using saved IBM Quantum credentials"
fi
echo ""
if $DO_STATUS; then
    echo -e "Action: ${BOLD}STATUS CHECK${NC}"
elif $DO_UPDATE && $DO_RECOVER; then
    echo -e "Action: ${BOLD}FULL UPDATE + RECOVERY${NC}"
elif $DO_RECOVER; then
    echo -e "Action: ${BOLD}RECOVER TIMED-OUT BATCHES${NC}"
else
    echo -e "Action: ${BOLD}UPDATE EXISTING FILES${NC}"
fi
echo ""

# Function to run status check
run_status() {
    local algo_name=$1
    local script=$2

    echo -e "${YELLOW}>>> $algo_name: Checking data files...${NC}"
    python "$script" --status

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}>>> $algo_name status check completed${NC}"
    else
        echo -e "${RED}>>> $algo_name status check FAILED (exit code: $exit_code)${NC}"
    fi
    echo ""
}

# Function to run update on existing files
run_update() {
    local algo_name=$1
    local script=$2

    echo -e "${YELLOW}>>> $algo_name: Updating existing data files...${NC}"
    python "$script" --update $AUTH_ARGS $EXTRA_FLAGS

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}>>> $algo_name update completed successfully${NC}"
    else
        echo -e "${RED}>>> $algo_name update FAILED (exit code: $exit_code)${NC}"
    fi
    echo ""
}

# Function to recover timed-out batches
run_recover() {
    local algo_name=$1
    local script=$2

    echo -e "${YELLOW}>>> $algo_name: Recovering timed-out batches...${NC}"
    python "$script" --recover $AUTH_ARGS $EXTRA_FLAGS

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}>>> $algo_name recovery completed successfully${NC}"
    else
        echo -e "${RED}>>> $algo_name recovery FAILED (exit code: $exit_code)${NC}"
    fi
    echo ""
}

# Function to process an algorithm
process_algo() {
    local algo_name=$1
    local script=$2

    if $DO_STATUS; then
        run_status "$algo_name" "$script"
    fi

    if $DO_UPDATE; then
        run_update "$algo_name" "$script"
    fi

    if $DO_RECOVER; then
        run_recover "$algo_name" "$script"
    fi
}

# =============================================================================
# MAIN
# =============================================================================
case "$ALGO" in
    grover)
        process_algo "GROVER" "grover/grover_ibm.py"
        ;;
    qft)
        process_algo "QFT" "qft/qft_ibm.py"
        ;;
    all|*)
        process_algo "GROVER" "grover/grover_ibm.py"
        process_algo "QFT" "qft/qft_ibm.py"
        ;;
esac

echo -e "${CYAN}${BOLD}=============================================="
echo "ALL OPERATIONS COMPLETE"
echo "==============================================${NC}"
echo ""
echo "Timed-out batches from Feb 5 2026 campaign:"
echo "  GROVER:"
echo "    M3-1:    batch=265019eb... (ibm_fez, 3 qubits)"
echo "  QFT:"
echo "    SP5-P4:  batch=1ed95be0... (ibm_marrakesh, 5 qubits)"
echo "    SR8:     batch=ca57b497... (ibm_fez, 8 qubits)"
echo "    SR10:    batch=dbf406ea... (ibm_fez, 10 qubits)"
echo "    SP8-P4:  batch=d992be90... (ibm_fez, 8 qubits)"
echo "    SP10-P4: batch=78d22327... (ibm_fez, 10 qubits)"
echo ""
echo "Data locations:"
echo "  - grover/data/qpu/raw/"
echo "  - qft/data/qpu/raw/"
echo "=============================================="
