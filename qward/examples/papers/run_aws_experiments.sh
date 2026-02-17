#!/bin/bash
# =============================================================================
# AWS Braket QPU Experiment Runner
# =============================================================================
# Run this script to execute all missing experiments on AWS Braket hardware.
#
# Usage:
#   ./run_aws_experiments.sh [grover|qft|all]
#
# Environment Variables (optional):
#   AWS_BRAKET_DEVICE      - Device name (default: Ankaa-3)
#   AWS_BRAKET_REGION      - Region (default: us-west-1)
#   AWS_ACCESS_KEY_ID      - AWS access key
#   AWS_SECRET_ACCESS_KEY  - AWS secret key
#   AWS_NO_WAIT            - Set to 1 to submit jobs without waiting
#
# Examples:
#   ./run_aws_experiments.sh
#   AWS_BRAKET_DEVICE="Ankaa-3" AWS_BRAKET_REGION="us-west-1" ./run_aws_experiments.sh grover
#   AWS_NO_WAIT=1 ./run_aws_experiments.sh qft
# =============================================================================

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DEVICE="${AWS_BRAKET_DEVICE:-Ankaa-3}"
REGION="${AWS_BRAKET_REGION:-${AWS_REGION:-us-west-1}}"

COMMON_ARGS=(--device "$DEVICE" --region "$REGION")

if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    COMMON_ARGS+=(--aws-access-key-id "$AWS_ACCESS_KEY_ID")
fi
if [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    COMMON_ARGS+=(--aws-secret-access-key "$AWS_SECRET_ACCESS_KEY")
fi
if [ "${AWS_NO_WAIT:-0}" = "1" ]; then
    COMMON_ARGS+=(--no-wait)
fi

echo "=============================================="
echo "AWS BRAKET EXPERIMENT RUNNER"
echo "=============================================="
echo "Device: $DEVICE"
echo "Region: $REGION"
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Using AWS credentials from environment variables"
else
    echo "Using default AWS credential chain (.env/aws configure/role)"
fi
if [ "${AWS_NO_WAIT:-0}" = "1" ]; then
    echo "Mode: submit only (--no-wait)"
else
    echo "Mode: wait for results"
fi
echo ""

# Function to run a single experiment
run_experiment() {
    local algo=$1
    local config=$2
    local script=$3

    echo -e "${YELLOW}>>> Running $algo: $config${NC}"
    python "$script" --config "$config" "${COMMON_ARGS[@]}"
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

    # 3-qubit configs
    run_experiment "GROVER" "ASYM-1" "grover/grover_aws.py"
    run_experiment "GROVER" "ASYM-2" "grover/grover_aws.py"
    run_experiment "GROVER" "M3-2" "grover/grover_aws.py"
    run_experiment "GROVER" "SYM-1" "grover/grover_aws.py"
    run_experiment "GROVER" "SYM-2" "grover/grover_aws.py"
    run_experiment "GROVER" "H3-3" "grover/grover_aws.py"
    run_experiment "GROVER" "H3-2" "grover/grover_aws.py"
    run_experiment "GROVER" "S3-1" "grover/grover_aws.py"
    run_experiment "GROVER" "M3-1" "grover/grover_aws.py"

    # 4-qubit configs
    run_experiment "GROVER" "M4-4" "grover/grover_aws.py"
    run_experiment "GROVER" "M4-2" "grover/grover_aws.py"
    run_experiment "GROVER" "S4-1" "grover/grover_aws.py"
    run_experiment "GROVER" "H4-4" "grover/grover_aws.py"

    # Larger/deeper configs
    run_experiment "GROVER" "S5-1" "grover/grover_aws.py"
    run_experiment "GROVER" "S7-1" "grover/grover_aws.py"
    run_experiment "GROVER" "S8-1" "grover/grover_aws.py"
}

# =============================================================================
# QFT EXPERIMENTS
# =============================================================================
run_qft() {
    echo "=============================================="
    echo "QFT EXPERIMENTS"
    echo "=============================================="

    # Small roundtrip configs
    run_experiment "QFT" "SR2" "qft/qft_aws.py"
    run_experiment "QFT" "SR3" "qft/qft_aws.py"
    run_experiment "QFT" "SR4" "qft/qft_aws.py"
    run_experiment "QFT" "SR5" "qft/qft_aws.py"
    run_experiment "QFT" "SR6" "qft/qft_aws.py"

    # Period detection configs
    run_experiment "QFT" "PV4-P8" "qft/qft_aws.py"
    run_experiment "QFT" "SP4-P4" "qft/qft_aws.py"
    run_experiment "QFT" "PV4-P4" "qft/qft_aws.py"
    run_experiment "QFT" "PV6-P16" "qft/qft_aws.py"
    run_experiment "QFT" "SP5-P4" "qft/qft_aws.py"
    run_experiment "QFT" "PV6-P8" "qft/qft_aws.py"
    run_experiment "QFT" "SP6-P8" "qft/qft_aws.py"

    # Input variation configs
    run_experiment "QFT" "IV4-0000" "qft/qft_aws.py"
    run_experiment "QFT" "IV4-0101" "qft/qft_aws.py"

    # Larger configs
    run_experiment "QFT" "SR8" "qft/qft_aws.py"
    run_experiment "QFT" "SR10" "qft/qft_aws.py"
    run_experiment "QFT" "SP8-P4" "qft/qft_aws.py"
    run_experiment "QFT" "SP10-P4" "qft/qft_aws.py"
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
echo "  - grover/data/qpu/aws/"
echo "  - qft/data/qpu/aws/"
echo "=============================================="
