#!/bin/bash

# Pre-push verification script for qiskit-qward
# Run this script before pushing to ensure CI will pass

set -e  # Exit on any error

echo "🚀 Running pre-push verification for qiskit-qward..."
echo "=================================================="

echo ""
echo "1️⃣  Checking code formatting with Black..."
uv run black --check .
echo "✅ Black formatting check passed"

echo ""
echo "2️⃣  Running Pylint on core library and tests..."
uv run pylint -rn --disable=C,R --ignore-paths=qward/examples qward tests
echo "✅ Pylint check passed"

echo ""
echo "3️⃣  Running MyPy type checking..."
uv run mypy --exclude qward/examples qward
echo "✅ MyPy type checking passed"

echo ""
echo "4️⃣  Running test suite..."
uv run -m pytest tests/ -v
echo "✅ All tests passed"

echo ""
echo "🎉 All verification checks passed!"
echo "✅ Your code is ready to push"
echo ""
echo "Optional: Run 'tox -e lint' if you have tox installed for additional checks" 
