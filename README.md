![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-informational)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%201.0.0-6133BD)](https://github.com/Qiskit/qiskit)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)

# Qiskit Qward

Qiskit Qward is a framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs). It helps developers and researchers understand how their quantum algorithms perform on real hardware, providing insights into QPU behavior and code quality metrics.

Qward provides tools to execute quantum circuits on QPUs, collect comprehensive execution metrics, analyze circuit performance, validate algorithm correctness, generate insights about QPU behavior, and compare results across different backends.

### Table of Contents

##### For Users

1.  [About the Project](docs/project_overview.md)
2.  [Beginner's Guide](docs/beginners_guide.md)
3.  [Installation](INSTALL.md)
4.  [Quickstart Guide](docs/quickstart_guide.md)
5.  [Tutorials](docs/tutorials/index.rst)
6.  [How-Tos](docs/how_tos/index.rst)
7.  [How to Give Feedback](#how-to-give-feedback)
8.  [Contribution Guidelines](#contribution-guidelines)
9.  [License](#license)

##### For Developers/Contributors

1. [Contribution Guide](CONTRIBUTING.md)
2. [Technical Docs](docs/technical_docs.md)
3. [Pre-Push Verification](#pre-push-verification)


----------------------------------------------------------------------------------------------------

### How to Give Feedback

We encourage your feedback! You can share your thoughts with us by:
- Opening an issue in the repository

----------------------------------------------------------------------------------------------------

### Contribution Guidelines

For information on how to contribute to this project, please take a look at our [contribution guidelines](CONTRIBUTING.md).

----------------------------------------------------------------------------------------------------

### License
[Apache License 2.0](LICENSE.txt)

## Pre-Push Verification

Before pushing changes to ensure CI passes, run these commands in order:

### Quick Start: Use the Verification Script
```bash
# Run the automated verification script
./verify.sh
```
This script runs all essential checks automatically and provides clear feedback.

### Manual Verification Commands

### 1. Code Formatting
```bash
# Check code formatting (should show no changes needed)
black --check .

# If formatting is needed, run:
black .
```

### 2. Linting
```bash
# Run pylint on core library and tests (should get 10.00/10)
pylint -rn qward tests
```

### 3. Type Checking
```bash
# Run mypy type checking (should show no errors)
mypy qward tests
```

### 4. Unit Tests
```bash
# Run the test suite
python -m pytest tests/ -v

# Or run specific test file
python -m pytest tests/test_validator.py -v
```

### 5. All-in-One Verification
```bash
# Run all essential checks at once
black --check . && \
pylint -rn qward tests && \
mypy qward tests && \
python -m pytest tests/ -v
```

### 6. Full Tox Suite (if tox is available)
```bash
# Run the complete lint environment
tox -e lint

# Run tests with coverage
tox -e coverage

# Run tests for specific Python version
tox -e py310  # or py311, py312
```

### 7. Optional: Notebook Linting (if nbqa is installed)
```bash
# Check for any notebooks in docs (usually none)
nbqa pylint -rn docs/
```

### Expected Results
- **Black**: `All done! ✨ 🍰 ✨ X files would be left unchanged.`
- **Pylint**: `Your code has been rated at 10.00/10`
- **MyPy**: `Success: no issues found in X source files`
- **Tests**: All tests should pass with `PASSED` status

### Quick Verification (Minimum Required)
```bash
# Essential checks only (fastest)
black --check . && pylint -rn qward tests && mypy qward tests
```

### Notes
- Examples folder is excluded from linting (configured in `.pylintrc` and `mypy.ini`)
- Core library maintains strict code quality standards (10.00/10 pylint score)
- Type hints are enforced for all public APIs
- If any command fails, fix the issues before pushing

### Troubleshooting
**Black formatting issues:**
```bash
# Fix formatting automatically
black .
```

**Pylint score below 10.00:**
- Check the pylint output for specific issues
- Fix code quality issues or add appropriate disable comments for false positives
- Ensure examples folder is excluded (should be automatic)

**MyPy type errors:**
- Add missing type hints
- Fix type mismatches
- Use `# type: ignore` for legitimate cases where mypy is wrong

**Test failures:**
```bash
# Run tests with more verbose output
python -m pytest tests/ -v -s

# Run specific test
python -m pytest tests/test_validator.py::TestScanner::test_scanner_init -v
```

**Missing dependencies:**
```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install specific tools
pip install black pylint mypy pytest
```

### Summary
The verification process ensures:
- ✅ **Code Quality**: 10.00/10 pylint score maintained
- ✅ **Type Safety**: All type hints validated with mypy  
- ✅ **Code Style**: Consistent formatting with black
- ✅ **Functionality**: All tests pass
- ✅ **CI Compatibility**: Same checks as CI pipeline

**Recommended workflow:**
1. Make your changes
2. Run `./verify.sh` (or manual commands)
3. Fix any issues reported
4. Commit and push with confidence

----------------------------------------------------------------------------------------------------
