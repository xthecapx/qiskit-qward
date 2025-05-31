# Developer Guide

This guide provides essential information for developers contributing to QWARD, including setup instructions, code quality standards, and verification procedures.

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- Virtual environment (recommended)

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/your-org/qiskit-qward.git
cd qiskit-qward

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Code Quality Standards

QWARD maintains strict code quality standards:

- **Code Style**: Black formatting (automatic)
- **Linting**: Pylint score of 10.00/10 required
- **Type Safety**: Full type hints with mypy validation
- **Testing**: Comprehensive test coverage
- **Documentation**: Complete docstrings and examples

## Pre-Push Verification

Before pushing changes to ensure CI passes, run these commands in order:

### Quick Start: Use the Verification Script
```bash
# Run the automated verification script
./verify.sh
```
This script runs all essential checks automatically and provides clear feedback.

### Manual Verification Commands

#### 1. Code Formatting
```bash
# Check code formatting (should show no changes needed)
black --check .

# If formatting is needed, run:
black .
```

#### 2. Linting
```bash
# Run pylint on core library and tests (should get 10.00/10)
pylint -rn qward tests
```

#### 3. Type Checking
```bash
# Run mypy type checking (should show no errors)
mypy qward tests
```

#### 4. Unit Tests
```bash
# Run the test suite
python -m pytest tests/ -v

# Or run specific test file
python -m pytest tests/test_validator.py -v
```

#### 5. All-in-One Verification
```bash
# Run all essential checks at once
black --check . && \
pylint -rn qward tests && \
mypy qward tests && \
python -m pytest tests/ -v
```

#### 6. Full Tox Suite (if tox is available)
```bash
# Run the complete lint environment
tox -e lint

# Run tests with coverage
tox -e coverage

# Run tests for specific Python version
tox -e py310  # or py311, py312
```

#### 7. Optional: Notebook Linting (if nbqa is installed)
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

## Configuration Notes

- Examples folder is excluded from linting (configured in `.pylintrc` and `mypy.ini`)
- Core library maintains strict code quality standards (10.00/10 pylint score)
- Type hints are enforced for all public APIs
- If any command fails, fix the issues before pushing

## Troubleshooting

### Black formatting issues:
```bash
# Fix formatting automatically
black .
```

### Pylint score below 10.00:
- Check the pylint output for specific issues
- Fix code quality issues or add appropriate disable comments for false positives
- Ensure examples folder is excluded (should be automatic)

### MyPy type errors:
- Add missing type hints
- Fix type mismatches
- Use `# type: ignore` for legitimate cases where mypy is wrong

### Test failures:
```bash
# Run tests with more verbose output
python -m pytest tests/ -v -s

# Run specific test
python -m pytest tests/test_validator.py::TestScanner::test_scanner_init -v
```

### Missing dependencies:
```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install specific tools
pip install black pylint mypy pytest
```

## Verification Summary

The verification process ensures:
- ✅ **Code Quality**: 10.00/10 pylint score maintained
- ✅ **Type Safety**: All type hints validated with mypy  
- ✅ **Code Style**: Consistent formatting with black
- ✅ **Functionality**: All tests pass
- ✅ **CI Compatibility**: Same checks as CI pipeline

## Recommended Workflow

1. Make your changes
2. Run `./verify.sh` (or manual commands)
3. Fix any issues reported
4. Commit and push with confidence

## Contributing Guidelines

For detailed contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/qiskit-qward/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/qiskit-qward/discussions)
- **Documentation**: [Technical Documentation](technical_docs.md) 