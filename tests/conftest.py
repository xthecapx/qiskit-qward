"""Pytest configuration for QWARD tests."""

import os
import sys
from pathlib import Path

# Set matplotlib backend BEFORE any imports that might use it
# This MUST be done before matplotlib.pyplot is imported anywhere
os.environ["MPLBACKEND"] = "Agg"

# Also set it programmatically as a fallback
import matplotlib  # pylint: disable=wrong-import-position

matplotlib.use("Agg", force=True)


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.

    This ensures matplotlib backend is set before any test modules are imported.
    Critical for headless CI/CD environments (especially Windows).
    """
    # Set environment variable (highest priority for matplotlib)
    os.environ["MPLBACKEND"] = "Agg"
    matplotlib.use("Agg", force=True)


import pytest


@pytest.fixture
def bash():
    """Use Git Bash on Windows so the driver sees the same files as native Python."""
    if os.name == "nt":
        git_bash = Path(os.environ["ProgramFiles"]) / "Git/bin/bash.exe"
        if git_bash.is_file():
            return str(git_bash)
    return "bash"
