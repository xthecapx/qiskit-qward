"""Pytest configuration for QWARD tests."""

import os
import sys

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
