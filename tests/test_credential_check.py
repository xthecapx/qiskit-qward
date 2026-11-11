"""Focused credential guard: recognize code/notebook values without exposing them."""

import importlib.util
from pathlib import Path
import string

spec = importlib.util.spec_from_file_location(
    "credential_check", Path(__file__).resolve().parents[1] / "scripts/check_credentials.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_plain_and_notebook_assignments_are_detected():
    candidate = string.ascii_letters[:44].encode()
    assert check.findings(b'TOKEN = "' + candidate + b'"') == [1]
    assert check.findings(b'"TOKEN = \\"' + candidate + b'\\""') == [1]


def test_environment_access_is_not_a_secret():
    assert check.findings(b'TOKEN = os.environ["IBM_QUANTUM_TOKEN"]') == []
