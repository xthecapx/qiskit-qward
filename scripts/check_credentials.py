"""Reject hardcoded token/API-key assignments without printing secret values.

This focused check complements GitHub secret scanning; it is not a substitute
for provider-side revocation or a comprehensive secret scanner.
"""

import re
import subprocess
from pathlib import Path

PATTERNS = (
    re.compile(
        rb"(?i)(?:api[_-]?key|token|ibm[_-]?cloud[_-]?key)[^\r\n]{0,30}?[\x22\x27]([A-Za-z0-9_\-]{40,128})[\x22\x27]"
    ),
    re.compile(
        rb"(?i)(?:api[_-]?key|token)[^\r\n]{0,30}?\\[\x22\x27]([A-Za-z0-9_\-]{40,128})\\[\x22\x27]"
    ),
)


def findings(data):
    """Return suspicious line numbers, never credential text."""
    lines = set()
    for pattern in PATTERNS:
        for match in pattern.finditer(data):
            value = match.group(1)
            if len(set(value)) < 12 or any(
                word in value.lower() for word in (b"your_", b"example", b"placeholder")
            ):
                continue
            lines.add(data[: match.start()].count(b"\n") + 1)
    return sorted(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    failed = False
    for name in files:
        if not name:
            continue
        path = root / name.decode("utf-8", errors="surrogateescape")
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".ipynb",
            ".json",
            ".yml",
            ".yaml",
            ".sh",
            ".md",
            ".txt",
        }:
            continue
        for line in findings(path.read_bytes()):
            print(
                f"{path.relative_to(root)}:{line}: possible hardcoded credential (value withheld)"
            )
            failed = True
    if not failed:
        print("No hardcoded token/API-key assignments found in tracked text files")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
