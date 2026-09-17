"""Check or synchronize the manuscript files shared with the release artifact.

Default usage checks source files without writing. Pass --source artifact or
--source qward to choose which copy to retain when synchronizing. PDFs remain
in the artifact folder unless --include-pdf is explicitly requested.
"""

import argparse
import hashlib
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("artifact", "qward"))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--include-pdf", action="store_true")
    args = parser.parse_args()
    papers = Path(__file__).resolve().parent
    repo = papers.parents[2]
    artifact = args.artifact or repo.parent / "dsr-quantum-validation"
    release = artifact / "paper"
    pairs = [
        (papers / "main-springer.tex", release / "sn-article.tex"),
    ]
    if args.include_pdf:
        pairs.append((papers / "main-springer.pdf", release / "sn-article.pdf"))
    source_tex = pairs[0][1 if args.source == "artifact" else 0]
    text = source_tex.read_text()
    names = {"sn-jnl.cls", "sn-mathphys-num.bst"}
    for group in re.findall(r"\\bibliography\{([^}]+)\}", text):
        names.update(f"{name.strip()}.bib" for name in group.split(","))
    names.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a path inside the paper folder: {name}")
        pairs.append((papers / relative, release / relative))
    if args.source:
        directed = [(b, a) if args.source == "artifact" else (a, b) for a, b in pairs]
        missing = [str(a) for a, _ in directed if not a.is_file()]
        if missing:
            parser.error("Missing source files; nothing copied: " + ", ".join(missing))
        for source, target in directed:
            if target.exists() and source.read_bytes() == target.read_bytes():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    different = []
    for a, b in pairs:
        if not a.is_file() or not b.is_file():
            different.append(f"Missing: {a.name} / {b.name}")
        elif hashlib.sha256(a.read_bytes()).digest() != hashlib.sha256(b.read_bytes()).digest():
            different.append(f"Different: {a.name} / {b.name}")
    if different:
        print("\n".join(different))
        raise SystemExit(1)
    print(f"All {len(pairs)} shared manuscript files match.")
    print("Compile the PDF in the artifact folder after synchronizing source changes.")


if __name__ == "__main__":
    main()
