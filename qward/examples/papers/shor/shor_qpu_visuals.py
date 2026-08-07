"""Plot helpers for Shor QPU / simulator results (thesis figures)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def load_latest_json(raw_dir: Optional[Path] = None) -> Dict:
    raw_dir = raw_dir or Path(__file__).parent / "data" / "qpu" / "raw"
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON results in {raw_dir}")
    return json.loads(files[-1].read_text())


def plot_counts_histogram(counts: Dict[str, int], title: str = "Shor counts", out: Optional[Path] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for visuals") from exc

    items = sorted(counts.items(), key=lambda kv: -kv[1])[:32]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(labels)), values, color="#0f766e")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(title)
    ax.set_ylabel("counts")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def main():
    print("shor_qpu_visuals: load_latest_json() / plot_counts_histogram(counts)")


if __name__ == "__main__":
    main()
