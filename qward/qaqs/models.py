"""Loading the frozen QAQS models from their JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data" / "models"

#: Quality targets and what they measure.
TARGETS: Dict[str, str] = {
    "success": "Success rate",
    "hellinger": "Hellinger fidelity",
    "tvd": "Total variation distance",
    "dsr": "Differential success rate (DSR)",
}
#: Chips with a quality model.
CHIPS: Tuple[str, ...] = ("ibm_fez", "ibm_kingston")
#: Targets for which lower is better.
LOWER_IS_BETTER = frozenset({"tvd"})
#: Targets that did not reach the thesis threshold (R² >= 0.80) on new circuits.
UNRELIABLE_TARGETS = frozenset({"dsr"})
H1_R2_THRESHOLD = 0.80


@dataclass(frozen=True)
class ModelCard:
    """What is known about a frozen model: how well it did and what it was trained on."""

    id: str
    chip: str
    target: str
    model_name: str
    features: Tuple[str, ...]
    trained_on_jobs: int
    unseen_circuit_r2: Optional[float]
    unseen_circuit_mae: Optional[float]
    same_chip_new_circuits: Optional[Mapping[str, Any]]
    unseen_chip_new_circuits: Optional[Mapping[str, Any]]
    training_range: Mapping[str, Mapping[str, Optional[float]]]
    provenance: Mapping[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=None)
def load_model(model_id: str) -> Dict[str, Any]:
    """Raw JSON of one model, e.g. ``"ibm_fez/success"`` or ``"cost/billed_seconds"``."""
    path = DATA_DIR / (model_id.replace("/", "_") + ".json")
    if not path.is_file():
        raise KeyError(f"Unknown QAQS model: {model_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_models() -> List[str]:
    """Identifiers of every bundled model."""
    return sorted(
        p.stem.replace("_", "/", 1) if p.stem.startswith("cost") else _quality_id(p.stem)
        for p in DATA_DIR.glob("*.json")
    )


def _quality_id(stem: str) -> str:
    chip, target = stem.rsplit("_", 1)
    return f"{chip}/{target}"


def model_card(model_id: str) -> ModelCard:
    """The :class:`ModelCard` of a quality model."""
    raw = load_model(model_id)
    card = raw["card"]
    prospective = card.get("prospectiveValidation", {})
    return ModelCard(
        id=raw["id"],
        chip=raw["chip"],
        target=raw["target"],
        model_name=raw.get("modelName", raw["kind"]),
        features=tuple(raw["features"]),
        trained_on_jobs=card["trainedOnJobs"],
        unseen_circuit_r2=card["unseenCircuitValidation"].get("r2"),
        unseen_circuit_mae=card["unseenCircuitValidation"].get("mae"),
        same_chip_new_circuits=prospective.get("sameChip"),
        unseen_chip_new_circuits=prospective.get("unseenChip"),
        training_range=card["trainingRange"],
        provenance=card.get("provenance", {}),
    )
