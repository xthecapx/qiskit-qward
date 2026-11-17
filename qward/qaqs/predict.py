"""QAQS inference: predict execution quality and cost from a circuit, before running it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

from qward.qaqs import _inference
from qward.qaqs.footprint import as_row, compute_footprint
from qward.qaqs.models import (
    CHIPS,
    H1_R2_THRESHOLD,
    LOWER_IS_BETTER,
    TARGETS,
    UNRELIABLE_TARGETS,
    load_model,
    model_card,
)

#: Shots the cost model was observed at. Other shot counts are an extrapolation.
VALIDATED_SHOTS = 1024
#: IBM's published rule of thumb, for comparison: ~2 s per job plus 350 µs per execution.
IBM_RULE_OVERHEAD_S = 2.0
IBM_RULE_PER_EXECUTION_S = 0.00035


@dataclass(frozen=True)
class QualityPrediction:
    """Predicted execution quality of one circuit on one chip for one target."""

    chip: str
    target: str
    value: float
    model: str
    #: R² of this model on circuits it never saw, in the thesis validation.
    validation_r2: Optional[float]
    #: Mean absolute error in the same validation.
    validation_mae: Optional[float]
    #: Verdict of the new-run test against the thesis threshold (R² >= 0.80).
    verdict: Optional[str]
    higher_is_better: bool
    warnings: Sequence[str] = field(default_factory=tuple)

    @property
    def reliable(self) -> bool:
        """False when the model missed the thesis threshold or a warning applies."""
        return not self.warnings and self.verdict == "pass"


@dataclass(frozen=True)
class CostPrediction:
    """Predicted billed QPU time of one job."""

    seconds: float
    resource_units: float
    billed_seconds: int
    shots: int
    ibm_rule_seconds: float
    mae_seconds: Optional[float]
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class QAQSResult:
    """Footprint and predictions of one circuit."""

    footprint: Mapping[str, Any]
    quality: Mapping[str, QualityPrediction]
    cost: Optional[CostPrediction]
    backend: str
    warnings: Sequence[str] = field(default_factory=tuple)

    def quality_table(self) -> List[Dict[str, Any]]:
        """Predictions as a list of dictionaries, ready for ``pandas.DataFrame``."""
        return [
            {
                "chip": q.chip,
                "target": q.target,
                "prediction": q.value,
                "validation_r2": q.validation_r2,
                "validation_mae": q.validation_mae,
                "verdict": q.verdict,
                "higher_is_better": q.higher_is_better,
                "warnings": "; ".join(q.warnings),
            }
            for q in self.quality.values()
        ]


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    return name() if callable(name) else str(name)


def _range_warnings(
    footprint: Mapping[str, Any], features: Iterable[str], training_range: Mapping[str, Any]
) -> List[str]:
    """Warnings for circuits larger than anything trained on, and for inputs outside their range."""
    notes = []
    for column, label in (
        ("b.two_qubit_gates", "two-qubit gates"),
        ("b.depth", "compiled depth"),
        ("a.num_qubits", "qubits"),
    ):
        bounds = training_range.get(column)
        value = footprint.get(column)
        if (
            bounds
            and bounds.get("max") is not None
            and isinstance(value, (int, float))
            and value > bounds["max"]
        ):
            notes.append(
                f"circuit has {value:g} {label}, more than the largest in training "
                f"({bounds['max']:g}); predictions for larger circuits were not reliable in validation"
            )
    outside = []
    for column in features:
        bounds = training_range.get(column)
        value = footprint.get(column)
        if not bounds or bounds.get("min") is None or not isinstance(value, (int, float, bool)):
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        low, high = bounds["min"], bounds["max"]
        if value < low - 1e-9 * (1 + abs(low)) or value > high + 1e-9 * (1 + abs(high)):
            outside.append(f"{column} = {value:.3g} (training {low:.3g} to {high:.3g})")
    if outside:
        shown = "; ".join(outside[:3]) + (
            f"; and {len(outside) - 3} more" if len(outside) > 3 else ""
        )
        notes.append(
            f"{len(outside)} model input(s) outside the range seen in training: {shown}. "
            "The model has not seen circuits like this and the prediction may be unreliable"
        )
    return notes


def _quality(
    chip: str, target: str, footprint: Mapping[str, Any], size_notes: Sequence[str]
) -> QualityPrediction:
    model = load_model(f"{chip}/{target}")
    card = model_card(f"{chip}/{target}")
    row = as_row(dict(footprint), list(model["features"]))
    value = float(np.clip(_inference.predict_quality(model, row)[0], *model["clip"]))
    notes: List[str] = list(size_notes)
    qs = [f for f in model["features"] if f.startswith("a.quantum_specific.")]
    if qs and footprint.get("a.quantum_specific_present", 0.0) != 1.0:
        notes.append(
            "quantum-specific metrics are missing (circuit above 8 qubits, dynamic, or PyTorch not installed); "
            "the model fills them with the training median"
        )
    new = card.same_chip_new_circuits or {}
    if target in UNRELIABLE_TARGETS:
        notes.append(
            "DSR predictions missed the thesis threshold (R² >= 0.80) on new circuits; treat as indicative"
        )
    return QualityPrediction(
        chip=chip,
        target=target,
        value=value,
        model=card.model_name,
        validation_r2=new.get("r2", card.unseen_circuit_r2),
        validation_mae=new.get("mae", card.unseen_circuit_mae),
        verdict=new.get("h1Verdict"),
        higher_is_better=target not in LOWER_IS_BETTER,
        warnings=tuple(notes),
    )


def predict_cost(
    footprint: Mapping[str, Any],
    *,
    shots: int = VALIDATED_SHOTS,
    twirling: bool = False,
    dynamical_decoupling: bool = False,
) -> CostPrediction:
    """Billed QPU time from a footprint. ``shots`` other than 1,024 are extrapolated."""
    model = load_model("cost/billed_seconds")
    dyn = bool(footprint.get("dynamic", False))
    duration = footprint.get("duration_s")
    values = {
        "shots": shots,
        "duration_s": float("nan") if duration is None else float(duration),
        "twirling": float(twirling),
        "dynamic": float(dyn),
        "dynamical_decoupling": float(dynamical_decoupling),
    }
    for column in model["features"]:
        values.setdefault(column, footprint.get(column, float("nan")))
    out = _inference.predict_cost(
        model,
        {k: np.asarray([float(v) if v is not None else float("nan")]) for k, v in values.items()},
    )
    seconds = float(out["billed_seconds"][0])
    notes = []
    if shots != VALIDATED_SHOTS:
        notes.append(
            f"cost was only measured at {VALIDATED_SHOTS} shots; {shots} shots follows IBM's usage formula "
            "with the fitted per-shot terms and is not validated"
        )
    if dyn:
        notes.append(
            "dynamic circuits are not scheduled; the scheduled duration is replaced by the training median"
        )
    card = model["card"]
    return CostPrediction(
        seconds=seconds,
        resource_units=seconds / model["secondsPerRu"],
        billed_seconds=int(math.ceil(seconds - 1e-9)),
        shots=shots,
        ibm_rule_seconds=IBM_RULE_OVERHEAD_S + IBM_RULE_PER_EXECUTION_S * shots,
        mae_seconds=card["newRunValidation"]["all"]["maeSeconds"],
        warnings=tuple(notes),
    )


def predict(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    chips: Sequence[str] = CHIPS,
    targets: Sequence[str] = tuple(TARGETS),
    cost: bool = True,
    shots: int = VALIDATED_SHOTS,
    twirling: bool = False,
    dynamical_decoupling: bool = False,
    quantum_specific_metrics: bool = True,
    add_barrier: bool = True,
) -> QAQSResult:
    """Predict execution quality and billed cost of ``circuit`` before running it.

    Args:
        circuit: the logical circuit (measurements included).
        backend: the backend to compile for; a real IBM backend or a fake one such as
            ``FakeFez()``. The footprint depends on the compiled circuit, so a real backend
            reproduces what will run; a fake backend is close but not identical.
        chips: which chip models to use, from ``("ibm_fez", "ibm_kingston")``.
        targets: quality targets, from ``("success", "hellinger", "tvd", "dsr")``.
        cost: also predict billed time.
        shots: shots for the cost estimate (validated only at 1,024).
        twirling / dynamical_decoupling: the runtime options of the job (cost only).
        quantum_specific_metrics: compute the PyTorch-based metrics when possible. Needs
            ``pip install "qiskit-qward[torch]"``; slow above 6 qubits.
        add_barrier: add one barrier before the final measurements when the circuit has none.
            Every training circuit had one and the footprint is sensitive to it; without it the
            predictions are usually out of range. Set to ``False`` to predict the circuit as given.

    Raises:
        ValueError: for unknown chips or targets, or if the circuit compiles to no gates.
    """
    unknown = [c for c in chips if c not in CHIPS] + [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(
            f"Unknown chips or targets: {unknown}; use chips {CHIPS} and targets {tuple(TARGETS)}"
        )
    footprint = compute_footprint(
        circuit,
        backend,
        quantum_specific_metrics=quantum_specific_metrics,
        add_barrier=add_barrier,
    )
    notes: List[str] = []
    if footprint.get("barrier_added"):
        notes.append(
            "no barrier found: added one before the final measurements, as in every training "
            "circuit (the footprint is sensitive to it; pass add_barrier=False to predict the "
            "circuit as given)"
        )
    name = _backend_name(backend)
    if name.startswith("fake"):
        notes.append(
            "compiled for a fake backend: two-qubit gate counts matched the real chips in validation, "
            "but compiled depth can differ by up to ~12%; use a real backend for the closest footprint"
        )
    quality: Dict[str, QualityPrediction] = {}
    for chip in chips:
        for target in targets:
            model = load_model(f"{chip}/{target}")
            size = _range_warnings(
                footprint, model["features"], model_card(f"{chip}/{target}").training_range
            )
            quality[f"{chip}/{target}"] = _quality(chip, target, footprint, size)
    cost_prediction = (
        predict_cost(
            footprint, shots=shots, twirling=twirling, dynamical_decoupling=dynamical_decoupling
        )
        if cost
        else None
    )
    return QAQSResult(
        footprint=footprint,
        quality=quality,
        cost=cost_prediction,
        backend=name,
        warnings=tuple(notes),
    )
