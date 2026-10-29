"""Small BV calibration-prediction campaign and analytic diagnostic controls."""

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit

from qward.algorithms import BernsteinVazirani
from qward.examples.papers.bv.bv_configs import get_config as get_bv_config

LAYOUTS = ("low_predicted_error", "contrasting_predicted_error")
WIDTHS = (2, 4)
# Simulator-only widths for testing how error models scale with output size.
SCALING_WIDTHS = (6, 8)
LAYERS = (0, 2, 8, 32)
PATTERNS = ("ONES", "ALT", "SINGLE")
ANGLES = (0.0, math.pi / 6, math.pi / 2, math.pi)


@dataclass(frozen=True)
class BVCalibrationConfig:
    """One identifiable treatment; execution window and repetition are run metadata."""

    config_id: str
    family: str
    num_qubits: int
    layout_label: str
    pattern: str = "ALT"
    forward_layers: int = 0
    padding_seed: int = 11
    angle: float = 0.0
    basis: str = "Z"
    mask: str = "zero"

    @property
    def bv_config(self):
        """Use the same secret definitions and bit order as existing BV campaigns."""
        return get_bv_config(f"BV{self.num_qubits}-{self.pattern}")

    @property
    def num_total_qubits(self) -> int:
        """Count BV's unmeasured ancilla only for BV families."""
        return self.num_qubits + int(self.family.startswith("bv_"))

    def to_dict(self) -> Dict:
        """Serialize treatment and distinguish measured and total widths."""
        return {
            **asdict(self),
            "num_measured_qubits": self.num_qubits,
            "num_total_qubits": self.num_total_qubits,
            "base_config_id": self.bv_config.config_id if self.family.startswith("bv_") else None,
            "secret_string": (
                self.bv_config.secret_string if self.family.startswith("bv_") else None
            ),
            "role": "primary" if self.family == "bv_padded" else "diagnostic_control",
            "ideal_histogram_method": "analytic",
        }


def _configs() -> List[BVCalibrationConfig]:
    result = []
    for width, layout in product(WIDTHS, LAYOUTS):
        prefix = f"BVCP{width}-{layout}"
        for pattern, layers in product(PATTERNS, LAYERS):
            result.append(
                BVCalibrationConfig(
                    f"{prefix}-{pattern}-L{layers:03d}",
                    "bv_padded",
                    width,
                    layout,
                    pattern=pattern,
                    forward_layers=layers,
                )
            )
        for mask in ("zero", "one", "alternating_one", "alternating_zero"):
            result.append(
                BVCalibrationConfig(f"{prefix}-SPAM-{mask}", "spam", width, layout, mask=mask)
            )
        for phase, basis in product((0.0, math.pi), ("Z", "X")):
            result.append(
                BVCalibrationConfig(
                    f"{prefix}-GHZ-{basis}-P{int(phase > 0)}",
                    "ghz_phase",
                    width,
                    layout,
                    angle=phase,
                    basis=basis,
                )
            )
        result.append(BVCalibrationConfig(f"{prefix}-UNIFORM", "uniform", width, layout))
        for index, angle in enumerate(ANGLES):
            result.append(
                BVCalibrationConfig(
                    f"{prefix}-BVFAULT-A{index}",
                    "bv_target_deviation",
                    width,
                    layout,
                    angle=angle,
                )
            )
    return result


def _scaling_configs() -> List[BVCalibrationConfig]:
    return [
        BVCalibrationConfig(
            f"BVCP{width}-{layout}-{pattern}-L{layers:03d}",
            "bv_padded",
            width,
            layout,
            pattern=pattern,
            forward_layers=layers,
        )
        for width, layout, pattern, layers in product(SCALING_WIDTHS, LAYOUTS, PATTERNS, LAYERS)
    ]


CONFIGS = {config.config_id: config for config in _configs()}
# Primary BV only; kept apart so the IBM campaign matrix and budgets are unchanged.
SCALING_CONFIGS = {config.config_id: config for config in _scaling_configs()}


def get_config(config_id: str) -> BVCalibrationConfig:
    """Retrieve a campaign treatment using the usual experiment configuration ID."""
    try:
        return CONFIGS.get(config_id) or SCALING_CONFIGS[config_id]
    except KeyError as exc:
        raise ValueError(f"Unknown BV calibration config: {config_id}") from exc


def _padding(width: int, layers: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(width)
    for layer in range(layers):
        for qubit in range(width):
            (circuit.h if rng.integers(2) == 0 else circuit.s)(qubit)
        for left in range(layer % 2, width - 1, 2):
            circuit.cx(left, left + 1)
        circuit.barrier()
    return circuit


def build_circuit(config: BVCalibrationConfig) -> QuantumCircuit:
    """Adapt the library BV circuit at validated stages; never execute it."""
    width = config.num_qubits
    if config.family.startswith("bv_"):
        base = BernsteinVazirani(config.bv_config.secret_string, use_barriers=True).circuit
        if config.family == "bv_padded" and config.forward_layers == 0:
            return base.copy()
        barriers = [i for i, item in enumerate(base.data) if item.operation.name == "barrier"]
        if len(barriers) != 2:
            raise ValueError("BV builder stage boundaries changed; review the treatment adapter")
        boundary = barriers[-1]
        tail = [item.operation.name for item in base.data[boundary + 1 :]]
        if tail != ["h"] * width + ["measure"] * width:
            raise ValueError("BV final-H/measurement stage changed; review the adapter")
        circuit = base.copy_empty_like()
        padding = _padding(width, config.forward_layers, config.padding_seed)
        inserted_fault = False
        for index, item in enumerate(base.data):
            if item.operation.name == "measure" and not inserted_fault:
                if config.family == "bv_target_deviation":
                    circuit.barrier()
                    circuit.ry(config.angle, 0)
                    circuit.barrier()
                inserted_fault = True
            circuit.append(item.operation, item.qubits, item.clbits)
            if index == boundary and config.family == "bv_padded":
                circuit.compose(padding, qubits=range(width), inplace=True)
                circuit.barrier()
                circuit.compose(padding.inverse(), qubits=range(width), inplace=True)
                circuit.barrier()
        return circuit
    circuit = QuantumCircuit(width, width)
    if config.family == "spam":
        for index in range(width):
            if (
                config.mask == "one"
                or (config.mask == "alternating_one" and index % 2 == 0)
                or (config.mask == "alternating_zero" and index % 2 == 1)
            ):
                circuit.x(index)
    elif config.family == "uniform":
        circuit.h(range(width))
    elif config.family == "ghz_phase":
        circuit.h(0)
        for index in range(width - 1):
            circuit.cx(index, index + 1)
        circuit.rz(config.angle, 0)
        if config.basis == "X":
            circuit.h(range(width))
    else:
        raise ValueError(f"Unsupported circuit family: {config.family}")
    circuit.measure(range(width), range(width))
    return circuit


def analytic_references(config: BVCalibrationConfig) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return nominal task and deliberately injected circuit references, without simulation."""
    width = config.num_qubits
    if config.family.startswith("bv_"):
        target = config.bv_config.expected_outcome
        nominal = {target: 1.0}
        if config.family == "bv_target_deviation":
            alternate = target[:-1] + ("1" if target[-1] == "0" else "0")
            injected = {
                target: math.cos(config.angle / 2) ** 2,
                alternate: math.sin(config.angle / 2) ** 2,
            }
            return nominal, {k: p for k, p in injected.items() if p > 1e-14}
        return nominal, dict(nominal)
    if config.family == "spam":
        bits = [
            int(
                config.mask == "one"
                or (config.mask == "alternating_one" and i % 2 == 0)
                or (config.mask == "alternating_zero" and i % 2 == 1)
            )
            for i in range(width)
        ]
        reference = {"".join(str(bit) for bit in reversed(bits)): 1.0}
    elif config.family == "uniform":
        reference = {f"{x:0{width}b}": 1 / 2**width for x in range(2**width)}
    elif config.family == "ghz_phase":
        if config.basis == "Z":
            reference = {"0" * width: 0.5, "1" * width: 0.5}
        else:
            nominal = {
                f"{x:0{width}b}": 2 / 2**width for x in range(2**width) if x.bit_count() % 2 == 0
            }
            injected = {
                f"{x:0{width}b}": (1 + (-1) ** x.bit_count() * math.cos(config.angle)) / 2**width
                for x in range(2**width)
            }
            return nominal, {k: p for k, p in injected.items() if p > 1e-14}
    else:
        raise ValueError(config.family)
    return reference, dict(reference)
