"""Small BV-derived signal-plus-background proof experiment.

The experiment prepares two analytically known BV-derived target peaks and a
separate broad, non-Clifford background branch. The target set is sufficient
for DSR, while full Hellinger fidelity and TVD fidelity use the complete ideal
distribution obtained from a statevector at the small proof sizes.

This module is intentionally limited to the pre-IBM proof. It does not submit
jobs to quantum hardware.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit.quantum_info import Statevector

from qward.metrics.differential_success_rate import DSRProfiler
from qward.utils.styles import COLORBREWER_PALETTE

PROOF_TOTAL_QUBITS = (6, 8, 10)
CAMPAIGN_TOTAL_QUBITS = (27, 28, 29)
SUPPORTED_TOTAL_QUBITS = PROOF_TOTAL_QUBITS + CAMPAIGN_TOTAL_QUBITS
DEFAULT_TARGET_MASS = 0.9
DEFAULT_SEED = 20260721
DEFAULT_SHOTS = 16_384
PROOF_ONE_QUBIT_ERROR = 0.002
PROOF_TWO_QUBIT_ERROR = 0.02
PROOF_READOUT_ERROR = 0.015

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = EXAMPLES_DIR / "img" / "bv_signal_background"
DEFAULT_RESULTS_PATH = EXAMPLES_DIR / "bv_signal_background_proof_results.json"


@dataclass(frozen=True)
class SignalBackgroundSpec:
    """Configuration for one proof circuit."""

    num_total_qubits: int
    num_data_qubits: int
    num_measured_qubits: int
    target_mass: float
    seed: int
    secret_string: str
    flip_qubits: tuple[int, int]
    targets: tuple[str, str]

    @property
    def background_mass(self) -> float:
        """Return the ideal probability assigned to the background branch."""
        return 1.0 - self.target_mass


def _alternating_secret(num_data_qubits: int) -> str:
    return "".join("1" if index % 2 == 0 else "0" for index in range(num_data_qubits))


def _flip_bits(value: str, positions: tuple[int, int]) -> str:
    bits = list(value)
    for position in positions:
        bits[position] = "0" if bits[position] == "1" else "1"
    return "".join(bits)


def _signal_targets(
    secret_string: str,
    flip_qubits: tuple[int, int],
) -> tuple[str, str]:
    """Return full measured bitstrings in Qiskit's displayed bit order.

    The two leading bits are ``branch`` and ``selector``. Data bits follow in
    reverse qubit order, matching Qiskit's classical-count representation.
    """
    base_data = secret_string[::-1]
    flipped_secret = _flip_bits(secret_string, flip_qubits)
    flipped_data = flipped_secret[::-1]
    return f"00{base_data}", f"01{flipped_data}"


def make_spec(
    num_total_qubits: int,
    *,
    target_mass: float = DEFAULT_TARGET_MASS,
    seed: int = DEFAULT_SEED,
) -> SignalBackgroundSpec:
    """Create a validated proof configuration.

    Registers consist of data qubits, one selector, one branch, and one
    unmeasured BV phase-kickback ancilla.
    """
    if num_total_qubits not in SUPPORTED_TOTAL_QUBITS:
        raise ValueError(
            f"num_total_qubits must be one of {SUPPORTED_TOTAL_QUBITS}, " f"got {num_total_qubits}"
        )
    if not 0.0 < target_mass < 1.0:
        raise ValueError(f"target_mass must be between 0 and 1, got {target_mass}")

    num_data_qubits = num_total_qubits - 3
    num_measured_qubits = num_total_qubits - 1
    secret_string = _alternating_secret(num_data_qubits)
    flip_qubits = (0, num_data_qubits - 1)
    targets = _signal_targets(secret_string, flip_qubits)
    return SignalBackgroundSpec(
        num_total_qubits=num_total_qubits,
        num_data_qubits=num_data_qubits,
        num_measured_qubits=num_measured_qubits,
        target_mass=target_mass,
        seed=seed,
        secret_string=secret_string,
        flip_qubits=flip_qubits,
        targets=targets,
    )


def build_signal_background_circuit(spec: SignalBackgroundSpec) -> QuantumCircuit:
    """Build a scalable BV-derived signal-plus-background circuit.

    The branch qubit assigns ``target_mass`` to the signal sector. Standard BV
    decoding creates the alternating secret, while the selector coherently
    produces a second target differing in two data bits. Gates controlled by
    the background branch create a seeded, broad, non-Clifford distribution
    without changing either signal target in the ideal circuit.
    """
    num_data = spec.num_data_qubits
    selector = num_data
    branch = num_data + 1
    ancilla = num_data + 2
    circuit = QuantumCircuit(spec.num_total_qubits, spec.num_measured_qubits)

    branch_angle = 2.0 * math.acos(math.sqrt(spec.target_mass))
    circuit.ry(branch_angle, branch)
    circuit.h(selector)

    circuit.x(ancilla)
    circuit.h(ancilla)
    circuit.h(range(num_data))
    circuit.barrier(label="BV oracle")
    for index, bit in enumerate(spec.secret_string):
        if bit == "1":
            circuit.cx(index, ancilla)
    circuit.h(range(num_data))

    circuit.barrier(label="Second target")
    for qubit in spec.flip_qubits:
        circuit.cx(selector, qubit)

    circuit.barrier(label="Background")
    rng = np.random.default_rng(spec.seed + spec.num_total_qubits)
    first_rotations = rng.uniform(0.30, 1.10, size=num_data)
    phases = rng.uniform(-0.85, 0.85, size=num_data)
    second_rotations = rng.uniform(-0.55, 0.55, size=num_data)

    for qubit, angle in enumerate(first_rotations):
        circuit.cry(float(angle), branch, qubit)
    for qubit, angle in enumerate(phases):
        circuit.rz(float(angle), qubit)
    for qubit in range(0, num_data - 1, 2):
        circuit.cz(qubit, qubit + 1)
    for qubit in range(1, num_data - 1, 2):
        circuit.cz(qubit, qubit + 1)
    for qubit, angle in enumerate(second_rotations):
        circuit.cry(float(angle), branch, qubit)

    circuit.barrier(label="Measurement")
    circuit.measure(range(num_data), range(num_data))
    circuit.measure(selector, num_data)
    circuit.measure(branch, num_data + 1)
    return circuit


def build_dynamic_signal_background_circuit(
    spec: SignalBackgroundSpec,
    *,
    include_background: bool = True,
) -> QuantumCircuit:
    """Build the IBM campaign circuit with a dynamic background branch.

    The branch is measured before the background block. Signal shots skip that
    block physically, avoiding the routed fanout of coherent controlled gates.
    The resulting ideal measurement distribution is equivalent to the
    coherently controlled proof circuit.
    """
    num_data = spec.num_data_qubits
    selector = num_data
    branch = num_data + 1
    ancilla = num_data + 2
    branch_clbit = num_data + 1
    circuit = QuantumCircuit(spec.num_total_qubits, spec.num_measured_qubits)

    branch_angle = 2.0 * math.acos(math.sqrt(spec.target_mass))
    circuit.ry(branch_angle, branch)
    circuit.h(selector)

    circuit.x(ancilla)
    circuit.h(ancilla)
    circuit.h(range(num_data))
    circuit.barrier(label="BV oracle")
    for index, bit in enumerate(spec.secret_string):
        if bit == "1":
            circuit.cx(index, ancilla)
    circuit.h(range(num_data))

    circuit.barrier(label="Second target")
    for qubit in spec.flip_qubits:
        circuit.cx(selector, qubit)

    circuit.measure(branch, branch_clbit)
    rng = np.random.default_rng(spec.seed + spec.num_total_qubits)
    first_rotations = rng.uniform(0.30, 1.10, size=num_data)
    phases = rng.uniform(-0.85, 0.85, size=num_data)
    second_rotations = rng.uniform(-0.55, 0.55, size=num_data)
    if include_background:
        with circuit.if_test((circuit.clbits[branch_clbit], True)):
            for qubit, angle in enumerate(first_rotations):
                circuit.ry(float(angle), qubit)
            for qubit, angle in enumerate(phases):
                circuit.rz(float(angle), qubit)
            for qubit in range(0, num_data - 1, 2):
                circuit.cz(qubit, qubit + 1)
            for qubit in range(1, num_data - 1, 2):
                circuit.cz(qubit, qubit + 1)
            for qubit, angle in enumerate(second_rotations):
                circuit.ry(float(angle), qubit)

    circuit.barrier(label="Measurement")
    circuit.measure(range(num_data), range(num_data))
    circuit.measure(selector, num_data)
    return circuit


def ideal_distribution(spec: SignalBackgroundSpec) -> Dict[str, float]:
    """Return the exact measured distribution for a small proof circuit."""
    circuit = build_signal_background_circuit(spec)
    bare = circuit.remove_final_measurements(inplace=False)
    statevector = Statevector.from_instruction(bare)
    measured_qubits = list(range(spec.num_measured_qubits))
    probabilities = statevector.probabilities_dict(qargs=measured_qubits)
    return {
        str(outcome): float(probability)
        for outcome, probability in probabilities.items()
        if probability > 1e-14
    }


def probabilities_to_counts(
    probabilities: Mapping[str, float],
    shots: int = DEFAULT_SHOTS,
) -> Dict[str, int]:
    """Convert exact probabilities to deterministic integer pseudo-counts."""
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    if not probabilities:
        raise ValueError("probabilities must not be empty")

    scaled = {outcome: float(probability) * shots for outcome, probability in probabilities.items()}
    counts = {outcome: int(math.floor(value)) for outcome, value in scaled.items()}
    remainder = shots - sum(counts.values())
    ranked = sorted(
        scaled,
        key=lambda outcome: scaled[outcome] - counts[outcome],
        reverse=True,
    )
    for outcome in ranked[:remainder]:
        counts[outcome] += 1
    return {outcome: count for outcome, count in counts.items() if count > 0}


def build_proof_noise_model() -> NoiseModel:
    """Create a deterministic generic noise model for pre-IBM validation."""
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(PROOF_ONE_QUBIT_ERROR, 1),
        ["h", "x", "ry"],
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(PROOF_TWO_QUBIT_ERROR, 2),
        ["cx", "cry", "cz"],
    )
    readout_error = ReadoutError(
        [
            [1.0 - PROOF_READOUT_ERROR, PROOF_READOUT_ERROR],
            [PROOF_READOUT_ERROR, 1.0 - PROOF_READOUT_ERROR],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def noisy_counts(
    spec: SignalBackgroundSpec,
    shots: int = DEFAULT_SHOTS,
) -> Dict[str, int]:
    """Sample the proof circuit with a generic pre-IBM noise model."""
    circuit = build_dynamic_signal_background_circuit(spec)
    simulator = AerSimulator(noise_model=build_proof_noise_model())
    compiled = transpile(
        circuit,
        simulator,
        optimization_level=1,
        seed_transpiler=spec.seed,
    )
    result = simulator.run(
        compiled,
        shots=shots,
        seed_simulator=spec.seed,
    ).result()
    return {
        str(outcome).replace(" ", ""): int(count) for outcome, count in result.get_counts().items()
    }


def heavy_hex_transpiled_complexity(
    spec: SignalBackgroundSpec,
    *,
    dynamic: bool = False,
    signal_path_only: bool = False,
) -> Dict[str, int]:
    """Estimate routing cost on a heavy-hex coupling map."""
    circuit = (
        build_dynamic_signal_background_circuit(
            spec,
            include_background=not signal_path_only,
        )
        if dynamic
        else build_signal_background_circuit(spec)
    )
    heavy_hex_distance = 3 if spec.num_total_qubits <= 19 else 5
    coupling_map = CouplingMap.from_heavy_hex(heavy_hex_distance)
    compiled = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "ecr"],
        coupling_map=coupling_map,
        optimization_level=3,
        seed_transpiler=spec.seed,
    )

    def iter_instructions(circuit_to_walk):
        for instruction in circuit_to_walk.data:
            blocks = getattr(instruction.operation, "blocks", ())
            if blocks:
                for block in blocks:
                    yield from iter_instructions(block)
            else:
                yield instruction

    def expanded_depth(circuit_to_walk):
        depth = circuit_to_walk.depth()
        for instruction in circuit_to_walk.data:
            blocks = getattr(instruction.operation, "blocks", ())
            if blocks:
                depth += max(expanded_depth(block) for block in blocks) - 1
        return depth

    instructions = list(iter_instructions(compiled))
    two_qubit_gates = sum(1 for instruction in instructions if len(instruction.qubits) == 2)
    three_qubit_gates = sum(1 for instruction in instructions if len(instruction.qubits) == 3)
    return {
        "coupling_map_qubits": coupling_map.size(),
        "depth": expanded_depth(compiled),
        "size": len(instructions),
        "two_qubit_gates": two_qubit_gates,
        "three_qubit_gates": three_qubit_gates,
        "dynamic_circuit": dynamic,
        "signal_path_only": signal_path_only,
    }


def dsr_profile(
    counts: Mapping[str, int],
    spec: SignalBackgroundSpec,
) -> Dict[str, object]:
    """Compute DSR using only observed counts and the analytic target set."""
    profiler = DSRProfiler(
        counts,
        spec.targets,
        num_measured_qubits=spec.num_measured_qubits,
        expected_weights={target: 0.5 for target in spec.targets},
        include_michelson=True,
    )
    return profiler.profile().to_flat_dict()


def distribution_fidelities(
    observed_counts: Mapping[str, int],
    ideal_probabilities: Mapping[str, float],
) -> Dict[str, float]:
    """Compute full-distribution Hellinger fidelity and TVD fidelity."""
    total = sum(observed_counts.values())
    if total <= 0:
        raise ValueError("observed_counts must sum to a positive value")
    ideal_total = sum(ideal_probabilities.values())
    if ideal_total <= 0:
        raise ValueError("ideal_probabilities must sum to a positive value")

    observed = {outcome: count / total for outcome, count in observed_counts.items()}
    ideal = {
        outcome: float(probability) / ideal_total
        for outcome, probability in ideal_probabilities.items()
    }
    outcomes = set(observed) | set(ideal)
    coefficient = sum(
        math.sqrt(observed.get(outcome, 0.0) * ideal.get(outcome, 0.0)) for outcome in outcomes
    )
    tvd = 0.5 * sum(
        abs(observed.get(outcome, 0.0) - ideal.get(outcome, 0.0)) for outcome in outcomes
    )
    return {
        "hellinger_fidelity": min(1.0, coefficient**2),
        "tvd": min(1.0, tvd),
        "tvd_fidelity": max(0.0, 1.0 - tvd),
    }


def analyze_spec(
    spec: SignalBackgroundSpec,
    shots: int = DEFAULT_SHOTS,
) -> Dict[str, object]:
    """Run the exact small-size proof analysis for one configuration."""
    circuit = build_signal_background_circuit(spec)
    probabilities = ideal_distribution(spec)
    ideal_counts = probabilities_to_counts(probabilities, shots)
    ideal_profile = dsr_profile(ideal_counts, spec)
    ideal_fidelities = distribution_fidelities(ideal_counts, probabilities)
    simulated_noisy_counts = noisy_counts(spec, shots)
    noisy_profile = dsr_profile(simulated_noisy_counts, spec)
    noisy_fidelities = distribution_fidelities(simulated_noisy_counts, probabilities)
    target_probability = sum(probabilities.get(target, 0.0) for target in spec.targets)
    return {
        "spec": asdict(spec),
        "circuit_depth": circuit.depth(),
        "circuit_size": circuit.size(),
        "operation_counts": dict(circuit.count_ops()),
        "heavy_hex_transpiled": heavy_hex_transpiled_complexity(spec),
        "ideal_support_size": len(probabilities),
        "ideal_target_probability": target_probability,
        "pseudo_shots": shots,
        "pseudo_counts": ideal_counts,
        "dsr_profile": ideal_profile,
        "full_distribution_fidelities": ideal_fidelities,
        "noisy_counts": simulated_noisy_counts,
        "noisy_dsr_profile": noisy_profile,
        "noisy_full_distribution_fidelities": noisy_fidelities,
        "noise_model": {
            "one_qubit_depolarizing": PROOF_ONE_QUBIT_ERROR,
            "two_qubit_depolarizing": PROOF_TWO_QUBIT_ERROR,
            "symmetric_readout": PROOF_READOUT_ERROR,
        },
    }


def _prepare_output_directory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_circuit_image(
    spec: SignalBackgroundSpec,
    output_dir: Path = DEFAULT_IMAGE_DIR,
) -> Path:
    """Render the circuit as a readable monospaced PNG."""
    output_dir = _prepare_output_directory(output_dir)
    circuit = build_signal_background_circuit(spec)
    drawing = str(circuit.draw(output="text", fold=110))
    line_count = drawing.count("\n") + 1
    figure_height = max(5.0, line_count * 0.30)
    figure, axis = plt.subplots(figsize=(20, figure_height))
    axis.axis("off")
    axis.text(
        0.01,
        0.99,
        drawing,
        family="monospace",
        fontsize=9,
        ha="left",
        va="top",
        transform=axis.transAxes,
    )
    axis.set_title(
        f"BV-derived signal-plus-background circuit ({spec.num_total_qubits} total qubits)",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    path = output_dir / f"bv_signal_background_circuit_q{spec.num_total_qubits}.png"
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def generate_architecture_image(
    output_dir: Path = DEFAULT_IMAGE_DIR,
) -> Path:
    """Render a conceptual diagram of the signal and background branches."""
    output_dir = _prepare_output_directory(output_dir)
    figure, axis = plt.subplots(figsize=(16, 7))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 8)
    axis.axis("off")

    boxes = [
        (0.5, 3.0, 2.5, 2.0, "Branch rotation\n90% signal / 10% background", "#dddddd"),
        (4.0, 5.0, 3.2, 1.6, "Signal branch\nBV decode + selector", "#80cdc1"),
        (8.2, 5.0, 3.0, 1.6, "Known target set E\nTwo basis states", "#80cdc1"),
        (4.0, 1.2, 3.2, 1.6, "Background branch\nSeeded controlled gates", "#bdbdbd"),
        (8.2, 1.2, 3.0, 1.6, "Broad nonuniform\nbasis distribution", "#bdbdbd"),
        (
            12.4,
            3.0,
            3.0,
            2.0,
            "Measurement counts\nDSR uses E only\nHF/TVDF use full ideal",
            "#fdb863",
        ),
    ]
    for x_pos, y_pos, width, height, label, color in boxes:
        axis.text(
            x_pos + width / 2,
            y_pos + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": color,
                "edgecolor": "#333333",
                "linewidth": 1.8,
            },
        )

    arrow = {"arrowstyle": "->", "linewidth": 2.2, "color": "#333333"}
    axis.annotate("", xy=(4.0, 5.8), xytext=(3.0, 4.4), arrowprops=arrow)
    axis.annotate("", xy=(4.0, 2.0), xytext=(3.0, 3.6), arrowprops=arrow)
    axis.annotate("", xy=(8.2, 5.8), xytext=(7.2, 5.8), arrowprops=arrow)
    axis.annotate("", xy=(8.2, 2.0), xytext=(7.2, 2.0), arrowprops=arrow)
    axis.annotate("", xy=(12.4, 4.2), xytext=(11.2, 5.5), arrowprops=arrow)
    axis.annotate("", xy=(12.4, 3.8), xytext=(11.2, 2.3), arrowprops=arrow)
    axis.set_title(
        "BV-derived signal-plus-background proof architecture",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    path = output_dir / "bv_signal_background_architecture.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def generate_distribution_image(
    probabilities: Mapping[str, float],
    spec: SignalBackgroundSpec,
    output_dir: Path = DEFAULT_IMAGE_DIR,
    max_outcomes: int = 18,
) -> Path:
    """Plot dominant ideal outcomes and distinguish signal from background."""
    output_dir = _prepare_output_directory(output_dir)
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    selected = ordered[:max_outcomes]
    outcomes = [outcome for outcome, _probability in selected]
    values = [probability for _outcome, probability in selected]
    colors = [
        COLORBREWER_PALETTE[1] if outcome in spec.targets else COLORBREWER_PALETTE[8]
        for outcome in outcomes
    ]

    figure, axis = plt.subplots(figsize=(16, 8))
    axis.bar(range(len(outcomes)), values, color=colors, edgecolor="black", alpha=0.85)
    axis.set_xticks(range(len(outcomes)))
    axis.set_xticklabels(outcomes, rotation=60, ha="right", fontsize=10)
    axis.set_ylabel("Ideal Probability", fontsize=14, fontweight="bold")
    axis.set_xlabel("Measured Basis State", fontsize=14, fontweight="bold")
    axis.set_title(
        f"Dominant ideal outcomes ({spec.num_total_qubits} total qubits)",
        fontsize=17,
        fontweight="bold",
    )
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.text(
        0.98,
        0.95,
        "Teal: analytic BV targets\nGray: broad background",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "#555555", "alpha": 0.9},
    )
    path = output_dir / f"bv_signal_background_distribution_q{spec.num_total_qubits}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def generate_metric_image(
    analysis: Mapping[str, object],
    spec: SignalBackgroundSpec,
    output_dir: Path = DEFAULT_IMAGE_DIR,
) -> Path:
    """Compare task-level scores with full-distribution fidelities."""
    output_dir = _prepare_output_directory(output_dir)
    ideal_profile = analysis["dsr_profile"]
    ideal_fidelities = analysis["full_distribution_fidelities"]
    noisy_profile = analysis["noisy_dsr_profile"]
    noisy_fidelities = analysis["noisy_full_distribution_fidelities"]
    labels = ["Success Rate", "DSR", "HF", "TVDF"]
    ideal_values = [
        ideal_profile["success_rate"],
        ideal_profile["dsr_michelson"],
        ideal_fidelities["hellinger_fidelity"],
        ideal_fidelities["tvd_fidelity"],
    ]
    noisy_values = [
        noisy_profile["success_rate"],
        noisy_profile["dsr_michelson"],
        noisy_fidelities["hellinger_fidelity"],
        noisy_fidelities["tvd_fidelity"],
    ]

    figure, axis = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(labels))
    width = 0.36
    ideal_bars = axis.bar(
        positions - width / 2,
        ideal_values,
        width,
        color="#bdbdbd",
        edgecolor="black",
        label="Noiseless self-comparison",
    )
    noisy_bars = axis.bar(
        positions + width / 2,
        noisy_values,
        width,
        color=COLORBREWER_PALETTE[1],
        edgecolor="black",
        label="Generic noisy simulation",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Score", fontsize=14, fontweight="bold")
    axis.set_title(
        f"Proof metrics ({spec.num_total_qubits} total qubits)",
        fontsize=17,
        fontweight="bold",
    )
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(loc="lower right")
    for bars, values in ((ideal_bars, ideal_values), (noisy_bars, noisy_values)):
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    path = output_dir / f"bv_signal_background_metrics_q{spec.num_total_qubits}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def generate_scaling_image(
    analyses: list[Mapping[str, object]],
    output_dir: Path = DEFAULT_IMAGE_DIR,
) -> Path:
    """Plot proof metrics and original circuit complexity across sizes."""
    output_dir = _prepare_output_directory(output_dir)
    qubits = [row["spec"]["num_total_qubits"] for row in analyses]
    success = [row["noisy_dsr_profile"]["success_rate"] for row in analyses]
    dsr = [row["noisy_dsr_profile"]["dsr_michelson"] for row in analyses]
    hf = [row["noisy_full_distribution_fidelities"]["hellinger_fidelity"] for row in analyses]
    tvdf = [row["noisy_full_distribution_fidelities"]["tvd_fidelity"] for row in analyses]
    depths = [row["circuit_depth"] for row in analyses]
    transpiled_depths = [row["heavy_hex_transpiled"]["depth"] for row in analyses]
    two_qubit_gates = [row["heavy_hex_transpiled"]["two_qubit_gates"] for row in analyses]

    figure, (metric_axis, circuit_axis) = plt.subplots(1, 2, figsize=(16, 7))
    metric_axis.plot(qubits, success, "o-", linewidth=3, label="Success Rate")
    metric_axis.plot(qubits, dsr, "s-", linewidth=3, label="DSR")
    metric_axis.plot(qubits, hf, "^-", linewidth=3, label="HF")
    metric_axis.plot(qubits, tvdf, "D-", linewidth=3, label="TVDF")
    metric_axis.set_ylim(0.0, 1.08)
    metric_axis.set_xlabel("Total Qubits", fontsize=13, fontweight="bold")
    metric_axis.set_ylabel("Score", fontsize=13, fontweight="bold")
    metric_axis.set_title("Generic noisy simulation", fontsize=16, fontweight="bold")
    metric_axis.grid(linestyle="--", alpha=0.5)
    metric_axis.legend()

    circuit_axis.plot(qubits, depths, "o-", linewidth=3, label="Circuit Depth")
    circuit_axis.plot(
        qubits,
        transpiled_depths,
        "s-",
        linewidth=3,
        label="Heavy-Hex Transpiled Depth",
    )
    circuit_axis.plot(
        qubits,
        two_qubit_gates,
        "^-",
        linewidth=3,
        label="Transpiled Two-Qubit Gates",
    )
    circuit_axis.set_xlabel("Total Qubits", fontsize=13, fontweight="bold")
    circuit_axis.set_ylabel("Count", fontsize=13, fontweight="bold")
    circuit_axis.set_title("Routing-aware complexity", fontsize=16, fontweight="bold")
    circuit_axis.grid(linestyle="--", alpha=0.5)
    circuit_axis.legend()

    path = output_dir / "bv_signal_background_scaling_proof.png"
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def run_proof(
    output_dir: Path = DEFAULT_IMAGE_DIR,
    results_path: Path = DEFAULT_RESULTS_PATH,
    shots: int = DEFAULT_SHOTS,
) -> Dict[str, object]:
    """Analyze all small proof circuits and write figures plus JSON results."""
    output_dir = _prepare_output_directory(output_dir)
    analyses = []
    images = [str(generate_architecture_image(output_dir))]
    for total_qubits in PROOF_TOTAL_QUBITS:
        spec = make_spec(total_qubits)
        analysis = analyze_spec(spec, shots)
        probabilities = ideal_distribution(spec)
        analyses.append(analysis)
        images.extend(
            [
                str(generate_circuit_image(spec, output_dir)),
                str(generate_distribution_image(probabilities, spec, output_dir)),
                str(generate_metric_image(analysis, spec, output_dir)),
            ]
        )
    images.append(str(generate_scaling_image(analyses, output_dir)))

    payload = {
        "experiment": "BV_DERIVED_SIGNAL_BACKGROUND_PROOF",
        "description": ("Small statevector proof only; no IBM Quantum jobs are submitted."),
        "analyses": analyses,
        "images": images,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> None:
    """Generate all proof results and figures."""
    payload = run_proof()
    print(f"Saved {len(payload['images'])} images to {DEFAULT_IMAGE_DIR}")
    print(f"Saved proof results to {DEFAULT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
