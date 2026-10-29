"""The ibm_fez dataset campaign: which configs run, and the plan files.

    uv run -m qward.examples.papers.campaign_matrix --report          # sizes, jobs, QPU time
    uv run -m qward.examples.papers.campaign_matrix --write           # write 27 plan files

The selection is explicit data (``MATRIX``), validated against each runner's
config registry, so a renamed or removed config fails loudly. ``--report``
transpiles every selected config at optimization level 3 on FakeFez (local, no
IBM) to show two-qubit gates and the predicted error that motivated the cut.

Plan files are written to ``<algo>/plans/<algo>-w<window>.json``. They are
inputs to ``run_campaign.sh prepare``; nothing here contacts IBM.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional

PAPERS = Path(__file__).resolve().parent
# Not part of the dataset: GHZ and random volumetric already cover the noise
# floor. The coin-toss runner stays available; 4 pilot jobs were run before
# this decision and are kept on disk but excluded from the export.
EXCLUDED_ALGORITHMS = ("coin_toss",)
CAMPAIGN_ID = "fez-dataset-v2"
BACKEND = "ibm_fez"
SHOTS = 1024
OPTIMIZATION_LEVEL = 3
SEED_TRANSPILER = 11
RUNS_PER_WINDOW = 2
WINDOWS = (1, 2, 3)

# algorithm key -> (module, directory, config selection)
RUNNERS: Dict[str, Dict[str, str]] = {
    "ghz": {"module": "ghz.ghz_ibm", "dir": "ghz"},
    "bv": {"module": "bv.bv_ibm", "dir": "bv"},
    "vtp": {"module": "vtp.vtp_ibm", "dir": "vtp"},
    "qft": {"module": "qft.qft_ibm", "dir": "qft"},
    "pe": {"module": "phase_estimation.pe_ibm", "dir": "phase_estimation"},
    "grover": {"module": "grover.grover_ibm", "dir": "grover"},
    "rv": {"module": "random_volumetric.rv_ibm", "dir": "random_volumetric"},
    "shor": {"module": "shor.shor_ibm", "dir": "shor"},
}


def _range(prefix: str, start: int, stop: int, suffixes=("",)) -> List[str]:
    return [f"{prefix}{n}{s}" for n in range(start, stop + 1) for s in suffixes]


MATRIX: Dict[str, List[str]] = {
    # 2Q gates grow linearly 3 -> 15; predicted error 1 -> 15%.
    "ghz": _range("GHZ", 2, 16),
    # ONES (most CNOTs), ALT, SINGLE for 2-14 data qubits.
    "bv": _range("BV", 2, 14, ("-ONES", "-ALT", "-SINGLE")),
    # Static V1-V5 and dynamic S1-S5, three payload sizes each.
    "vtp": [
        f"VTP-{kind}{n}-{payload}"
        for kind in ("S", "V")
        for n in range(1, 6)
        for payload in ("SINGLE", "TRIPLE", "FIVE")
    ],
    # Round trip to 8 qubits, period detection <= 9 qubits, input variants at 4.
    "qft": (
        _range("SR", 2, 8)
        + ["SP3-P2", "SP4-P2", "SP4-P4", "SP5-P4", "SP6-P4", "SP6-P8", "SP8-P4", "SP8-P16"]
        + ["PV4-P2", "PV4-P4", "PV4-P8", "PV6-P2", "PV6-P4", "PV6-P8", "PV6-P16"]
        + ["IV4-0000", "IV4-1111", "IV4-0101", "IV4-1010", "IV4-0001"]
    ),
    # Three phase families, 2-8 counting qubits.
    "pe": [f"PE-{family}{n}" for family in ("T", "S", "Z") for n in range(2, 9)],
    # Scaling to 4 qubits, multi-marked, Hamming weight, symmetry; S5 as a failure anchor.
    "grover": [
        "S2-1",
        "S2-00",
        "S2-10",
        "S2-11",
        "S3-1",
        "S4-1",
        "S5-1",
        "M3-1",
        "M3-2",
        "M3-4",
        "M4-1",
        "M4-2",
        "M4-4",
        "H3-0",
        "H3-1",
        "H3-2",
        "H3-3",
        "H4-0",
        "H4-2",
        "H4-4",
        "SYM-1",
        "SYM-2",
        "ASYM-1",
        "ASYM-2",
    ],
    # Mirror circuits: RV2-RV5 at three depths, plus the edge points of RV6-RV8.
    "rv": (
        [
            f"RV{n}-D{d}"
            for n, ds in ((2, (2, 4, 6)), (3, (3, 6, 9)), (4, (4, 8, 12)), (5, (5, 10, 15)))
            for d in ds
        ]
        + ["RV6-D6", "RV7-D7", "RV8-D8"]
    ),
    # One config as a known-failure anchor (~99% predicted error).
    "shor": ["SHOR-N15-M3"],
}


# Pilot (Phase 7): one small and one mid-size config per algorithm. Window 0
# keeps pilot plans, manifests and budgets apart from the three real windows.
PILOT_WINDOW = 0
PILOT_RUNS = 2
PILOT: Dict[str, List[str]] = {
    "ghz": ["GHZ3", "GHZ10"],
    "bv": ["BV3-ALT", "BV8-ONES"],
    "vtp": ["VTP-V2-SINGLE", "VTP-S2-TRIPLE"],  # One static, one dynamic.
    "qft": ["SR3", "SP6-P4"],
    "pe": ["PE-T3", "PE-S6"],
    "grover": ["S3-1", "M4-2"],
    "rv": ["RV2-D2", "RV4-D4"],
    "shor": ["SHOR-N15-M3"],
}


# Fixed per algorithm so adding or removing an algorithm never changes the
# shuffle of plans that already have manifests (alphabetical order of the
# original nine, coin toss included).
ORDER_SEED_OFFSET: Dict[str, int] = {
    "bv": 0,
    "coin_toss": 1,
    "ghz": 2,
    "grover": 3,
    "pe": 4,
    "qft": 5,
    "rv": 6,
    "shor": 7,
    "vtp": 8,
}


def load_runner(key: str):
    """Instantiate the IBM runner for an algorithm key."""
    module = importlib.import_module(f"qward.examples.papers.{RUNNERS[key]['module']}")
    cls = [
        c
        for n, c in inspect.getmembers(module, inspect.isclass)
        if n.endswith("IBMExperiment") and c.__module__ == module.__name__
    ][0]
    return cls()


def validate_matrix() -> Dict[str, List[str]]:
    """Every selected config must exist and none may repeat."""
    problems: Dict[str, List[str]] = {}
    for key, configs in MATRIX.items():
        known = set(load_runner(key).get_all_config_ids())
        bad = [c for c in configs if c not in known]
        if bad:
            problems.setdefault(key, []).append(f"unknown configs: {bad}")
        if len(set(configs)) != len(configs):
            problems.setdefault(key, []).append("duplicate configs")
        stray = [c for c in PILOT.get(key, []) if c not in configs]
        if stray or key not in PILOT:
            problems.setdefault(key, []).append(f"pilot configs outside the matrix: {stray}")
    return problems


def plan_for(key: str, window: int) -> Dict:
    """Plan JSON for one algorithm and window (window 0 is the pilot)."""
    pilot = window == PILOT_WINDOW
    configs = PILOT[key] if pilot else MATRIX[key]
    runs = PILOT_RUNS if pilot else RUNS_PER_WINDOW
    return {
        "backend": BACKEND,
        "campaign_id": CAMPAIGN_ID,
        "window": window,
        "shots": SHOTS,
        "optimization_level": OPTIMIZATION_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "order_seed": 1000 * window + ORDER_SEED_OFFSET[key],
        "runs": runs,
        "planned_executions": len(configs) * runs,
        "purpose": (
            f"{key.upper()} PILOT: {len(configs)} configs, {runs} runs each, one Batch. "
            "Verify the export end to end before the real windows."
            if pilot
            else f"{key.upper()} window {window} of {len(WINDOWS)}: all configs, "
            f"{runs} runs each, one Batch. Prepare just before submitting; "
            "windows on different days."
        ),
        "items": [{"config": c} for c in configs],
    }


def plan_path(key: str, window: int, root: Path = PAPERS) -> Path:
    return Path(root) / RUNNERS[key]["dir"] / "plans" / f"{key}-w{window}.json"


def write_plans(root: Path = PAPERS, *, overwrite: bool = False) -> List[Path]:
    """Write all plan files; refuse to replace one that already has a manifest."""
    problems = validate_matrix()
    if problems:
        raise ValueError(f"Invalid matrix: {problems}")
    written = []
    for key in MATRIX:
        for window in (PILOT_WINDOW, *WINDOWS):
            path = plan_path(key, window, root)
            manifest = path.with_name(f"{path.stem}_manifest.json")
            if manifest.exists():
                # The plan is in use: keep it as it is and leave the others alone.
                if json.loads(path.read_text(encoding="utf-8")) != plan_for(key, window):
                    raise ValueError(f"{path} differs from the matrix but {manifest} exists")
                continue
            if path.exists() and not overwrite:
                raise ValueError(f"{path} exists (use --overwrite)")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(plan_for(key, window), indent=2) + "\n", encoding="utf-8")
            written.append(path)
    return written


def measure(key: str, config_id: str, backend, pass_manager) -> Dict:
    """Local transpile at the campaign settings: size and predicted error."""
    from qward.metrics.circuit_error_prediction import capture_calibration, is_dynamic

    runner = load_runner(key)
    circuit = runner.create_circuit(runner.get_config(config_id))
    isa = pass_manager.run(circuit)
    dynamic = is_dynamic(isa)
    two_qubit = sum(len(i.qubits) == 2 for i in isa.data)
    predicted: Optional[float] = None
    if not dynamic:
        survive = 1.0
        for entry in capture_calibration(isa, backend, include_backend_properties=False)[
            "instructions"
        ]:
            if entry["name"] not in {"delay", "barrier"} and entry["error"] is not None:
                survive *= 1 - entry["error"]
        predicted = 1 - survive
    return {
        "config": config_id,
        "qubits": circuit.num_qubits,
        "measured_bits": circuit.num_clbits,
        "isa_depth": isa.depth(lambda i: i.operation.name not in {"delay", "barrier"}),
        "two_qubit_gates": two_qubit,
        "dynamic": dynamic,
        "m1_error": predicted,
    }


# QPU seconds per job at 1,024 shots. Real ibm_fez jobs at 2,048 shots with
# 30-200 two-qubit gates billed 3 s each (usage.quantum_seconds, integer
# seconds). Deeper circuits take longer to execute; the model below is a
# conservative upper bound, not a measurement.
def qpu_seconds(isa_depth: int, shots: int = SHOTS, rep_delay_s: float = 250e-6) -> int:
    per_shot_s = rep_delay_s + isa_depth * 5e-8  # ~50 ns per layer
    return max(2, int(round(shots * per_shot_s)) + 2)


def report(sample: Optional[List[str]] = None) -> Dict:
    """Sizes, job counts and QPU-time estimates per algorithm (local)."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    pm = generate_preset_pass_manager(
        optimization_level=OPTIMIZATION_LEVEL, backend=backend, seed_transpiler=SEED_TRANSPILER
    )
    summary: Dict[str, Dict] = {}
    for key, configs in MATRIX.items():
        if sample and key not in sample:
            continue
        rows = [measure(key, c, backend, pm) for c in configs]
        seconds = sum(qpu_seconds(r["isa_depth"]) for r in rows) * RUNS_PER_WINDOW
        summary[key] = {
            "configs": len(rows),
            "jobs_per_window": len(rows) * RUNS_PER_WINDOW,
            "jobs_total": len(rows) * RUNS_PER_WINDOW * len(WINDOWS),
            "qpu_seconds_per_window": seconds,
            "max_two_qubit_gates": max(r["two_qubit_gates"] for r in rows),
            "max_m1_error": max(
                (r["m1_error"] for r in rows if r["m1_error"] is not None), default=None
            ),
            "dynamic_configs": sum(r["dynamic"] for r in rows),
            "rows": rows,
        }
    return summary


def precompute_quantum_specific(only: Optional[List[str]] = None) -> None:
    """Compute and cache QuantumSpecific features for the matrix, config by config."""
    from qward.examples.papers import dataset_features  # pylint: disable=import-outside-toplevel

    for key, configs in MATRIX.items():
        if only and key not in only:
            continue
        runner = load_runner(key)
        for config_id in configs:
            circuit = runner.create_circuit(runner.get_config(config_id))
            result = dataset_features.quantum_specific(circuit)
            state = (
                "cached"
                if result.get("cached")
                else "computed" if result.get("available") else result.get("reason", "?")
            )
            print(f"{key:10}{config_id:16}{circuit.num_qubits:>3} qubits  {state}", flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--precompute-quantum-specific",
        action="store_true",
        help="Fill the QuantumSpecific cache for every matrix config (~24 min once)",
    )
    parser.add_argument("--only", nargs="+", choices=sorted(MATRIX))
    parser.add_argument("--json", type=Path, help="Write the full report here")
    args = parser.parse_args(argv)
    problems = validate_matrix()
    if problems:
        raise SystemExit(f"Invalid matrix: {problems}")
    if args.report:
        summary = report(args.only)
        print(
            f"{'algo':10}{'configs':>8}{'jobs/win':>9}{'jobs':>7}{'QPU s/win':>10}{'max 2Q':>8}{'max M1 err':>11}{'dyn':>5}"
        )
        for key, s in summary.items():
            err = f"{s['max_m1_error']:.2f}" if s["max_m1_error"] is not None else "-"
            print(
                f"{key:10}{s['configs']:>8}{s['jobs_per_window']:>9}{s['jobs_total']:>7}"
                f"{s['qpu_seconds_per_window']:>10}{s['max_two_qubit_gates']:>8}{err:>11}{s['dynamic_configs']:>5}"
            )
        total_jobs = sum(s["jobs_total"] for s in summary.values())
        seconds = sum(s["qpu_seconds_per_window"] for s in summary.values())
        print(
            f"{'TOTAL':10}{sum(s['configs'] for s in summary.values()):>8}"
            f"{sum(s['jobs_per_window'] for s in summary.values()):>9}{total_jobs:>7}{seconds:>10}"
        )
        print(
            f"QPU time: ~{seconds / 60:.0f} min per window, ~{seconds * len(WINDOWS) / 60:.0f} min total (upper bound)"
        )
        if args.json:
            args.json.write_text(
                json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
            )
    if args.precompute_quantum_specific:
        precompute_quantum_specific(args.only)
    if args.write:
        for path in write_plans(overwrite=args.overwrite):
            print(f"wrote {path.relative_to(PAPERS)}")


if __name__ == "__main__":
    main()
