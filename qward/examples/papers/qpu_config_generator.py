#!/usr/bin/env python3
"""
QPU Configuration Generator

This script analyzes raw simulator data files and generates configurations
for executing quantum algorithms on real IBM QPU hardware.

It reads raw JSON files directly (no campaign summary needed), classifies
configurations into three regions, and outputs the configurations worth
executing on QPU.

Usage:
    python qpu_config_generator.py --algorithm grover --data-dir grover/data/simulator/raw
    python qpu_config_generator.py --algorithm qft --data-dir qft/data/raw

    # Filter by noise model
    python qpu_config_generator.py --algorithm grover --noise-models IBM-HERON-R2 IBM-HERON-R3

    # Output configurations as Python code
    python qpu_config_generator.py --algorithm grover --output-format python
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys


@dataclass
class RawResult:
    """Parsed result from a raw data file."""

    config_id: str
    noise_model: str
    algorithm: str
    mean_success_rate: float
    std_success_rate: float
    num_runs: int
    num_qubits: int
    circuit_depth: int
    total_gates: int

    # Derived
    region: str = ""
    random_chance: float = 0.0
    quantum_advantage_ratio: float = 0.0

    # For Grover
    marked_states: list = None
    num_marked: int = 1

    # For QFT
    test_mode: str = None
    input_state: str = None
    period: int = None


def parse_raw_file(file_path: Path) -> Optional[RawResult]:
    """Parse a raw JSON data file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        config_id = data.get("config_id", "")
        noise_id = data.get("noise_id", "")
        algorithm = data.get("algorithm", "").upper()

        batch = data.get("batch_summary", {})
        mean_success = batch.get("mean_success_rate", 0)
        std_success = batch.get("std_success_rate", 0)
        num_runs = batch.get("num_runs", 10)

        # Get circuit info from first individual result
        individual = data.get("individual_results", [{}])
        first_result = individual[0] if individual else {}

        num_qubits = first_result.get("num_qubits", 0)
        circuit_depth = first_result.get("circuit_depth", 0)
        total_gates = first_result.get("total_gates", 0)

        result = RawResult(
            config_id=config_id,
            noise_model=noise_id,
            algorithm=algorithm,
            mean_success_rate=mean_success,
            std_success_rate=std_success,
            num_runs=num_runs,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            total_gates=total_gates,
        )

        # Algorithm-specific parsing
        if algorithm == "GROVER":
            result.marked_states = first_result.get("marked_states", [])
            result.num_marked = first_result.get(
                "num_marked", len(result.marked_states) if result.marked_states else 1
            )
        elif algorithm == "QFT":
            result.input_state = first_result.get("input_state")
            result.period = first_result.get("period")
            result.test_mode = "period_detection" if result.period else "roundtrip"

        return result

    except Exception as e:
        print(f"Warning: Failed to parse {file_path.name}: {e}")
        return None


def calculate_random_chance(num_qubits: int, num_marked: int) -> float:
    """Calculate random chance baseline."""
    search_space = 2**num_qubits
    return num_marked / search_space


def classify_region(
    success_rate: float,
    random_chance: float,
    region1_threshold: float = 0.50,
    advantage_multiplier: float = 2.0,
) -> tuple[str, float]:
    """Classify into Region 1, 2, or 3."""
    advantage_threshold = advantage_multiplier * random_chance
    advantage_ratio = success_rate / random_chance if random_chance > 0 else float("inf")

    if success_rate > region1_threshold and success_rate > advantage_threshold:
        return "Region 1", advantage_ratio
    elif success_rate > advantage_threshold:
        return "Region 2", advantage_ratio
    else:
        return "Region 3", advantage_ratio


def scan_raw_data(data_dir: Path, noise_models: Optional[list[str]] = None) -> list[RawResult]:
    """Scan directory for raw JSON files and parse them."""
    results = []

    # Find all JSON files
    json_files = list(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {data_dir}")

    for file_path in json_files:
        result = parse_raw_file(file_path)
        if result is None:
            continue

        # Filter by noise model if specified
        if noise_models and result.noise_model not in noise_models:
            continue

        # Calculate random chance and classify
        if result.algorithm == "GROVER":
            result.random_chance = calculate_random_chance(result.num_qubits, result.num_marked)
        elif result.algorithm == "QFT":
            if result.test_mode == "roundtrip":
                result.random_chance = calculate_random_chance(result.num_qubits, 1)
            else:
                # Period detection: multiple valid peaks
                # Number of peaks = N / period (e.g., 16 states / period 4 = 4 peaks)
                search_space = 2**result.num_qubits
                num_peaks = search_space // result.period if result.period else 1
                result.random_chance = num_peaks / search_space

        region, advantage_ratio = classify_region(result.mean_success_rate, result.random_chance)
        result.region = region
        result.quantum_advantage_ratio = advantage_ratio

        results.append(result)

    return results


def group_by_config(results: list[RawResult]) -> dict[str, list[RawResult]]:
    """Group results by config_id."""
    groups = {}
    for r in results:
        if r.config_id not in groups:
            groups[r.config_id] = []
        groups[r.config_id].append(r)
    return groups


def format_percentage(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.1f}%"


def region_symbol(region: str) -> str:
    """Emoji for region."""
    if "Region 1" in region:
        return "✅"
    elif "Region 2" in region:
        return "⚠️"
    else:
        return "❌"


def print_analysis_table(results: list[RawResult], reference_noise: str) -> str:
    """Print analysis table."""
    groups = group_by_config(results)

    # Get all noise models present
    all_noise = sorted(set(r.noise_model for r in results))

    # Sort by qubits then config_id
    sorted_configs = sorted(groups.keys(), key=lambda c: (groups[c][0].num_qubits, c))

    lines = []
    lines.append("\n## Configuration Analysis\n")

    header = ["Config", "Qubits", "Depth", "Gates"] + all_noise + ["Random", "Region"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for config_id in sorted_configs:
        config_results = groups[config_id]
        first = config_results[0]

        row = [
            f"**{config_id}**",
            str(first.num_qubits),
            str(first.circuit_depth),
            str(first.total_gates),
        ]

        # Add success rates for each noise model
        for nm in all_noise:
            nm_result = next((r for r in config_results if r.noise_model == nm), None)
            if nm_result:
                row.append(f"**{format_percentage(nm_result.mean_success_rate)}**")
            else:
                row.append("-")

        # Get region from reference noise
        ref_result = next(
            (r for r in config_results if r.noise_model == reference_noise), config_results[0]
        )

        row.append(format_percentage(first.random_chance))
        row.append(f"{ref_result.region} {region_symbol(ref_result.region)}")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def print_region_summary(results: list[RawResult], reference_noise: str) -> str:
    """Print region summary."""
    # Filter by reference noise
    ref_results = [r for r in results if r.noise_model == reference_noise]

    regions = {"Region 1": [], "Region 2": [], "Region 3": []}
    for r in ref_results:
        regions[r.region].append(r)

    lines = []
    lines.append(f"\n## Region Summary (Reference: {reference_noise})\n")

    for region, configs in regions.items():
        symbol = region_symbol(region)
        count = len(configs)

        if region == "Region 1":
            desc = "Worth running on QPU (algorithm works)"
        elif region == "Region 2":
            desc = "Marginal - boundary exploration"
        else:
            desc = "Algorithm fails - document limits only"

        lines.append(f"### {region} {symbol} ({count} configs)")
        lines.append(f"**{desc}**\n")

        if configs:
            sorted_configs = sorted(configs, key=lambda x: -x.mean_success_rate)
            config_list = ", ".join(c.config_id for c in sorted_configs)
            lines.append(f"Configs: {config_list}\n")

            rates = [c.mean_success_rate for c in configs]
            lines.append(
                f"Success range: {format_percentage(min(rates))} - {format_percentage(max(rates))}\n"
            )

    return "\n".join(lines)


def generate_grover_qpu_configs(
    results: list[RawResult], reference_noise: str, max_configs: int = 15
) -> str:
    """Generate Python code for Grover QPU configurations."""
    ref_results = [r for r in results if r.noise_model == reference_noise]

    # Sort by region priority and success rate
    priority_order = {"Region 1": 1, "Region 2": 2, "Region 3": 3}
    sorted_results = sorted(
        ref_results, key=lambda r: (priority_order.get(r.region, 4), -r.mean_success_rate)
    )

    lines = []
    lines.append("\n## QPU Configurations (Python)\n")
    lines.append("```python")
    lines.append("# Grover QPU Experiment Configurations")
    lines.append("# Generated from simulator data analysis")
    lines.append("# Reference noise model: " + reference_noise)
    lines.append("")
    lines.append(
        "from qward.examples.papers.grover.grover_configs import get_config, get_noise_config"
    )
    lines.append("")
    lines.append("# Configurations to run on QPU (prioritized by expected success)")
    lines.append("QPU_CONFIGS = [")

    for i, r in enumerate(sorted_results[:max_configs], 1):
        runs = 5 if "Region 1" in r.region else (3 if "Region 2" in r.region else 2)
        lines.append(
            f"    # Priority {i}: {r.config_id} - {format_percentage(r.mean_success_rate)} expected ({r.region})"
        )
        lines.append(f"    {{")
        lines.append(f'        "config_id": "{r.config_id}",')
        lines.append(f'        "num_qubits": {r.num_qubits},')
        lines.append(f'        "marked_states": {r.marked_states},')
        lines.append(f'        "expected_success": {r.mean_success_rate:.4f},')
        lines.append(f'        "circuit_depth": {r.circuit_depth},')
        lines.append(f'        "total_gates": {r.total_gates},')
        lines.append(f'        "runs": {runs},')
        lines.append(f'        "region": "{r.region}",')
        lines.append(f"    }},")

    lines.append("]")
    lines.append("")
    lines.append("# Quick access to configurations")
    lines.append("def get_qpu_config(config_id: str):")
    lines.append('    """Get full ExperimentConfig for a QPU experiment."""')
    lines.append("    return get_config(config_id)")
    lines.append("")
    lines.append("# Total experiments")
    total_runs = sum(
        5 if "Region 1" in r.region else (3 if "Region 2" in r.region else 2)
        for r in sorted_results[:max_configs]
    )
    lines.append(f"# Total runs: {total_runs}")
    lines.append(f"# Estimated QPU time: ~{total_runs * 5} seconds")
    lines.append("```")

    return "\n".join(lines)


def generate_qft_qpu_configs(
    results: list[RawResult], reference_noise: str, max_configs: int = 15
) -> str:
    """Generate Python code for QFT QPU configurations."""
    ref_results = [r for r in results if r.noise_model == reference_noise]

    priority_order = {"Region 1": 1, "Region 2": 2, "Region 3": 3}
    sorted_results = sorted(
        ref_results, key=lambda r: (priority_order.get(r.region, 4), -r.mean_success_rate)
    )

    lines = []
    lines.append("\n## QPU Configurations (Python)\n")
    lines.append("```python")
    lines.append("# QFT QPU Experiment Configurations")
    lines.append("# Generated from simulator data analysis")
    lines.append("# Reference noise model: " + reference_noise)
    lines.append("")
    lines.append("from qward.examples.papers.qft.qft_configs import get_config, get_noise_config")
    lines.append("")
    lines.append("# Configurations to run on QPU (prioritized by expected success)")
    lines.append("QPU_CONFIGS = [")

    for i, r in enumerate(sorted_results[:max_configs], 1):
        runs = 5 if "Region 1" in r.region else (3 if "Region 2" in r.region else 2)
        lines.append(
            f"    # Priority {i}: {r.config_id} - {format_percentage(r.mean_success_rate)} expected ({r.region})"
        )
        lines.append(f"    {{")
        lines.append(f'        "config_id": "{r.config_id}",')
        lines.append(f'        "num_qubits": {r.num_qubits},')
        lines.append(f'        "test_mode": "{r.test_mode}",')
        if r.test_mode == "roundtrip":
            lines.append(f'        "input_state": "{r.input_state}",')
        else:
            lines.append(f'        "period": {r.period},')
        lines.append(f'        "expected_success": {r.mean_success_rate:.4f},')
        lines.append(f'        "circuit_depth": {r.circuit_depth},')
        lines.append(f'        "total_gates": {r.total_gates},')
        lines.append(f'        "runs": {runs},')
        lines.append(f'        "region": "{r.region}",')
        lines.append(f"    }},")

    lines.append("]")
    lines.append("")
    lines.append("# Quick access to configurations")
    lines.append("def get_qpu_config(config_id: str):")
    lines.append('    """Get full QFTExperimentConfig for a QPU experiment."""')
    lines.append("    return get_config(config_id)")
    lines.append("")
    total_runs = sum(
        5 if "Region 1" in r.region else (3 if "Region 2" in r.region else 2)
        for r in sorted_results[:max_configs]
    )
    lines.append(f"# Total runs: {total_runs}")
    lines.append(f"# Estimated QPU time: ~{total_runs * 5} seconds")
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate QPU configurations from raw simulator data"
    )
    parser.add_argument(
        "--algorithm", "-a", required=True, choices=["grover", "qft"], help="Algorithm type"
    )
    parser.add_argument("--data-dir", "-d", required=True, help="Path to raw data directory")
    parser.add_argument(
        "--noise-models",
        "-n",
        nargs="+",
        default=["IBM-HERON-R3", "IBM-HERON-R2", "IBM-HERON-R1", "RIGETTI-ANKAA3"],
        help="Noise models to include",
    )
    parser.add_argument(
        "--reference-noise",
        "-r",
        default="IBM-HERON-R2",
        help="Reference noise model for classification",
    )
    parser.add_argument(
        "--max-configs", "-m", type=int, default=15, help="Maximum configs to output"
    )
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    # Find data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        data_dir = script_dir / args.data_dir

    if not data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    print(f"Scanning {data_dir}...")
    results = scan_raw_data(data_dir, args.noise_models)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found. Check data directory and noise model filters.")
        sys.exit(1)

    # Check if reference noise model is in results
    available_noise = set(r.noise_model for r in results)
    if args.reference_noise not in available_noise:
        print(f"Warning: Reference noise '{args.reference_noise}' not in data.")
        print(f"Available: {sorted(available_noise)}")
        args.reference_noise = sorted(available_noise)[0]
        print(f"Using: {args.reference_noise}")

    # Generate report
    output_lines = []
    output_lines.append(f"# {args.algorithm.upper()} QPU Configuration Report\n")
    output_lines.append(f"**Data source:** {data_dir}")
    output_lines.append(f"**Noise models:** {', '.join(sorted(available_noise))}")
    output_lines.append(f"**Reference:** {args.reference_noise}")
    output_lines.append(f"**Configs analyzed:** {len(set(r.config_id for r in results))}\n")

    # Analysis table
    output_lines.append(print_analysis_table(results, args.reference_noise))

    # Region summary
    output_lines.append(print_region_summary(results, args.reference_noise))

    # QPU configurations
    if args.algorithm == "grover":
        output_lines.append(
            generate_grover_qpu_configs(results, args.reference_noise, args.max_configs)
        )
    else:
        output_lines.append(
            generate_qft_qpu_configs(results, args.reference_noise, args.max_configs)
        )

    # Output
    full_output = "\n".join(output_lines)
    print(full_output)

    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir.parent / f"qpu_configs_{args.algorithm}.md"

    with open(output_path, "w") as f:
        f.write(full_output)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
