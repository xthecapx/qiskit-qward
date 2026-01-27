#!/usr/bin/env python3
"""
Region Analysis Script for Quantum Algorithm Experiments

This script analyzes simulator data to classify algorithm configurations
into three regions:
- Region 1 (Signal Dominant): Algorithm works well, quantum advantage clear
- Region 2 (Signal + Noise): Marginal performance, transition zone
- Region 3 (Noise Dominant): Algorithm fails, near random chance

Usage:
    python region_analysis.py <campaign_summary.json> [--noise-model IBM-HERON-R2]
    
Example:
    python region_analysis.py grover/data/simulator/aggregated/campaign_summary_*.json
    python region_analysis.py qft/data/aggregated/campaign_summary_*.json --noise-model RIGETTI-ANKAA3
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import sys


@dataclass
class RegionConfig:
    """Configuration for region classification thresholds."""
    
    # Region 1: Success > region1_threshold AND > advantage_multiplier * random
    region1_threshold: float = 0.50
    
    # Region 2: Success > advantage_multiplier * random (but not meeting Region 1)
    # Region 3: Success <= advantage_multiplier * random
    advantage_multiplier: float = 2.0
    
    # For algorithms with multiple valid outcomes (like QFT period detection)
    # we may want different thresholds
    algorithm_specific: dict = field(default_factory=dict)


@dataclass
class ConfigResult:
    """Parsed result for a single configuration."""
    config_id: str
    noise_model: str
    mean_success_rate: float
    std_success_rate: float
    min_success_rate: float
    max_success_rate: float
    num_runs: int
    
    # Derived fields (set after parsing)
    num_qubits: int = 0
    num_marked: int = 1
    search_space: int = 0
    random_chance: float = 0.0
    region: str = ""
    quantum_advantage_ratio: float = 0.0
    
    # Optional metadata
    circuit_depth: Optional[int] = None
    total_gates: Optional[int] = None
    test_mode: Optional[str] = None  # For QFT: "roundtrip" or "period_detection"
    period: Optional[int] = None  # For QFT period detection


def parse_grover_config(config_id: str) -> tuple[int, int]:
    """
    Parse Grover config ID to extract num_qubits and num_marked.
    
    Config naming conventions:
    - S{n}-1: Scalability study, n qubits, 1 marked state
    - M{n}-{m}: Marked count study, n qubits, m marked states
    - H{n}-{h}: Hamming weight study, n qubits, 1 marked state
    - SYM-{n}, ASYM-{n}: Symmetry study, 3 qubits, 2 marked states
    """
    config_id = config_id.upper()
    
    # Symmetry: SYM-1, SYM-2, ASYM-1, ASYM-2 (check BEFORE scalability since both start with S)
    if config_id.startswith("SYM") or config_id.startswith("ASYM"):
        return 3, 2  # 3 qubits, 2 marked states
    
    # Scalability: S2-1, S3-1, etc. (single letter S followed by digit)
    if config_id.startswith("S") and len(config_id) > 1 and config_id[1].isdigit() and "-" in config_id:
        parts = config_id.split("-")
        num_qubits = int(parts[0][1:])
        num_marked = int(parts[1]) if len(parts) > 1 else 1
        return num_qubits, num_marked
    
    # Marked count: M3-1, M3-2, M4-4, etc.
    if config_id.startswith("M") and "-" in config_id:
        parts = config_id.split("-")
        num_qubits = int(parts[0][1:])
        num_marked = int(parts[1])
        return num_qubits, num_marked
    
    # Hamming weight: H3-0, H3-3, H4-2, etc.
    if config_id.startswith("H") and "-" in config_id:
        parts = config_id.split("-")
        num_qubits = int(parts[0][1:])
        num_marked = 1  # Always 1 marked state in Hamming study
        return num_qubits, num_marked
    
    # Default fallback
    print(f"Warning: Unknown config format '{config_id}', defaulting to 3 qubits, 1 marked")
    return 3, 1


def parse_qft_config(config_id: str) -> tuple[int, int, str, Optional[int]]:
    """
    Parse QFT config ID to extract num_qubits, num_valid_outcomes, mode, period.
    
    Config naming conventions:
    - SR{n}: Scalability roundtrip, n qubits, 1 valid outcome
    - SP{n}-P{p}: Scalability period detection, n qubits, period p
    - PV{n}-P{p}: Period variation, n qubits, period p
    - IV{n}-{state}: Input variation, n qubits, roundtrip mode
    """
    config_id = config_id.upper()
    
    # Scalability roundtrip: SR2, SR3, etc.
    if config_id.startswith("SR") and "-" not in config_id:
        num_qubits = int(config_id[2:])
        return num_qubits, 1, "roundtrip", None
    
    # Scalability period detection: SP3-P2, SP4-P4, etc.
    if config_id.startswith("SP") and "-P" in config_id:
        parts = config_id.split("-P")
        num_qubits = int(parts[0][2:])
        period = int(parts[1])
        search_space = 2 ** num_qubits
        num_valid = search_space // period  # Number of valid peaks
        return num_qubits, num_valid, "period_detection", period
    
    # Period variation: PV4-P2, PV6-P8, etc.
    if config_id.startswith("PV") and "-P" in config_id:
        parts = config_id.split("-P")
        num_qubits = int(parts[0][2:])
        period = int(parts[1])
        search_space = 2 ** num_qubits
        num_valid = search_space // period
        return num_qubits, num_valid, "period_detection", period
    
    # Input variation: IV4-0000, IV4-1111, etc.
    if config_id.startswith("IV"):
        parts = config_id.split("-")
        num_qubits = int(parts[0][2:])
        return num_qubits, 1, "roundtrip", None
    
    # Default fallback
    print(f"Warning: Unknown QFT config format '{config_id}', defaulting to 4 qubits, roundtrip")
    return 4, 1, "roundtrip", None


def calculate_random_chance(num_qubits: int, num_valid_outcomes: int) -> float:
    """Calculate the random chance (classical baseline) for success."""
    search_space = 2 ** num_qubits
    return num_valid_outcomes / search_space


def classify_region(
    success_rate: float,
    random_chance: float,
    config: RegionConfig
) -> tuple[str, float]:
    """
    Classify a configuration into one of three regions.
    
    Returns:
        Tuple of (region_name, quantum_advantage_ratio)
    """
    advantage_threshold = config.advantage_multiplier * random_chance
    advantage_ratio = success_rate / random_chance if random_chance > 0 else float('inf')
    
    if success_rate > config.region1_threshold and success_rate > advantage_threshold:
        return "Region 1", advantage_ratio
    elif success_rate > advantage_threshold:
        return "Region 2", advantage_ratio
    else:
        return "Region 3", advantage_ratio


def load_campaign_data(file_path: str) -> list[dict]:
    """Load campaign summary JSON data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Handle case where data is wrapped in a dict
        if 'results' in data:
            return data['results']
        return [data]
    
    return data


def analyze_grover_data(
    data: list[dict],
    noise_models: Optional[list[str]] = None,
    region_config: Optional[RegionConfig] = None
) -> list[ConfigResult]:
    """
    Analyze Grover experiment data and classify into regions.
    """
    if region_config is None:
        region_config = RegionConfig()
    
    results = []
    
    for entry in data:
        config_id = entry.get("config_id", "")
        noise_model = entry.get("noise_model", "")
        
        # Filter by noise model if specified
        if noise_models and noise_model not in noise_models:
            continue
        
        # Parse config to get qubit/marked info
        num_qubits, num_marked = parse_grover_config(config_id)
        search_space = 2 ** num_qubits
        random_chance = calculate_random_chance(num_qubits, num_marked)
        
        # Get success rate
        mean_success = entry.get("mean_success_rate", 0)
        
        # Classify region
        region, advantage_ratio = classify_region(mean_success, random_chance, region_config)
        
        # Create result object
        result = ConfigResult(
            config_id=config_id,
            noise_model=noise_model,
            mean_success_rate=mean_success,
            std_success_rate=entry.get("std_success_rate", 0),
            min_success_rate=entry.get("min_success_rate", 0),
            max_success_rate=entry.get("max_success_rate", 0),
            num_runs=entry.get("num_runs", 10),
            num_qubits=num_qubits,
            num_marked=num_marked,
            search_space=search_space,
            random_chance=random_chance,
            region=region,
            quantum_advantage_ratio=advantage_ratio,
        )
        
        # Try to get circuit info from nested analysis
        if "analysis" in entry:
            analysis = entry["analysis"]
            # Could extract more metrics here if available
        
        results.append(result)
    
    return results


def analyze_qft_data(
    data: list[dict],
    noise_models: Optional[list[str]] = None,
    region_config: Optional[RegionConfig] = None
) -> list[ConfigResult]:
    """
    Analyze QFT experiment data and classify into regions.
    """
    if region_config is None:
        # QFT period detection has different thresholds
        region_config = RegionConfig(
            region1_threshold=0.70,  # Lower threshold for period detection
            advantage_multiplier=1.5,  # Smaller multiplier since period detection has many valid outcomes
        )
    
    results = []
    
    for entry in data:
        config_id = entry.get("config_id", "")
        noise_model = entry.get("noise_model", "")
        
        # Filter by noise model if specified
        if noise_models and noise_model not in noise_models:
            continue
        
        # Parse config to get qubit/mode info
        num_qubits, num_valid, mode, period = parse_qft_config(config_id)
        search_space = 2 ** num_qubits
        random_chance = calculate_random_chance(num_qubits, num_valid)
        
        # Get success rate
        mean_success = entry.get("mean_success_rate", 0)
        
        # Use mode-specific thresholds
        if mode == "roundtrip":
            mode_config = RegionConfig(region1_threshold=0.80, advantage_multiplier=2.0)
        else:
            mode_config = RegionConfig(region1_threshold=0.70, advantage_multiplier=1.5)
        
        # Classify region
        region, advantage_ratio = classify_region(mean_success, random_chance, mode_config)
        
        # Create result object
        result = ConfigResult(
            config_id=config_id,
            noise_model=noise_model,
            mean_success_rate=mean_success,
            std_success_rate=entry.get("std_success_rate", 0),
            min_success_rate=entry.get("min_success_rate", 0),
            max_success_rate=entry.get("max_success_rate", 0),
            num_runs=entry.get("num_runs", 10),
            num_qubits=num_qubits,
            num_marked=num_valid,
            search_space=search_space,
            random_chance=random_chance,
            region=region,
            quantum_advantage_ratio=advantage_ratio,
            test_mode=mode,
            period=period,
        )
        
        results.append(result)
    
    return results


def group_by_config(results: list[ConfigResult]) -> dict[str, list[ConfigResult]]:
    """Group results by config_id."""
    groups = {}
    for r in results:
        if r.config_id not in groups:
            groups[r.config_id] = []
        groups[r.config_id].append(r)
    return groups


def group_by_noise_model(results: list[ConfigResult]) -> dict[str, list[ConfigResult]]:
    """Group results by noise_model."""
    groups = {}
    for r in results:
        if r.noise_model not in groups:
            groups[r.noise_model] = []
        groups[r.noise_model].append(r)
    return groups


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value * 100:.1f}%"


def region_symbol(region: str) -> str:
    """Return emoji/symbol for region."""
    if "Region 1" in region:
        return "✅"
    elif "Region 2" in region:
        return "⚠️"
    else:
        return "❌"


def print_summary_table(
    results: list[ConfigResult],
    noise_models: list[str],
    title: str = "Configuration Summary",
    show_circuit_info: bool = False,
    reference_noise_model: Optional[str] = None
) -> str:
    """
    Print a summary table comparing configurations across noise models.
    
    Args:
        results: List of ConfigResult objects
        noise_models: List of noise model names to include in table
        title: Table title
        show_circuit_info: Whether to show circuit depth/gates columns
        reference_noise_model: Noise model to use for region classification (defaults to first)
    
    Returns the table as a string.
    """
    # Group by config
    config_groups = group_by_config(results)
    
    # Sort configs by some logical order (qubits, then name)
    sorted_configs = sorted(config_groups.keys(), 
                           key=lambda c: (config_groups[c][0].num_qubits, c))
    
    # Default reference noise model to first in list
    if reference_noise_model is None and noise_models:
        reference_noise_model = noise_models[0]
    
    # Build header
    header_parts = ["Config", "Qubits", "Marked"]
    if show_circuit_info:
        header_parts.extend(["Depth", "Gates"])
    header_parts.extend(noise_models)
    header_parts.extend(["Random", "Region"])
    
    lines = []
    lines.append(f"\n## {title}\n")
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")
    
    for config_id in sorted_configs:
        config_results = config_groups[config_id]
        
        # Get basic info from first result
        first = config_results[0]
        
        # Build row
        row_parts = [
            f"**{config_id}**",
            str(first.num_qubits),
            str(first.num_marked),
        ]
        
        if show_circuit_info:
            depth = first.circuit_depth if first.circuit_depth else "-"
            gates = first.total_gates if first.total_gates else "-"
            row_parts.extend([str(depth), str(gates)])
        
        # Add success rates for each noise model
        for nm in noise_models:
            nm_result = next((r for r in config_results if r.noise_model == nm), None)
            if nm_result:
                row_parts.append(f"**{format_percentage(nm_result.mean_success_rate)}**")
            else:
                row_parts.append("-")
        
        # Get region from reference noise model specifically
        ref_result = next((r for r in config_results if r.noise_model == reference_noise_model), None)
        region_for_row = ref_result.region if ref_result else "Region 3"
        
        # Add random chance and region
        row_parts.append(format_percentage(first.random_chance))
        row_parts.append(f"{region_for_row} {region_symbol(region_for_row)}")
        
        lines.append("| " + " | ".join(row_parts) + " |")
    
    return "\n".join(lines)


def print_region_summary(results: list[ConfigResult], reference_noise_model: str) -> str:
    """Print summary of configs by region."""
    # Filter by reference noise model
    ref_results = [r for r in results if r.noise_model == reference_noise_model]
    
    # Group by region
    regions = {"Region 1": [], "Region 2": [], "Region 3": []}
    for r in ref_results:
        regions[r.region].append(r)
    
    lines = []
    lines.append(f"\n## Region Summary (using {reference_noise_model})\n")
    
    for region, configs in regions.items():
        symbol = region_symbol(region)
        count = len(configs)
        config_ids = ", ".join(sorted([c.config_id for c in configs]))
        
        if region == "Region 1":
            desc = "Signal Dominant - Worth running on QPU"
        elif region == "Region 2":
            desc = "Transition Zone - Marginal quantum advantage"
        else:
            desc = "Noise Dominant - Algorithm fails, use for documenting limits"
        
        lines.append(f"### {region} {symbol} ({count} configs)")
        lines.append(f"**{desc}**\n")
        if configs:
            lines.append(f"Configs: {config_ids}\n")
            
            # Show success rate range
            rates = [c.mean_success_rate for c in configs]
            lines.append(f"Success range: {format_percentage(min(rates))} - {format_percentage(max(rates))}\n")
    
    return "\n".join(lines)


def print_qpu_recommendations(results: list[ConfigResult], reference_noise_model: str) -> str:
    """Print QPU experiment recommendations."""
    ref_results = [r for r in results if r.noise_model == reference_noise_model]
    
    # Sort by region and then by success rate
    priority_order = {"Region 1": 1, "Region 2": 2, "Region 3": 3}
    sorted_results = sorted(ref_results, 
                           key=lambda r: (priority_order.get(r.region, 4), -r.mean_success_rate))
    
    lines = []
    lines.append(f"\n## QPU Experiment Recommendations\n")
    lines.append("Prioritized list of configurations to run on real QPU:\n")
    
    lines.append("| Priority | Config | Qubits | Expected Success | Region | Runs Suggested |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    
    for i, r in enumerate(sorted_results[:15], 1):  # Top 15
        runs = 5 if "Region 1" in r.region else (3 if "Region 2" in r.region else 2)
        lines.append(
            f"| {i} | {r.config_id} | {r.num_qubits} | "
            f"{format_percentage(r.mean_success_rate)} | {r.region} {region_symbol(r.region)} | {runs} |"
        )
    
    return "\n".join(lines)


def print_noise_model_comparison(results: list[ConfigResult]) -> str:
    """Compare performance across noise models."""
    by_noise = group_by_noise_model(results)
    
    lines = []
    lines.append("\n## Noise Model Comparison\n")
    lines.append("| Noise Model | Avg Success | Min | Max | Region 1 Count | Region 3 Count |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    
    noise_stats = []
    for nm, nm_results in by_noise.items():
        rates = [r.mean_success_rate for r in nm_results]
        r1_count = sum(1 for r in nm_results if "Region 1" in r.region)
        r3_count = sum(1 for r in nm_results if "Region 3" in r.region)
        avg_rate = sum(rates) / len(rates)
        noise_stats.append((nm, avg_rate, min(rates), max(rates), r1_count, r3_count))
    
    # Sort by average success rate descending
    noise_stats.sort(key=lambda x: x[1], reverse=True)
    
    for nm, avg, min_r, max_r, r1, r3 in noise_stats:
        lines.append(
            f"| {nm} | {format_percentage(avg)} | {format_percentage(min_r)} | "
            f"{format_percentage(max_r)} | {r1} | {r3} |"
        )
    
    return "\n".join(lines)


def export_to_json(results: list[ConfigResult], output_path: str):
    """Export analysis results to JSON."""
    export_data = []
    for r in results:
        export_data.append({
            "config_id": r.config_id,
            "noise_model": r.noise_model,
            "num_qubits": r.num_qubits,
            "num_marked": r.num_marked,
            "search_space": r.search_space,
            "random_chance": r.random_chance,
            "mean_success_rate": r.mean_success_rate,
            "std_success_rate": r.std_success_rate,
            "region": r.region,
            "quantum_advantage_ratio": r.quantum_advantage_ratio,
            "test_mode": r.test_mode,
            "period": r.period,
        })
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze quantum algorithm simulator data and classify into regions"
    )
    parser.add_argument(
        "input_file",
        help="Path to campaign summary JSON file"
    )
    parser.add_argument(
        "--algorithm", "-a",
        choices=["grover", "qft", "auto"],
        default="auto",
        help="Algorithm type (default: auto-detect from filename)"
    )
    parser.add_argument(
        "--noise-models", "-n",
        nargs="+",
        default=None,
        help="Noise models to include (default: all QPU-calibrated models)"
    )
    parser.add_argument(
        "--reference-noise", "-r",
        default="IBM-HERON-R2",
        help="Reference noise model for region classification (default: IBM-HERON-R2)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for analysis results"
    )
    parser.add_argument(
        "--region1-threshold",
        type=float,
        default=0.50,
        help="Success threshold for Region 1 (default: 0.50)"
    )
    parser.add_argument(
        "--advantage-multiplier",
        type=float,
        default=2.0,
        help="Multiplier for quantum advantage threshold (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    data = load_campaign_data(str(input_path))
    print(f"Loaded {len(data)} entries from {input_path.name}")
    
    # Auto-detect algorithm
    algorithm = args.algorithm
    if algorithm == "auto":
        if "grover" in str(input_path).lower():
            algorithm = "grover"
        elif "qft" in str(input_path).lower():
            algorithm = "qft"
        else:
            print("Warning: Could not auto-detect algorithm, defaulting to grover")
            algorithm = "grover"
    
    print(f"Algorithm: {algorithm.upper()}")
    
    # Default noise models (QPU-calibrated)
    default_qpu_noise = ["IBM-HERON-R3", "IBM-HERON-R2", "IBM-HERON-R1", "RIGETTI-ANKAA3"]
    noise_models = args.noise_models or default_qpu_noise
    
    # Filter to only noise models present in data
    available_noise = set(entry.get("noise_model", "") for entry in data)
    noise_models = [nm for nm in noise_models if nm in available_noise]
    print(f"Noise models: {', '.join(noise_models)}")
    
    # Configure region thresholds
    region_config = RegionConfig(
        region1_threshold=args.region1_threshold,
        advantage_multiplier=args.advantage_multiplier,
    )
    
    # Analyze data
    if algorithm == "grover":
        results = analyze_grover_data(data, noise_models, region_config)
    else:
        results = analyze_qft_data(data, noise_models, region_config)
    
    print(f"Analyzed {len(results)} configurations")
    
    # Print tables
    output_lines = []
    output_lines.append(f"# {algorithm.upper()} Region Analysis Report\n")
    output_lines.append(f"**Input file:** {input_path.name}")
    output_lines.append(f"**Reference noise model:** {args.reference_noise}")
    output_lines.append(f"**Region 1 threshold:** {args.region1_threshold}")
    output_lines.append(f"**Advantage multiplier:** {args.advantage_multiplier}x random chance\n")
    
    # Summary table
    table = print_summary_table(
        results, noise_models, 
        f"{algorithm.upper()} Configuration Summary",
        reference_noise_model=args.reference_noise
    )
    output_lines.append(table)
    
    # Region summary
    region_summary = print_region_summary(results, args.reference_noise)
    output_lines.append(region_summary)
    
    # Noise model comparison
    noise_comparison = print_noise_model_comparison(results)
    output_lines.append(noise_comparison)
    
    # QPU recommendations
    recommendations = print_qpu_recommendations(results, args.reference_noise)
    output_lines.append(recommendations)
    
    # Print to console
    full_output = "\n".join(output_lines)
    print(full_output)
    
    # Export to JSON if requested
    if args.output:
        export_to_json(results, args.output)
    
    # Also save markdown report
    report_path = input_path.parent / f"region_analysis_{input_path.stem}.md"
    with open(report_path, 'w') as f:
        f.write(full_output)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
