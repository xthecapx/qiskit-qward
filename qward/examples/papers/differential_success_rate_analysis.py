"""
Analyze DSR_result.csv and generate degradation plots.

Supports two data schemas (produced by differential_success_rate_experiment.py):
  Schema 1 – Grover / QFT: JSON with `individual_results` (unified format)
  Schema 2 – Teleportation: CSV with per-row histogram

Generates plots per algorithm, grouped by optimization level, showing DSR
variants vs scaling axes (num_qubits and transpiled_depth).
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

from qward.utils.styles import (
    COLORBREWER_PALETTE,
    TITLE_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    FIG_SIZE,
    MARKER_SIZE,
    MARKER_STYLES,
    apply_axes_defaults,
)

# =============================================================================
# DSR-specific Style Extensions
# =============================================================================

# Figure size for heatmaps (larger for readability)
FIG_SIZE_HEATMAP = (18, 14)

# DSR variant colors using ColorBrewer palette
DSR_VARIANT_COLORS = {
    "Michelson": COLORBREWER_PALETTE[1],  # Teal
    "Ratio": COLORBREWER_PALETTE[2],  # Orange
    "Log-Ratio": COLORBREWER_PALETTE[3],  # Purple
    "Norm-Margin": COLORBREWER_PALETTE[4],  # Pink
}

# Algorithm colors for boxplots
ALGORITHM_COLORS = {
    "GROVER": COLORBREWER_PALETTE[1],  # Teal
    "QFT": COLORBREWER_PALETTE[2],  # Orange
    "TELEPORTATION": COLORBREWER_PALETTE[3],  # Purple
}

# ---------------------------------------------------------------------------
# Algorithm-specific filter limits
# These keep plots focused on the ranges where DSR degradation is visible.
# ---------------------------------------------------------------------------

# Teleportation: restrict to small payload sizes & moderate depth
# DSR is near-zero above ~1k depth, so cap there for readable plots
TELEPORTATION_MAX_QUBITS = 3
TELEPORTATION_MAX_DEPTH = 1000

# Grover: no qubit or depth cap (full range for variant plots)
GROVER_MAX_QUBITS = None
GROVER_MAX_DEPTH = None

# QFT: no qubit or depth cap (per-algorithm plots use IBM data, full 2-10 range)
QFT_MAX_QUBITS = None
QFT_MAX_DEPTH = None

# Depth binning for readable boxplots (too many unique depths otherwise)
DEPTH_BIN_SIZE = 500  # Default bin size
DEPTH_BIN_SIZE_FINE = 250  # Finer bins for 0-1.5k range
DEPTH_BIN_SIZE_EXTRA_FINE = 50  # When all data is in 0-250, show 0-50, 50-100, ...
# When using extra-fine bins, cap display at this depth so plot focuses on useful range
DEPTH_DISPLAY_CAP_EXTRA_FINE = 150

# Outlier filter: high DSR at high depth is suspicious (teleportation only)
OUTLIER_DEPTH_THRESHOLD = 2000  # Depths above this
OUTLIER_DSR_MAX = 0.5  # DSR values above this at high depth are filtered

DSR_VARIANTS = [
    ("dsr_michelson", "Michelson", DSR_VARIANT_COLORS["Michelson"]),
    ("dsr_ratio", "Ratio", DSR_VARIANT_COLORS["Ratio"]),
    ("dsr_log_ratio", "Log-Ratio", DSR_VARIANT_COLORS["Log-Ratio"]),
    ("dsr_normalized_margin", "Norm-Margin", DSR_VARIANT_COLORS["Norm-Margin"]),
]


def _apply_plot_style():
    """Apply consistent plot styling using shared styles."""
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "axes.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
            "lines.linewidth": 3,
            "lines.markersize": 12,
        }
    )


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _filter_teleportation(rows: List[Dict[str, str]], algorithm: str) -> List[Dict[str, str]]:
    """
    Filter Teleportation rows to focus on the range with visible degradation.

    Applies limits on both qubits (payload size) and transpiled depth to
    keep all data points (including zeros) for honest visualization while
    focusing on the meaningful range. Also filters suspicious outliers
    (high DSR at high depth).
    """
    # Only apply filtering to TELEPORTATION
    if algorithm != "TELEPORTATION":
        return rows

    filtered = []
    for r in rows:
        if r.get("algorithm") != algorithm:
            continue
        qubits = _to_float(r.get("num_qubits", ""))
        depth = _to_float(r.get("transpiled_depth", ""))
        dsr = _to_float(r.get("dsr_michelson", ""))

        # Apply qubit and depth filters
        qubits_ok = qubits is not None and qubits <= TELEPORTATION_MAX_QUBITS
        depth_ok = depth is None or depth <= TELEPORTATION_MAX_DEPTH

        # Filter suspicious outliers: high DSR at high depth
        outlier_ok = True
        if depth is not None and dsr is not None:
            if depth > OUTLIER_DEPTH_THRESHOLD and dsr > OUTLIER_DSR_MAX:
                outlier_ok = False

        if qubits_ok and depth_ok and outlier_ok:
            filtered.append(r)

    return filtered


def _bin_depth(depth: float, bin_size: Optional[int] = None) -> int:
    """Bin depth value into a range for readable boxplots."""
    size = bin_size if bin_size is not None else DEPTH_BIN_SIZE
    return int((depth // size) * size)


def _filter_by_depth(
    rows: List[Dict[str, str]],
    algorithm: str,
    max_depth: Optional[float],
) -> List[Dict[str, str]]:
    """Filter rows for *algorithm* keeping only depth ≤ *max_depth*."""
    filtered = []
    for r in rows:
        if r.get("algorithm") != algorithm:
            continue
        if max_depth is not None:
            depth = _to_float(r.get("transpiled_depth", ""))
            if depth is not None and depth > max_depth:
                continue
        filtered.append(r)
    return filtered


def _filter_by_qubits(
    rows: List[Dict[str, str]], algorithm: str, max_qubits: Optional[int]
) -> List[Dict[str, str]]:
    """Filter rows for *algorithm* keeping only num_qubits ≤ *max_qubits*."""
    if max_qubits is None:
        return rows
    filtered = []
    for r in rows:
        if r.get("algorithm") != algorithm:
            continue
        nq = _to_float(r.get("num_qubits", ""))
        if nq is not None and nq <= max_qubits:
            filtered.append(r)
    return filtered


def _filter_algorithm(
    rows: List[Dict[str, str]], algorithm: str
) -> Tuple[List[Dict[str, str]], bool]:
    """
    Apply algorithm-specific filtering and return filtered rows and whether filtering was applied.
    Uses algorithm-specific max qubits (Grover 3, QFT 5) and depth caps for readable plots.
    """
    original_count = len([r for r in rows if r.get("algorithm") == algorithm])

    if algorithm == "TELEPORTATION":
        filtered = _filter_teleportation(rows, algorithm)
    elif algorithm == "GROVER":
        filtered = _filter_by_depth(rows, algorithm, GROVER_MAX_DEPTH)
        filtered = _filter_by_qubits(filtered, algorithm, GROVER_MAX_QUBITS)
    elif algorithm == "QFT":
        filtered = _filter_by_depth(rows, algorithm, QFT_MAX_DEPTH)
        filtered = _filter_by_qubits(filtered, algorithm, QFT_MAX_QUBITS)
    else:
        filtered = [r for r in rows if r.get("algorithm") == algorithm]

    is_filtered = len(filtered) < original_count
    return filtered, is_filtered


def _filter_description(algorithm: str) -> str:
    """Return a short human-readable description of the filter applied to *algorithm*."""
    if algorithm == "TELEPORTATION":
        return f"qubits ≤ {TELEPORTATION_MAX_QUBITS}, depth < {TELEPORTATION_MAX_DEPTH // 1000}k"
    if algorithm == "GROVER":
        parts = []
        if GROVER_MAX_QUBITS is not None:
            parts.append(f"qubits ≤ {GROVER_MAX_QUBITS}")
        if GROVER_MAX_DEPTH is not None:
            parts.append(f"depth < {GROVER_MAX_DEPTH / 1000:.1f}k".replace(".0k", "k"))
        return ", ".join(parts) if parts else ""
    if algorithm == "QFT":
        parts = []
        if QFT_MAX_QUBITS is not None:
            parts.append(f"qubits ≤ {QFT_MAX_QUBITS}")
        if QFT_MAX_DEPTH is not None:
            parts.append(f"depth < {QFT_MAX_DEPTH / 1000:.1f}k".replace(".0k", "k"))
        return ", ".join(parts) if parts else ""
    return ""


def _to_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def _group_points(
    rows: List[Dict[str, str]],
    algorithm: str,
    opt_level: str,
    x_key: str,
    variant_key: str,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        if row.get("optimization_level", "") != opt_level:
            continue
        x_val = _to_float(row.get(x_key, ""))
        y_val = _to_float(row.get(variant_key, ""))
        if x_val is None or y_val is None:
            continue
        points.append((x_val, y_val))
    return points


def _group_points_3d(
    rows: List[Dict[str, str]],
    algorithm: str,
    opt_level: str,
    variant_key: str,
) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        if row.get("optimization_level", "") != opt_level:
            continue
        x_val = _to_float(row.get("num_qubits", ""))
        y_val = _to_float(row.get("transpiled_depth", ""))
        z_val = _to_float(row.get(variant_key, ""))
        if x_val is None or y_val is None or z_val is None:
            continue
        points.append((x_val, y_val, z_val))
    return points


def _median_trend(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    buckets: Dict[float, List[float]] = {}
    for x, y in points:
        buckets.setdefault(x, []).append(y)
    trend = []
    for x in sorted(buckets.keys()):
        trend.append((x, statistics.median(buckets[x])))
    return trend


def _grid_from_points(
    points: List[Tuple[float, float, float]],
) -> Tuple[List[float], List[float], np.ndarray]:
    xs = sorted({x for x, _, _ in points})
    ys = sorted({y for _, y, _ in points})
    grid: Dict[Tuple[float, float], List[float]] = {}
    for x, y, z in points:
        grid.setdefault((x, y), []).append(z)
    z_mat = np.full((len(ys), len(xs)), np.nan)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            values = grid.get((x, y))
            if values:
                z_mat[yi, xi] = statistics.median(values)
    return xs, ys, z_mat


def _edges_from_centers(values: List[float]) -> List[float]:
    if len(values) == 1:
        center = values[0]
        return [center - 0.5, center + 0.5]
    mids = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return [first] + mids + [last]


def _has_optimization_levels(rows: List[Dict[str, str]], algorithm: str) -> bool:
    """Check if an algorithm has meaningful optimization levels."""
    opt_levels = {
        row.get("optimization_level", "") for row in rows if row.get("algorithm") == algorithm
    }
    return len(opt_levels) > 1 or (len(opt_levels) == 1 and "" not in opt_levels)


def _plot_algorithm_boxplot(
    rows: List[Dict[str, str]],
    algorithm: str,
    x_key: str,
    x_label: str,
    output_path: Path,
) -> None:
    """Create boxplot visualization for DSR by x_key, similar to good-plots.py style."""
    algo_rows = [r for r in rows if r.get("algorithm") == algorithm]
    if not algo_rows:
        return

    # Apply algorithm-specific filtering
    algo_rows, is_filtered = _filter_algorithm(rows, algorithm)

    if not algo_rows:
        return

    # For depth, bin values into ranges for readable boxplots (algorithm-specific)
    use_binning = x_key == "transpiled_depth"
    depth_bin_size = DEPTH_BIN_SIZE
    depth_axis_max_cap: Optional[float] = None  # algorithm-specific cap for x-axis
    if use_binning and algo_rows:
        raw_depths = [
            _to_float(r.get("transpiled_depth", ""))
            for r in algo_rows
            if _to_float(r.get("transpiled_depth", "")) is not None
        ]
        if raw_depths:
            max_d = max(raw_depths)
            if algorithm == "GROVER":
                # Grover (max 3q): shallow circuits; fine bins, cap axis so no empty high bins
                depth_bin_size = 100
                depth_axis_max_cap = min(500, max(400, max_d + 100))
            elif algorithm == "QFT":
                # QFT (max 5q): data often 0-300; extra-fine or fine, cap to useful range
                if max_d <= 300:
                    depth_bin_size = DEPTH_BIN_SIZE_EXTRA_FINE  # 50
                    depth_axis_max_cap = min(300, max_d + 100)
                elif max_d < 600:
                    depth_bin_size = 100
                    depth_axis_max_cap = min(600, max_d + 100)
                else:
                    depth_bin_size = DEPTH_BIN_SIZE_FINE  # 250
                    depth_axis_max_cap = min(1500, max_d + 250)
            else:
                if max_d <= 300:
                    depth_bin_size = DEPTH_BIN_SIZE_EXTRA_FINE
                elif max_d < 1500:
                    depth_bin_size = DEPTH_BIN_SIZE_FINE

    # Collect data grouped by x value (or binned x value)
    grouped_data: Dict[float, List[float]] = {}
    for r in algo_rows:
        x_val = _to_float(r.get(x_key, ""))
        dsr_val = _to_float(r.get("dsr_michelson", ""))
        if x_val is None or dsr_val is None:
            continue

        # Apply binning for depth
        if use_binning:
            x_val = float(_bin_depth(x_val, depth_bin_size))

        if x_val not in grouped_data:
            grouped_data[x_val] = []
        grouped_data[x_val].append(dsr_val)

    if not grouped_data:
        return

    # Sort by x value and prepare for plotting
    x_values = sorted(grouped_data.keys())

    # For depth binning: algorithm-specific axis cap to avoid empty high-depth bins
    if use_binning and raw_depths:
        max_depth_val = max(raw_depths)
        if depth_axis_max_cap is not None:
            # Grover/QFT: use algorithm-specific cap so groups match data
            depth_axis_max = depth_axis_max_cap
            num_bins = max(4, int(depth_axis_max // depth_bin_size))
        elif (
            depth_bin_size == DEPTH_BIN_SIZE_EXTRA_FINE and DEPTH_DISPLAY_CAP_EXTRA_FINE is not None
        ):
            depth_axis_max = DEPTH_DISPLAY_CAP_EXTRA_FINE
            num_bins = int(depth_axis_max // depth_bin_size)
        else:
            num_bins = max(4, int(max_depth_val // depth_bin_size) + 1)
            depth_axis_max = num_bins * depth_bin_size
        all_bin_starts = [i * depth_bin_size for i in range(num_bins)]
        # Use full range for positions/labels; only bins with data get a box
        positions_for_axis = all_bin_starts
        box_data = [grouped_data.get(float(b), []) for b in all_bin_starts]
        positions = [b for b in all_bin_starts if grouped_data.get(float(b))]
        # Filter to boxes that have data for the actual boxplot
        box_data_nonempty = [grouped_data[float(p)] for p in positions]
    else:
        positions_for_axis = x_values
        box_data_nonempty = [grouped_data[x_val] for x_val in x_values]
        positions = x_values
        depth_axis_max = None

    if not box_data_nonempty:
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Calculate appropriate box width (half width for clearer dispersion brackets)
    if use_binning and depth_axis_max is not None:
        box_width = depth_bin_size * 0.25
    elif len(positions) > 1:
        spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        min_spacing = min(spacings)
        x_range = max(positions) - min(positions)
        box_width = min(min_spacing * 0.35, x_range * 0.075)
    else:
        box_width = depth_bin_size * 0.25 if use_binning else 0.2

    # Create boxplot with good-plots.py style (only non-empty bins)
    color = ALGORITHM_COLORS.get(algorithm, COLORBREWER_PALETTE[1])
    bp = ax.boxplot(
        box_data_nonempty,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.7),
        medianprops=dict(color="black", linewidth=3),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        flierprops=dict(
            marker="o", markerfacecolor="black", markersize=10, alpha=0.8, markeredgecolor="black"
        ),
    )

    # Also overlay median trend line
    medians = [statistics.median(d) for d in box_data_nonempty]
    ax.plot(
        positions,
        medians,
        "o-",
        color="black",
        linewidth=3,
        markersize=14,
        label="Median DSR",
        zorder=5,
    )

    # Styling
    display_label = f"{x_label} (binned)" if use_binning else x_label
    ax.set_xlabel(display_label, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("DSR", fontsize=LABEL_SIZE, fontweight="bold")

    apply_axes_defaults(ax)

    # Tighten y-axis to data range to reduce blank space (keep [0, 1] if data spans it)
    all_dsr = [v for d in box_data_nonempty for v in d]
    if all_dsr:
        y_min_data = min(all_dsr)
        y_max_data = max(all_dsr)
        padding = 0.08
        ax.set_ylim(
            max(0.0, y_min_data - padding),
            min(1.0, y_max_data + padding),
        )
    else:
        ax.set_ylim(-0.05, 1.05)

    # Set x-axis: for depth boxplot show only bins with data (no empty bins = less blank space)
    if use_binning and depth_axis_max is not None:
        # Use only positions that have data so empty bins don't create blank space
        ax.set_xticks(positions)
        labels = []
        for p in positions:
            start = int(p)
            end = int(p + depth_bin_size)
            if start >= 1000:
                start_str = f"{start/1000:.1f}k".replace(".0k", "k")
            else:
                start_str = str(start)
            if end >= 1000:
                end_str = f"{end/1000:.1f}k".replace(".0k", "k")
            else:
                end_str = str(end)
            labels.append(f"{start_str}-{end_str}")
        # Fewer bins = easier to read; use smaller rotation and full tick size
        rotation = 25 if len(positions) <= 4 else 45
        ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=TICK_SIZE)
        x_pad = depth_bin_size * 0.25
        ax.set_xlim(min(positions) - x_pad, max(positions) + x_pad)
    else:
        ax.set_xticks(positions)
        if use_binning:
            labels = []
            for p in positions:
                start = int(p)
                end = int(p + depth_bin_size)
                if start >= 1000:
                    start_str = f"{start/1000:.1f}k".replace(".0k", "k")
                else:
                    start_str = str(start)
                if end >= 1000:
                    end_str = f"{end/1000:.1f}k".replace(".0k", "k")
                else:
                    end_str = str(end)
                labels.append(f"{start_str}-{end_str}")
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE - 4)
        else:
            ax.set_xticklabels([f"{int(p)}" for p in positions])
        x_padding = (max(positions) - min(positions)) * 0.1 if len(positions) > 1 else 0.5
        ax.set_xlim(min(positions) - x_padding, max(positions) + x_padding)

    # Legend
    legend_elements = [
        Patch(facecolor=color, alpha=0.7, label=f"{algorithm} DSR"),
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_algorithm_line(
    rows: List[Dict[str, str]],
    algorithm: str,
    x_key: str,
    x_label: str,
    output_path: Path,
    figsize=None,
) -> None:
    """Create line plot with scatter showing all DSR variants."""
    algo_rows = [r for r in rows if r.get("algorithm") == algorithm]
    if not algo_rows:
        return

    # Apply algorithm-specific filtering
    algo_rows, is_filtered = _filter_algorithm(rows, algorithm)

    if not algo_rows:
        return

    fig, ax = plt.subplots(figsize=figsize or FIG_SIZE)

    for idx, (variant_key, label, color) in enumerate(DSR_VARIANTS):
        # Get all points
        points = []
        for r in algo_rows:
            x_val = _to_float(r.get(x_key, ""))
            y_val = _to_float(r.get(variant_key, ""))
            if x_val is not None and y_val is not None:
                points.append((x_val, y_val))

        if not points:
            continue

        xs, ys = zip(*points)

        # Scatter with transparency
        marker = MARKER_STYLES.get(idx + 1, "o")
        ax.scatter(
            xs,
            ys,
            s=MARKER_SIZE,
            color=color,
            alpha=0.4,
            edgecolors="none",
            marker=marker,
            label=None,
        )

        # Median trend line
        trend = _median_trend(points)
        tx, ty = zip(*trend)
        ax.plot(
            tx,
            ty,
            marker=marker,
            linestyle="-",
            color=color,
            linewidth=3,
            markersize=14,
            label=label,
            zorder=5,
        )

    # Styling
    ax.set_xlabel(x_label, fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("DSR", fontsize=LABEL_SIZE, fontweight="bold")

    # Add note if data was filtered
    apply_axes_defaults(ax)
    ax.set_ylim(-0.05, 1.05)

    ax.legend(fontsize=LEGEND_SIZE, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _create_log_colormap():
    """Create a colormap that emphasizes low values (for Michelson)."""
    # Use a power-law normalization to spread out low values
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    return mcolors.LinearSegmentedColormap.from_list("viridis_enhanced", colors)


def _plot_algorithm_heatmap(
    rows: List[Dict[str, str]],
    algorithm: str,
    variant_key: str,
    variant_label: str,
    output_dir: Path,
    use_log_scale: bool = False,
) -> None:
    """Create improved heatmap with better visibility."""
    algo_rows = [r for r in rows if r.get("algorithm") == algorithm]
    if not algo_rows:
        return

    # Apply algorithm-specific filtering
    algo_rows, is_filtered = _filter_algorithm(rows, algorithm)

    if not algo_rows:
        return

    # Collect all points (ignore optimization level for aggregation)
    points = []
    for r in algo_rows:
        x_val = _to_float(r.get("num_qubits", ""))
        y_val = _to_float(r.get("transpiled_depth", ""))
        z_val = _to_float(r.get(variant_key, ""))
        if x_val is not None and y_val is not None and z_val is not None:
            points.append((x_val, y_val, z_val))

    if len(points) < 4:
        return

    xs_grid, ys_grid, z_mat = _grid_from_points(points)
    if len(xs_grid) < 2 or len(ys_grid) < 2 or np.all(np.isnan(z_mat)):
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE_HEATMAP)

    x_edges = _edges_from_centers(xs_grid)
    y_edges = _edges_from_centers(ys_grid)

    # Use power normalization for better visibility of low values
    if use_log_scale or variant_key == "dsr_michelson":
        # Power normalization spreads out low values
        norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            z_mat,
            cmap="viridis",
            shading="auto",
            norm=norm,
        )
    else:
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            z_mat,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=1.0,
        )

    # Add value annotations on each cell with larger font
    annotation_size = max(12, TICK_SIZE - 8)  # Scale with tick size but not too small
    for yi, y in enumerate(ys_grid):
        for xi, x in enumerate(xs_grid):
            val = z_mat[yi, xi]
            if not np.isnan(val):
                # Choose text color based on background
                text_color = "white" if val < 0.5 else "black"
                ax.text(
                    x,
                    y,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    fontweight="bold",
                    color=text_color,
                )

    # Styling
    ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Transpiled Depth", fontsize=LABEL_SIZE, fontweight="bold")

    # Build title with notes
    notes = []
    if is_filtered:
        desc = _filter_description(algorithm)
        if desc:
            notes.append(desc)
    if use_log_scale or variant_key == "dsr_michelson":
        notes.append("enhanced contrast")
    title_suffix = f" ({', '.join(notes)})" if notes else ""

    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    ax.set_xticks(xs_grid)
    ax.set_xticklabels([f"{int(x)}" for x in xs_grid])

    # Colorbar with larger font
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_label("DSR", fontsize=LABEL_SIZE, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{algorithm.lower()}_{variant_key}_heatmap.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Per-algorithm qubit limits for the combined comparison plot.
# These match the ranges shown in the individual per-algorithm boxplots
# and are intentionally different from the AWS-specific constants above.
_COMBINED_MAX_QUBITS = {
    "GROVER": None,  # show all (2-8 in IBM data)
    "QFT": None,  # show all (2-10 in IBM data)
    "TELEPORTATION": TELEPORTATION_MAX_QUBITS,  # 3
}
_COMBINED_MAX_DEPTH = {
    "GROVER": GROVER_MAX_DEPTH,  # 5000
    "QFT": QFT_MAX_DEPTH,  # None
    "TELEPORTATION": TELEPORTATION_MAX_DEPTH,  # 1000
}


def _prepare_combined_data(
    rows: List[Dict[str, str]],
    optimization_level: Optional[str] = "3",
) -> Tuple[List[str], List[float], Dict[str, List[Dict[str, str]]]]:
    """Extract per-algorithm filtered data and the union of qubit counts.

    Uses combined-plot-specific limits (no qubit cap for Grover/QFT) so the
    plot matches the individual per-algorithm boxplots.

    Args:
        rows: All loaded CSV rows.
        optimization_level: If set, keep only rows with this optimization
            level.  Pass ``None`` to include all levels.

    Returns:
        (algorithms, qubits, filtered_data) where *filtered_data* maps each
        algorithm name to its filtered row list.
    """
    algorithms = sorted({r.get("algorithm", "") for r in rows if r.get("algorithm")})

    filtered_data: Dict[str, List[Dict[str, str]]] = {}
    all_qubits: set = set()
    for algo in algorithms:
        algo_rows = [r for r in rows if r.get("algorithm") == algo]
        # Filter by optimization level (skip for algorithms without opt data)
        if optimization_level is not None:
            has_opt = any(r.get("optimization_level", "") != "" for r in algo_rows)
            if has_opt:
                algo_rows = [
                    r for r in algo_rows if r.get("optimization_level", "") == optimization_level
                ]
        # Apply depth filter
        max_depth = _COMBINED_MAX_DEPTH.get(algo)
        if max_depth is not None:
            algo_rows = _filter_by_depth(algo_rows, algo, max_depth)
        # Apply qubit filter
        max_q = _COMBINED_MAX_QUBITS.get(algo)
        if max_q is not None:
            algo_rows = _filter_by_qubits(algo_rows, algo, max_q)
        # Teleportation outlier filter
        if algo == "TELEPORTATION":
            algo_rows = _filter_teleportation(algo_rows, algo)

        filtered_data[algo] = algo_rows
        for r in algo_rows:
            q = _to_float(r.get("num_qubits", ""))
            if q is not None:
                all_qubits.add(q)

    qubits = sorted(all_qubits)
    return algorithms, qubits, filtered_data


def _render_combined_comparison(
    ax,
    algorithms: List[str],
    qubits: List[float],
    filtered_data: Dict[str, List[Dict[str, str]]],
) -> None:
    """Render the combined DSR comparison boxplot onto *ax*.

    Draws one boxplot per algorithm per qubit group, with alternating
    background shading and dashed separators between qubit groups for
    readability.
    """
    if not qubits:
        return

    n_algos = len(algorithms)
    width = 0.25

    # Build a continuous integer range from min to max qubit so that
    # gaps (e.g. 7, 9) still receive shading and separators.
    q_min = int(min(qubits))
    q_max = int(max(qubits))
    full_range = list(range(q_min, q_max + 1))

    # Alternating background shading over the full range
    for idx, q in enumerate(full_range):
        left = q - 0.5
        right = q + 0.5
        if idx % 2 == 0:
            ax.axvspan(left, right, color="#f0f0f0", zorder=0)

    # Dashed vertical separators between every consecutive integer
    for idx in range(len(full_range) - 1):
        mid = (full_range[idx] + full_range[idx + 1]) / 2
        ax.axvline(mid, color="#cccccc", linestyle="--", linewidth=1, zorder=1)

    # Draw boxplots per algorithm
    for i, algo in enumerate(algorithms):
        algo_rows = filtered_data[algo]

        box_data = []
        positions = []

        for q in qubits:
            dsr_values = [
                _to_float(r.get("dsr_michelson", ""))
                for r in algo_rows
                if _to_float(r.get("num_qubits", "")) == q
            ]
            dsr_values = [v for v in dsr_values if v is not None]
            if dsr_values:
                box_data.append(dsr_values)
                positions.append(q + (i - n_algos / 2 + 0.5) * width)

        if box_data:
            color = ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8])
            ax.boxplot(
                box_data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=3),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
            )

    # Styling
    ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("DSR", fontsize=LABEL_SIZE, fontweight="bold")
    apply_axes_defaults(ax)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(full_range)
    ax.set_xticklabels([str(q) for q in full_range])
    ax.set_xlim(q_min - 0.5, q_max + 0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8]), alpha=0.7, label=algo)
        for algo in algorithms
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper right")


def _plot_combined_comparison(
    rows: List[Dict[str, str]],
    output_dir: Path,
) -> None:
    """Create a combined comparison plot of all algorithms."""
    algorithms, qubits, filtered_data = _prepare_combined_data(rows)

    if not qubits:
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    _render_combined_comparison(ax, algorithms, qubits, filtered_data)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "1_combined_dsr_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


_COMBINED_DEPTH_BIN = 200  # bin width for the depth-based combined boxplot
_COMBINED_DEPTH_CAP = 800  # drop data above this depth to avoid long empty tail


def _plot_combined_depth_comparison(
    rows: List[Dict[str, str]],
    output_dir: Path,
) -> None:
    """Combined DSR boxplot: all algorithms, x-axis = transpiled depth (binned).

    Mirrors ``_plot_combined_comparison`` but groups by depth bins instead of
    qubit count.  Includes **all** optimization levels (depth itself is the
    variable under study; different opt levels produce different depths for
    the same circuit).  Uses 200-unit bins capped at 800 for readability.
    """
    # Use optimization_level=None to include all levels
    algorithms, _qubits, filtered_data = _prepare_combined_data(rows, optimization_level=None)

    # Bin depths and collect DSR values per (algo, bin_start)
    bin_sz = _COMBINED_DEPTH_BIN
    cap = _COMBINED_DEPTH_CAP

    algo_binned: Dict[str, Dict[int, List[float]]] = {}
    all_bins: set = set()
    for algo in algorithms:
        binned: Dict[int, List[float]] = {}
        for r in filtered_data[algo]:
            d = _to_float(r.get("transpiled_depth", ""))
            dsr = _to_float(r.get("dsr_michelson", ""))
            if d is None or dsr is None or d > cap:
                continue
            b = int((d // bin_sz) * bin_sz)
            binned.setdefault(b, []).append(dsr)
        algo_binned[algo] = binned
        all_bins.update(binned.keys())

    if not all_bins:
        return

    bins_sorted = sorted(all_bins)
    n_algos = len(algorithms)
    width = 0.25

    # Map bin starts to integer positions (0, 1, 2, …) for compact x-axis
    pos_map = {b: i for i, b in enumerate(bins_sorted)}

    fig, ax = plt.subplots(figsize=(15, 6))

    # Alternating background shading
    for idx in range(len(bins_sorted)):
        left = idx - 0.5
        right = idx + 0.5
        if idx % 2 == 0:
            ax.axvspan(left, right, color="#f0f0f0", zorder=0)

    # Dashed separators
    for idx in range(len(bins_sorted) - 1):
        mid = idx + 0.5
        ax.axvline(mid, color="#cccccc", linestyle="--", linewidth=1, zorder=1)

    # Draw boxplots per algorithm
    for i, algo in enumerate(algorithms):
        binned = algo_binned[algo]

        box_data = []
        positions = []
        for b in bins_sorted:
            vals = binned.get(b, [])
            if vals:
                box_data.append(vals)
                pos = pos_map[b] + (i - n_algos / 2 + 0.5) * width
                positions.append(pos)

        if box_data:
            color = ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8])
            ax.boxplot(
                box_data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=3),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
            )

    # X-axis: bin labels
    labels = [f"{b}\u2013{b + bin_sz}" for b in bins_sorted]
    ax.set_xticks(range(len(bins_sorted)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(bins_sorted) - 0.5)

    ax.set_xlabel("Depth", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("DSR", fontsize=LABEL_SIZE, fontweight="bold")
    apply_axes_defaults(ax)
    ax.set_ylim(-0.05, 1.05)

    # Legend
    legend_elements = [
        Patch(
            facecolor=ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8]),
            alpha=0.7,
            label=algo,
        )
        for algo in algorithms
    ]
    ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper right")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "1_combined_dsr_comparison_depth.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _plot_qft_heatmap_by_optimization(
    rows: List[Dict[str, str]],
    output_dir: Path,
) -> None:
    """Heatmap of median QFT DSR: optimization level (y) vs qubits (x)."""
    from matplotlib.colors import LinearSegmentedColormap

    qft_rows = [r for r in rows if r.get("algorithm") == "QFT"]
    if not qft_rows:
        return

    opt_levels = sorted(
        {r.get("optimization_level", "") for r in qft_rows if r.get("optimization_level")}
    )
    if not opt_levels:
        return

    # Collect data per optimization level
    grouped_all: Dict[str, Dict[float, List[float]]] = {}
    all_xs: set = set()
    for ol in opt_levels:
        opt_rows = [r for r in qft_rows if r.get("optimization_level", "") == ol]
        grouped: Dict[float, List[float]] = {}
        for r in opt_rows:
            x = _to_float(r.get("num_qubits", ""))
            dsr = _to_float(r.get("dsr_michelson", ""))
            if x is not None and dsr is not None:
                grouped.setdefault(x, []).append(dsr)
        grouped_all[ol] = grouped
        all_xs.update(grouped.keys())

    q_min = int(min(all_xs))
    q_max = int(max(all_xs))
    full_range = list(range(q_min, q_max + 1))

    # Build matrix: rows = opt levels, cols = qubits
    matrix = np.full((len(opt_levels), len(full_range)), np.nan)
    for i, ol in enumerate(opt_levels):
        grouped = grouped_all[ol]
        for j, q in enumerate(full_range):
            fq = float(q)
            if fq in grouped:
                matrix[i, j] = np.median(grouped[fq])

    # Paired 4-class palette as custom colormap
    paired_colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]
    cmap = LinearSegmentedColormap.from_list("paired4", paired_colors, N=256)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")

    # Annotate cells with values
    for i in range(len(opt_levels)):
        for j in range(len(full_range)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.35 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=LABEL_SIZE - 2,
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_xticks(range(len(full_range)))
    ax.set_xticklabels([str(q) for q in full_range])
    ax.set_yticks(range(len(opt_levels)))
    ax.set_yticklabels([ol for ol in opt_levels])
    ax.set_xlabel("Number of Qubits", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Optimization", fontsize=LABEL_SIZE, fontweight="bold")
    apply_axes_defaults(ax)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DSR", fontsize=LABEL_SIZE, fontweight="bold")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "1_combined_dsr_qft_heatmap_optimization.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _render_region_on_ax(
    ax: plt.Axes,
    algo_rows: List[Dict[str, str]],
    opt_level: str,
    algorithms: List[str],
    filtered_data: Dict[str, List[Dict[str, str]]],
    x_key: str,
    use_binning: bool = False,
    max_x: Optional[float] = None,
) -> set:
    """
    Render DSR region bands for all algorithms onto a single Axes.

    Returns the set of x-values plotted (for tick configuration).
    """
    all_xs = set()

    for algo in algorithms:
        algo_rows = filtered_data[algo]

        opt_rows = [
            r
            for r in algo_rows
            if r.get("optimization_level", "") == opt_level or r.get("optimization_level", "") == ""
        ]

        if not opt_rows:
            continue

        color = ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8])

        # Collect DSR values grouped by x
        grouped = {}
        for r in opt_rows:
            x = _to_float(r.get(x_key, ""))
            dsr = _to_float(r.get("dsr_michelson", ""))
            if x is None or dsr is None:
                continue
            if max_x is not None and x > max_x:
                continue
            if use_binning:
                x = float(_bin_depth(x))
            if x not in grouped:
                grouped[x] = []
            grouped[x].append(dsr)

        if not grouped:
            continue

        xs = sorted(grouped.keys())
        all_xs.update(xs)

        medians = np.array([np.median(grouped[x]) for x in xs])
        q25s = np.array([np.percentile(grouped[x], 25) for x in xs])
        q75s = np.array([np.percentile(grouped[x], 75) for x in xs])
        q10s = np.array([np.percentile(grouped[x], 10) for x in xs])
        q90s = np.array([np.percentile(grouped[x], 90) for x in xs])
        xs_arr = np.array(xs)

        ax.fill_between(xs_arr, q10s, q90s, color=color, alpha=0.15)
        ax.fill_between(xs_arr, q25s, q75s, color=color, alpha=0.4)
        ax.plot(
            xs_arr,
            medians,
            color=color,
            linewidth=3,
            linestyle="-",
            marker="o",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
        )

    return all_xs


# Display names for legend (shared across region plots)
ALGO_DISPLAY_NAMES = {
    "GROVER": "Grover",
    "QFT": "QFT",
    "TELEPORTATION": "vTP",
}


def _format_depth_tick(value: float) -> str:
    """Format a depth value for tick labels (e.g., 1500 -> '1.5k')."""
    if value >= 1000:
        s = f"{value/1000:.1f}k"
        return s.replace(".0k", "k")
    return str(int(value))


def _plot_combined_regions_by_optimization(
    rows: List[Dict[str, str]],
    output_dir: Path,
    x_key: str = "num_qubits",
    x_label: str = "Number of Qubits",
    file_suffix: str = "",
    max_x: Optional[float] = None,
) -> None:
    """
    Create DSR region plots, one per optimization level.

    This visualization shows shaded regions representing percentile bands
    for each algorithm's DSR decay, separated by optimization level for clarity.
    """
    algorithms = sorted({r.get("algorithm", "") for r in rows if r.get("algorithm")})
    opt_levels = sorted(
        {r.get("optimization_level", "") for r in rows if r.get("optimization_level")}
    )

    use_binning = x_key == "transpiled_depth"

    # Apply algorithm-specific filtering
    filtered_data = {}
    for algo in algorithms:
        algo_rows, _ = _filter_algorithm(rows, algo)
        filtered_data[algo] = algo_rows

    for opt_level in opt_levels:
        fig, ax = plt.subplots(figsize=(12, 8))

        all_xs = _render_region_on_ax(
            ax,
            [],
            opt_level,
            algorithms,
            filtered_data,
            x_key,
            use_binning,
            max_x,
        )

        if not all_xs:
            plt.close(fig)
            continue

        xs = sorted(all_xs)

        display_label = f"{x_label} (binned)" if use_binning else x_label
        ax.set_xlabel(display_label, fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_ylabel("DSR", fontsize=LABEL_SIZE, fontweight="bold")

        apply_axes_defaults(ax)
        ax.set_ylim(-0.05, 1.05)

        ax.set_xticks(xs)
        if use_binning:
            ax.set_xticklabels([_format_depth_tick(x) for x in xs], rotation=45, ha="right")
        else:
            ax.set_xticklabels([f"{int(x)}" for x in xs])

        x_padding = (max(xs) - min(xs)) * 0.08 if len(xs) > 1 else 0.5
        ax.set_xlim(min(xs) - x_padding, max(xs) + x_padding)

        legend_elements = [
            Patch(
                facecolor=ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8]),
                alpha=0.6,
                label=ALGO_DISPLAY_NAMES.get(algo, algo),
            )
            for algo in algorithms
        ]
        ax.legend(handles=legend_elements, fontsize=LEGEND_SIZE, loc="upper right", framealpha=0.9)

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"combined_dsr_regions{file_suffix}_opt{opt_level}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def _plot_combined_regions_grid(
    rows: List[Dict[str, str]],
    output_dir: Path,
    x_key: str = "num_qubits",
    x_label: str = "Number of Qubits",
    file_suffix: str = "",
    max_x: Optional[float] = None,
) -> None:
    """
    Create a single 2x2 figure with all optimization levels side by side.
    """
    algorithms = sorted({r.get("algorithm", "") for r in rows if r.get("algorithm")})
    opt_levels = sorted(
        {r.get("optimization_level", "") for r in rows if r.get("optimization_level")}
    )

    if not opt_levels:
        return

    use_binning = x_key == "transpiled_depth"

    filtered_data = {}
    for algo in algorithms:
        algo_rows, _ = _filter_algorithm(rows, algo)
        filtered_data[algo] = algo_rows

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for idx, opt_level in enumerate(opt_levels[:4]):
        ax = axes_flat[idx]

        all_xs = _render_region_on_ax(
            ax,
            [],
            opt_level,
            algorithms,
            filtered_data,
            x_key,
            use_binning,
            max_x,
        )

        if not all_xs:
            continue

        xs = sorted(all_xs)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(xs)
        if use_binning:
            ax.set_xticklabels(
                [_format_depth_tick(x) for x in xs], rotation=45, ha="right", fontsize=TICK_SIZE - 4
            )
        else:
            ax.set_xticklabels([f"{int(x)}" for x in xs], fontsize=TICK_SIZE - 2)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 2)

        x_padding = (max(xs) - min(xs)) * 0.08 if len(xs) > 1 else 0.5
        ax.set_xlim(min(xs) - x_padding, max(xs) + x_padding)

        apply_axes_defaults(ax)

        display_label = f"{x_label} (binned)" if use_binning else x_label
        if idx >= 2:
            ax.set_xlabel(display_label, fontsize=LABEL_SIZE - 2, fontweight="bold")
        if idx % 2 == 0:
            ax.set_ylabel("DSR", fontsize=LABEL_SIZE - 2, fontweight="bold")

    legend_elements = [
        Patch(
            facecolor=ALGORITHM_COLORS.get(algo, COLORBREWER_PALETTE[8]),
            alpha=0.6,
            label=ALGO_DISPLAY_NAMES.get(algo, algo),
        )
        for algo in algorithms
    ]
    fig.legend(
        handles=legend_elements,
        fontsize=LEGEND_SIZE,
        loc="upper center",
        ncol=len(algorithms),
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.98),
    )

    title_axis = "Qubits" if x_key == "num_qubits" else "Circuit Depth"
    fig.suptitle(
        f"DSR vs {title_axis} Across Optimization Levels",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"combined_dsr_regions{file_suffix}_grid.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _compute_algorithm_stats(
    rows: List[Dict[str, str]],
    algorithm: str,
) -> Dict[str, any]:
    """Compute summary statistics for an algorithm."""
    algo_rows = [r for r in rows if r.get("algorithm") == algorithm]
    if not algo_rows:
        return {}

    has_opt = _has_optimization_levels(rows, algorithm)
    opt_levels = sorted({r.get("optimization_level", "") for r in algo_rows})

    stats = {
        "count": len(algo_rows),
        "has_optimization_levels": has_opt,
        "optimization_levels": opt_levels,
        "variants": {},
    }

    for variant_key, label, _ in DSR_VARIANTS:
        values = [_to_float(r.get(variant_key, "")) for r in algo_rows]
        values = [v for v in values if v is not None]
        if values:
            stats["variants"][label] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }

    peak_mismatches = sum(1 for r in algo_rows if r.get("peak_mismatch", "") == "True")
    stats["peak_mismatch_rate"] = peak_mismatches / len(algo_rows) if algo_rows else 0

    return stats


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_csv = repo_root / "examples" / "papers" / "DSR_result.csv"
    default_out = repo_root / "examples" / "papers" / "plots"

    parser = argparse.ArgumentParser(
        description="Analyze DSR_result.csv and plot degradation trends."
    )
    parser.add_argument("--input", type=Path, default=default_csv)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        metavar="N",
        help="Keep only rows with num_qubits < N (e.g. 4 for 2 and 3 qubits only)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        metavar="D",
        help="Keep only rows with transpiled_depth < D (e.g. 1000 for depth < 1k)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["ibm", "aws", "all"],
        help="Filter by provider: ibm, aws, or all (default: all)",
    )
    args = parser.parse_args()

    _apply_plot_style()
    rows = _load_rows(args.input)

    # Provider filter — match execution_type column
    if args.provider and args.provider != "all":
        _PROVIDER_PREFIXES = {
            "ibm": ("IBM", "SIMULATION", "QBRAID"),
            "aws": ("AWS",),
        }
        prefixes = _PROVIDER_PREFIXES.get(args.provider, ())
        before = len(rows)
        rows = [r for r in rows if any(r.get("execution_type", "").startswith(p) for p in prefixes)]
        print(f"Filter provider={args.provider}: {before} -> {len(rows)} rows")

    # Optional global filters (e.g. for AWS Rigetti: focus on 2q/3q and depth < 1k)
    if args.max_qubits is not None:
        before = len(rows)
        rows = [
            r
            for r in rows
            if _to_float(r.get("num_qubits", "")) is not None
            and _to_float(r.get("num_qubits", "")) < args.max_qubits
        ]
        print(f"Filter num_qubits < {args.max_qubits}: {before} -> {len(rows)} rows")
    if args.max_depth is not None:
        before = len(rows)
        rows = [
            r
            for r in rows
            if _to_float(r.get("transpiled_depth", "")) is None
            or _to_float(r.get("transpiled_depth", "")) < args.max_depth
        ]
        print(f"Filter transpiled_depth < {args.max_depth}: {before} -> {len(rows)} rows")

    algorithms = sorted({row.get("algorithm", "") for row in rows})
    if not algorithms:
        print("No algorithms found in input.")
        return 1

    # Per-algorithm plots use IBM-only data (their captions reference IBM
    # backends).  Combined comparison plots use all rows (their captions
    # explicitly say "IBM and Rigetti").
    _IBM_PREFIXES = ("IBM", "SIMULATION", "QBRAID")
    ibm_rows = [
        r for r in rows if any(r.get("execution_type", "").startswith(p) for p in _IBM_PREFIXES)
    ]

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"  IBM rows: {len(ibm_rows)}  (used for per-algorithm plots)")
    print(f"  All rows: {len(rows)}  (used for combined plots)")
    print(f"Algorithms found: {[a for a in algorithms if a]}")
    print()

    for algorithm in algorithms:
        if algorithm == "":
            continue

        # Generate boxplot (primary visualization)
        _plot_algorithm_boxplot(
            ibm_rows,
            algorithm,
            "num_qubits",
            "Number of Qubits",
            args.out_dir / f"{algorithm.lower()}_dsr_boxplot_qubits.png",
        )

        # Generate line plot with all variants (rectangular for column layout)
        _plot_algorithm_line(
            ibm_rows,
            algorithm,
            "num_qubits",
            "Number of Qubits",
            args.out_dir / f"1_{algorithm.lower()}_dsr_variants_qubits.png",
            figsize=(15, 6),
        )

        # Generate depth plots if available
        if any(
            _to_float(row.get("transpiled_depth", "")) is not None
            for row in ibm_rows
            if row.get("algorithm") == algorithm
        ):
            _plot_algorithm_boxplot(
                ibm_rows,
                algorithm,
                "transpiled_depth",
                "Transpiled Depth",
                args.out_dir / f"{algorithm.lower()}_dsr_boxplot_depth.png",
            )
            _plot_algorithm_line(
                ibm_rows,
                algorithm,
                "transpiled_depth",
                "Transpiled Depth",
                args.out_dir / f"{algorithm.lower()}_dsr_variants_depth.png",
            )

            # Generate heatmap for Michelson only (our selected metric)
            _plot_algorithm_heatmap(
                ibm_rows,
                algorithm,
                "dsr_michelson",
                "Michelson",
                args.out_dir,
                use_log_scale=True,
            )

        # Print stats (IBM only — matches per-algorithm plots)
        stats = _compute_algorithm_stats(ibm_rows, algorithm)
        algo_count = stats.get("count", 0)
        mismatch_rate = stats.get("peak_mismatch_rate", 0) * 100
        print(f"{algorithm}:")
        print(f"  - Rows: {algo_count}")
        print(f"  - Has optimization levels: {stats.get('has_optimization_levels', False)}")
        print(f"  - Peak mismatch rate: {mismatch_rate:.1f}%")
        if "Michelson" in stats.get("variants", {}):
            mic = stats["variants"]["Michelson"]
            print(
                f"  - Michelson DSR: mean={mic['mean']:.3f}, median={mic['median']:.3f}, range=[{mic['min']:.3f}, {mic['max']:.3f}]"
            )
        print()

    # Combined comparison plots use ALL rows (IBM + Rigetti)
    _plot_combined_comparison(rows, args.out_dir)
    print("Generated combined comparison plot (boxplots)")
    _plot_combined_depth_comparison(rows, args.out_dir)
    print("Generated combined comparison plot (depth)")
    _plot_qft_heatmap_by_optimization(ibm_rows, args.out_dir)
    print("Generated QFT heatmap by optimization level")

    # DSR vs Qubits (individual + grid) — IBM only (optimization-level plots)
    _plot_combined_regions_by_optimization(ibm_rows, args.out_dir)
    print("Generated DSR region plots by optimization level (qubits)")

    _plot_combined_regions_grid(ibm_rows, args.out_dir)
    print("Generated DSR regions 2x2 grid (qubits)")

    # DSR vs Circuit Depth (individual + grid)
    # Cap at 2k to keep the region where all three algorithms overlap readable
    _plot_combined_regions_by_optimization(
        ibm_rows,
        args.out_dir,
        x_key="transpiled_depth",
        x_label="Transpiled Depth",
        file_suffix="_depth",
        max_x=2000,
    )
    print("Generated DSR region plots by optimization level (depth < 2k)")

    _plot_combined_regions_grid(
        ibm_rows,
        args.out_dir,
        x_key="transpiled_depth",
        x_label="Transpiled Depth",
        file_suffix="_depth",
        max_x=2000,
    )
    print("Generated DSR regions 2x2 grid (depth < 2k)")

    print(f"\nWrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
