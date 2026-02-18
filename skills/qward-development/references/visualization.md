# Visualization Reference

QWARD provides a comprehensive, type-safe visualization system.

## Quick Start

```python
from qward import Scanner
from qward.visualization import Visualizer
from qward.visualization.constants import Metrics, Plots

# Fluent API (simplest)
Scanner(circuit).scan().visualize(save=True, show=False)

# Full control API
scanner = Scanner(circuit, strategies=[QiskitMetrics, ComplexityMetrics])
scanner.scan()

visualizer = Visualizer(scanner=scanner, output_dir="my_analysis")
visualizer.create_dashboard(save=True, show=False)
```

## Type-Safe Constants

```python
from qward.visualization.constants import Metrics, Plots

# Metric constants (prevents typos)
Metrics.QISKIT              # "QiskitMetrics"
Metrics.COMPLEXITY          # "ComplexityMetrics"
Metrics.CIRCUIT_PERFORMANCE # "CircuitPerformance"

# Plot constants (IDE autocomplete) - Note: PascalCase for metric, UPPER_CASE for plot
Plots.Qiskit.CIRCUIT_STRUCTURE      # "circuit_structure"
Plots.Qiskit.GATE_DISTRIBUTION      # "gate_distribution"
Plots.Qiskit.INSTRUCTION_METRICS    # "instruction_metrics"
Plots.Qiskit.CIRCUIT_SUMMARY        # "circuit_summary"

Plots.Complexity.GATE_BASED_METRICS # "gate_based_metrics"
Plots.Complexity.COMPLEXITY_RADAR   # "complexity_radar"
Plots.Complexity.EFFICIENCY_METRICS # "efficiency_metrics"

Plots.CircuitPerformance.SUCCESS_ERROR_COMPARISON  # "success_error_comparison"
Plots.CircuitPerformance.SHOT_DISTRIBUTION         # "shot_distribution"
Plots.CircuitPerformance.AGGREGATE_SUMMARY         # "aggregate_summary"
```

## Visualizer Class

### Constructor

```python
from qward.visualization import Visualizer, PlotConfig

visualizer = Visualizer(
    scanner=scanner,           # Scanner with calculated metrics
    # OR
    metrics_data=dict,         # Raw metrics dictionary

    output_dir="output",       # Where to save plots
    config=PlotConfig(),       # Styling configuration
)
```

### generate_plot() - Single Plot

```python
# Generate one specific plot
fig = visualizer.generate_plot(
    metric_name=Metrics.QISKIT,
    plot_name=Plots.QISKIT.CIRCUIT_STRUCTURE,
    save=True,
    show=False
)
```

### generate_plots() - Selected Plots

```python
# Generate specific plots for specific metrics
figs = visualizer.generate_plots(
    selections={
        Metrics.QISKIT: [
            Plots.Qiskit.CIRCUIT_STRUCTURE,
            Plots.Qiskit.GATE_DISTRIBUTION
        ],
        Metrics.COMPLEXITY: [
            Plots.Complexity.COMPLEXITY_RADAR
        ]
    },
    save=True,
    show=False
)

# Generate ALL plots for a metric (use None)
figs = visualizer.generate_plots(
    selections={Metrics.QISKIT: None},  # All Qiskit plots
    save=True,
    show=False
)
```

### create_dashboard() - All Dashboards

```python
# Create comprehensive dashboards for all available metrics
dashboards = visualizer.create_dashboard(save=True, show=False)
# Returns: Dict[str, Figure]
```

### Exploration Methods

```python
# List available plots
available = visualizer.get_available_plots()
# {"QiskitMetrics": ["circuit_structure", "gate_distribution", ...], ...}

# Get plot metadata
metadata = visualizer.get_plot_metadata(Metrics.QISKIT, Plots.QISKIT.CIRCUIT_STRUCTURE)
print(metadata.description)   # "Basic circuit metrics visualization"
print(metadata.plot_type)     # PlotType.BAR
print(metadata.dependencies)  # ["depth", "width", "size"]

# List available metrics
metrics = visualizer.get_available_metrics()

# Print summary
visualizer.print_available_metrics()
```

## PlotConfig

Customize plot appearance:

```python
from qward.visualization import PlotConfig

config = PlotConfig(
    figsize=(12, 8),          # Figure size in inches
    dpi=150,                  # Resolution
    style="default",          # "default", "quantum", "minimal"
    color_palette=[           # Custom colors
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
    ],
    save_format="png",        # "png", "svg", "pdf"
    grid=True,                # Show grid lines
    alpha=0.8,                # Transparency
)

visualizer = Visualizer(scanner=scanner, config=config)
```

## Individual Visualizers

For direct access to specific visualizers:

```python
from qward.visualization import (
    QiskitVisualizer,
    ComplexityVisualizer,
    CircuitPerformanceVisualizer
)

# Get metrics data
metrics_dict = scanner.calculate_metrics()

# Use specific visualizer
qiskit_viz = QiskitVisualizer(
    metrics_data={Metrics.QISKIT: metrics_dict["QiskitMetrics"]},
    output_dir="output"
)

# Generate plots
qiskit_viz.generate_plot(Plots.QISKIT.CIRCUIT_STRUCTURE, save=True)
qiskit_viz.generate_all_plots(save=True, show=False)
qiskit_viz.create_dashboard(save=True)
```

## Available Plots by Metric

### QiskitMetrics (4 plots)
| Plot | Description |
|------|-------------|
| `circuit_structure` | Basic metrics: depth, width, size, qubits |
| `gate_distribution` | Gate type analysis and distribution |
| `instruction_metrics` | Connectivity and instruction analysis |
| `circuit_summary` | Derived metrics summary |

### ComplexityMetrics (3 plots)
| Plot | Description |
|------|-------------|
| `gate_based_metrics` | Gate counts, T-gates, CNOTs |
| `complexity_radar` | Radar chart of normalized complexity |
| `efficiency_metrics` | Parallelism and circuit efficiency |

### CircuitPerformanceMetrics (3 plots)
| Plot | Description |
|------|-------------|
| `success_error_comparison` | Success vs error rate comparison |
| `shot_distribution` | Successful vs failed shots |
| `aggregate_summary` | Statistical summary across jobs |

## Memory-Efficient Batch Processing

```python
# Default: save=False, show=False for memory efficiency
for circuit in circuit_variants:
    scanner = Scanner(circuit, strategies=[QiskitMetrics])
    visualizer = Visualizer(scanner=scanner, output_dir=f"analysis_{i}")

    # Generate without displaying (memory efficient)
    all_plots = visualizer.generate_plots({Metrics.QISKIT: None})

    # Only save important ones
    visualizer.generate_plots(
        {Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE]},
        save=True
    )
```

## ScanResult.visualize()

Quick visualization via fluent API:

```python
Scanner(circuit).scan().visualize(
    save=True,
    show=False,
    output_dir="qward/examples/img",
    config=PlotConfig(),
    selections={Metrics.QISKIT: [Plots.Qiskit.CIRCUIT_STRUCTURE]}  # Optional
)
```
