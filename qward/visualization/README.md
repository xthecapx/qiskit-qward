# QWARD Visualization System

The QWARD visualization system provides a unified interface for creating comprehensive visualizations of quantum circuit metrics. It supports automatic detection of available metrics and provides appropriate visualizations for each metric type.

**Note**: By default, plots are not saved or displayed (`save=False`, `show=False`) to optimize memory usage when generating multiple visualizations. Explicitly set these parameters to `True` when you want to save or display plots.

## Overview

The visualization system consists of:

- **`Visualizer`**: Main unified interface that works with Scanner or custom data
- **Metric-specific visualizers**: Specialized visualizers for each metric type
- **`BaseVisualizer`**: Common base class with shared functionality
- **`PlotConfig`**: Configuration for plot appearance and styling

## Quick Start

### Basic Usage with Scanner

```python
from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create a quantum circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run on simulator for CircuitPerformance metrics
simulator = AerSimulator()
jobs = [simulator.run(qc, shots=1000) for _ in range(3)]

# Create scanner with all metrics
circuit_performance = CircuitPerformance(circuit=qc, jobs=jobs)
scanner = Scanner(circuit=qc, strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance])

# Create visualizer
visualizer = Visualizer(scanner=scanner)

# Create dashboard for all available metrics (explicitly save and show)
dashboards = visualizer.create_dashboard(save=True, show=True)

# Or visualize specific metrics
qiskit_plots = visualizer.visualize_metric("QiskitMetrics", save=True, show=True)
performance_plots = visualizer.visualize_metric("CircuitPerformance", save=True, show=True)
```

### Custom Configuration

```python
from qward.visualization import Visualizer, PlotConfig

# Create custom plot configuration
config = PlotConfig(
    figsize=(12, 8),
    style="quantum",
    color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1"],
    alpha=0.8,
    grid=True
)

# Use custom config
visualizer = Visualizer(scanner=scanner, config=config, output_dir="my_plots")
```

### Using Custom Data

```python
import pandas as pd
from qward.visualization import Visualizer

# Prepare your own metrics data
custom_data = {
    "QiskitMetrics": pd.DataFrame([{
        "basic_metrics.depth": 5,
        "basic_metrics.width": 6,
        "basic_metrics.size": 8,
        # ... more metrics
    }])
}

# Create visualizer with custom data
visualizer = Visualizer(metrics_data=custom_data)
```

## Available Visualizations

### QiskitMetrics Visualizations

The `QiskitMetricsVisualizer` provides:

1. **Circuit Structure**: Bar chart showing depth, width, size, qubits, and classical bits
2. **Gate Distribution**: Pie chart showing the distribution of different gate types
3. **Instruction Metrics**: Bar chart of connectivity and instruction-related metrics
4. **Circuit Summary**: Derived efficiency metrics like gate density and parallelism

### ComplexityMetrics Visualizations

The `ComplexityMetricsVisualizer` provides:

1. **Gate-Based Metrics**: Bar chart of gate counts, T-count, CNOT count, etc.
2. **Complexity Radar Chart**: Multi-dimensional radar chart showing normalized complexity metrics
3. **Quantum Volume Analysis**: Bar chart of quantum volume estimates
4. **Efficiency Metrics**: Circuit efficiency and resource utilization metrics

### CircuitPerformance Visualizations

The `CircuitPerformanceVisualizer` provides:

1. **Success vs Error Rates**: Comparison of success and error rates across jobs
2. **Fidelity Comparison**: Fidelity metrics across different jobs
3. **Shot Distribution**: Stacked bar chart showing successful vs failed shots
4. **Aggregate Summary**: Statistical summary of performance metrics across multiple jobs

## API Reference

### Visualizer Class

```python
class Visualizer:
    def __init__(
        self,
        scanner: Optional[Scanner] = None,
        metrics_data: Optional[Dict[str, pd.DataFrame]] = None,
        config: Optional[PlotConfig] = None,
        output_dir: str = "img"
    )
```

**Methods:**

- `get_available_metrics()`: List available metrics for visualization
- `visualize_metric(metric_name, save=True, show=True)`: Visualize specific metric
- `create_dashboard(save=True, show=True)`: Create comprehensive dashboard
- `visualize_all(save=True, show=True)`: Create all individual plots
- `register_visualizer(metric_name, visualizer_class)`: Register custom visualizer
- `get_metric_summary()`: Get summary of available metrics
- `print_available_metrics()`: Print information about available metrics

### PlotConfig Class

```python
@dataclass
class PlotConfig:
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 300
    style: str = "default"  # "default", "quantum", "minimal"
    color_palette: List[str] = None  # Auto-generated if None
    save_format: str = "png"
    grid: bool = True
    alpha: float = 0.7
```

## Extending the System

### Creating Custom Visualizers

To create a custom visualizer for your own metrics:

```python
from qward.visualization import BaseVisualizer
import matplotlib.pyplot as plt

class MyCustomVisualizer(BaseVisualizer):
    def __init__(self, metrics_dict, output_dir="img", config=None):
        super().__init__(output_dir, config)
        self.metrics_dict = metrics_dict
        self.my_df = metrics_dict["MyCustomMetric"]
    
    def create_plot(self) -> plt.Figure:
        # Required method - create default plot
        return self.plot_my_metric()
    
    def plot_my_metric(self, save=False, show=False):
        fig, ax, is_override = self._setup_plot_axes()
        
        # Use base class utilities
        my_data = self._extract_metrics_from_columns(
            self.my_df, 
            ["my_metric.value1", "my_metric.value2"],
            prefix_to_remove="my_metric."
        )
        
        self._create_bar_plot_with_labels(
            data=my_data,
            ax=ax,
            title="My Custom Metrics",
            value_format="auto"
        )
        
        return self._finalize_plot(fig, is_override, save, show, "my_custom_plot")
    
    def plot_all(self, save=False, show=False):
        return [self.plot_my_metric(save=save, show=show)]

# Register your custom visualizer
visualizer = Visualizer(scanner=scanner)
visualizer.register_visualizer("MyCustomMetric", MyCustomVisualizer)
```

### Data Format Requirements

Each metric visualizer expects data in a specific format:

**QiskitMetrics**: DataFrame with columns like:
- `basic_metrics.depth`, `basic_metrics.width`, `basic_metrics.size`
- `basic_metrics.count_ops.{gate_name}` for gate counts
- `instruction_metrics.num_connected_components`, etc.

**ComplexityMetrics**: DataFrame with columns like:
- `gate_based_metrics.gate_count`, `gate_based_metrics.circuit_depth`
- `standardized_metrics.gate_density`, `standardized_metrics.circuit_volume`
- `quantum_volume.standard_quantum_volume`, etc.

**CircuitPerformance**: Dictionary with keys:
- `CircuitPerformance.individual_jobs`: DataFrame with job-level metrics
- `CircuitPerformance.aggregate`: DataFrame with aggregate statistics (for multiple jobs)

## Examples

### Complete Example with All Metrics

```python
from qward import Scanner, Visualizer
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformance
from qward.visualization import PlotConfig
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create circuit
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

# Run on simulator for CircuitPerformance
simulator = AerSimulator()
jobs = [simulator.run(qc, shots=1000) for _ in range(3)]

# Create CircuitPerformance with multiple jobs
circuit_performance = CircuitPerformance(circuit=qc, jobs=jobs)

# Create scanner with all metrics
scanner = Scanner(
    circuit=qc, 
    strategies=[QiskitMetrics, ComplexityMetrics, circuit_performance]
)

# Create visualizer with custom config
config = PlotConfig(
    figsize=(14, 10),
    style="quantum",
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    alpha=0.8
)

visualizer = Visualizer(scanner=scanner, config=config)

# Print summary
visualizer.print_available_metrics()

# Create comprehensive dashboard
dashboards = visualizer.create_dashboard(save=True, show=False)

# Create individual plots
all_plots = visualizer.visualize_all(save=True, show=False)

print(f"Created {len(dashboards)} dashboards and {sum(len(plots) for plots in all_plots.values())} individual plots")
```

### Working with Single vs Multiple Jobs

```python
# Single job CircuitPerformance
single_job = simulator.run(qc, shots=1000)
single_performance = CircuitPerformance(circuit=qc, job=single_job)
single_scanner = Scanner(circuit=qc, strategies=[single_performance])

# Multiple jobs CircuitPerformance
multi_jobs = [simulator.run(qc, shots=1000) for _ in range(5)]
multi_performance = CircuitPerformance(circuit=qc, jobs=multi_jobs)
multi_scanner = Scanner(circuit=qc, strategies=[multi_performance])

# Both will work with the visualizer
single_visualizer = Visualizer(scanner=single_scanner)
multi_visualizer = Visualizer(scanner=multi_scanner)

# Single job creates a simplified dashboard
single_dashboard = single_visualizer.create_dashboard()

# Multiple jobs creates a full dashboard with aggregate statistics
multi_dashboard = multi_visualizer.create_dashboard()
```

## Output Files

By default, plots are saved to the `img/` directory with descriptive names:

- `qiskit_metrics_dashboard.png`: QiskitMetrics dashboard
- `qiskit_circuit_structure.png`: Individual circuit structure plot
- `complexity_metrics_dashboard.png`: ComplexityMetrics dashboard
- `complexity_radar_chart.png`: Complexity radar chart
- `circuit_performance_dashboard.png`: CircuitPerformance dashboard
- `success_error_rates.png`: Success vs error rates comparison
- `fidelity_comparison.png`: Fidelity across jobs
- `shot_distribution.png`: Shot distribution analysis
- `aggregate_statistics.png`: Aggregate performance statistics

You can customize the output directory using the `output_dir` parameter.

## Tips and Best Practices

1. **Use dashboards for overview**: Dashboards provide a comprehensive view of all metrics
2. **Individual plots for details**: Use individual plots when you need to focus on specific aspects
3. **Custom configurations**: Adjust colors, sizes, and styles to match your presentation needs
4. **Data validation**: The system validates data format and provides helpful error messages
5. **Extensibility**: Easy to add new visualizers for custom metrics
6. **Performance**: Visualizations are optimized for both single and multiple job scenarios
7. **Memory management**: Close figures when creating many plots to avoid memory warnings

## Troubleshooting

**Common Issues:**

1. **Missing data columns**: Check that your metrics data contains the required columns
2. **Empty DataFrames**: Ensure your Scanner has calculated metrics before visualization
3. **Import errors**: Make sure all dependencies (matplotlib, pandas) are installed
4. **Custom visualizers**: Ensure your custom visualizer inherits from `BaseVisualizer`
5. **CircuitPerformance data**: Ensure jobs have completed before creating CircuitPerformance metrics

**Getting Help:**

- Check the data format using `visualizer.get_metric_summary()`
- Use `visualizer.print_available_metrics()` to see what's available
- Look at the example files in `qward/examples/` for reference implementations
- For CircuitPerformance, ensure you have either a single job or multiple jobs properly configured

### Memory-Efficient Workflow

```python
# For batch processing or when generating many plots, use default settings
# This avoids memory issues from displaying plots and only saves when needed

# Generate plots without displaying (memory efficient)
dashboards = visualizer.create_dashboard()  # save=False, show=False by default
all_plots = visualizer.visualize_all()      # save=False, show=False by default

# Only save specific plots you need
important_dashboard = visualizer.create_dashboard(save=True)  # Save but don't show
specific_plots = visualizer.visualize_metric("QiskitMetrics", save=True, show=True)  # Save and show

# For interactive analysis, explicitly show plots
interactive_dashboard = visualizer.create_dashboard(show=True)  # Show but don't save
``` 