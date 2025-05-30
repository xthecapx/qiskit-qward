# QWARD Visualization Guide

## Overview

QWARD provides a comprehensive visualization system for quantum circuit metrics. The visualization module follows a clean, object-oriented architecture that makes it easy to create beautiful, informative plots for your quantum computing analysis.

## Architecture

The visualization system is built around a simple but powerful architecture:

- **`PlotConfig`**: A dataclass holding all plot appearance and saving configurations
- **`BaseVisualizer`**: An abstract base class that handles common functionality like output directory management, styling, and plot saving/showing. Subclasses implement `create_plot()` for their specific visualization logic
- **`CircuitPerformanceVisualizer`**: A concrete visualizer that creates various plots for circuit performance metrics analysis, including individual plots and comprehensive dashboards

```mermaid
classDiagram
    class BaseVisualizer {
        <<abstract>>
        +output_dir: str
        +config: PlotConfig
        +save_plot(fig, filename)
        +show_plot(fig)
        +create_plot()*
        -_setup_output_dir()
        -_apply_style()
    }
    
    class PlotConfig {
        +figsize: Tuple[int, int]
        +dpi: int
        +style: str
        +color_palette: List[str]
        +save_format: str
        +grid: bool
        +alpha: float
        +__post_init__()
    }
    
    class CircuitPerformanceVisualizer {
        +metrics_dict: Dict[str, DataFrame]
        +individual_df: DataFrame
        +aggregate_df: DataFrame
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
        +plot_all()
        -_add_stacked_bar_labels()
        -_add_stacked_bar_summary()
    }

    BaseVisualizer <|-- CircuitPerformanceVisualizer
    BaseVisualizer --> PlotConfig : uses
    
    note for CircuitPerformanceVisualizer "Handles CircuitPerformance metrics visualization with multiple plot types and dashboard creation"
```

## Quick Start

### Basic Usage

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import CircuitPerformance
from qward.visualization import CircuitPerformanceVisualizer

# Create a quantum circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Run the circuit
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# Calculate metrics
scanner = Scanner(circuit=circuit)
circuit_performance = CircuitPerformance(circuit=circuit, success_criteria=lambda x: x == "11")
circuit_performance.add_job(job)
scanner.add_strategy(circuit_performance)

metrics_dict = scanner.calculate_metrics()

# Create visualizations
visualizer = CircuitPerformanceVisualizer(metrics_dict, output_dir="img/my_plots")
figures = visualizer.plot_all(save=True, show=False)
```

### Custom Configuration

```python
from qward.visualization import PlotConfig

# Create custom configuration
custom_config = PlotConfig(
    figsize=(12, 8),
    style="quantum",
    dpi=150,
    save_format="svg",
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
)

# Use custom configuration
visualizer = CircuitPerformanceVisualizer(
    metrics_dict, 
    output_dir="img/custom_plots",
    config=custom_config
)
```

## Available Visualizers

### CircuitPerformanceVisualizer

The `CircuitPerformanceVisualizer` creates comprehensive visualizations for circuit performance metrics. It expects metrics data with specific keys:
- `"CircuitPerformance.individual_jobs"`: DataFrame with individual job metrics
- `"CircuitPerformance.aggregate"`: DataFrame with aggregate statistics

#### Individual Plots

1. **Success vs Error Rate Comparison**
   ```python
   visualizer.plot_success_error_comparison(save=True, show=True)
   ```
   Shows success and error rates across different jobs as a bar chart.

2. **Fidelity Comparison**
   ```python
   visualizer.plot_fidelity_comparison(save=True, show=True)
   ```
   Displays fidelity values for each job.

3. **Shot Distribution**
   ```python
   visualizer.plot_shot_distribution(save=True, show=True)
   ```
   Shows the distribution of successful vs failed shots as stacked bars with detailed labels.

4. **Aggregate Summary**
   ```python
   visualizer.plot_aggregate_summary(save=True, show=True)
   ```
   Provides a comprehensive summary of aggregate statistics.

#### Dashboard View

Create a comprehensive dashboard with all visualizations in a 2x2 subplot layout:

```python
dashboard_fig = visualizer.create_dashboard(save=True, show=False)
```

#### Plot All

Generate all individual plots at once:

```python
all_figures = visualizer.plot_all(save=True, show=False)
```

## Plot Configuration

### Available Styles

- `"default"`: Standard matplotlib style
- `"quantum"`: Custom quantum-themed style with clean backgrounds and seaborn-inspired appearance
- `"minimal"`: Minimalist style with white grid

### Color Palettes

The default color palette is ColorBrewer-inspired for better accessibility:

```python
default_palette = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]
```

### Save Formats

Supported formats: `"png"`, `"svg"`, `"pdf"`, `"eps"`

## Advanced Examples

### Multiple Jobs Visualization

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Create multiple jobs with different noise levels
jobs = []
noise_levels = [0.0, 0.01, 0.05, 0.1]

for noise_level in noise_levels:
    if noise_level == 0.0:
        # No noise
        job = simulator.run(circuit, shots=1024)
    else:
        # Add noise
        noise_model = NoiseModel()
        error = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        
        noisy_simulator = AerSimulator(noise_model=noise_model)
        job = noisy_simulator.run(circuit, shots=1024)
    
    jobs.append(job)

# Wait for completion
for job in jobs:
    job.result()

# Analyze with CircuitPerformance
scanner = Scanner(circuit=circuit)
circuit_performance_strategy = CircuitPerformance(circuit=circuit)
circuit_performance_strategy.add_job(jobs)

scanner.add_strategy(circuit_performance_strategy)
metrics_dict = scanner.calculate_metrics()

# Create comprehensive visualizations
visualizer = CircuitPerformanceVisualizer(metrics_dict, output_dir="img/noise_analysis")
dashboard = visualizer.create_dashboard(save=True, show=True)
```

### Custom Success Criteria

```python
# Define custom success criteria for different quantum algorithms
def bell_state_success(outcome):
    """Success for Bell state: |00⟩ or |11⟩"""
    clean = outcome.replace(" ", "")
    return clean in ["00", "11"]

circuit_performance = CircuitPerformance(
    circuit=circuit,
    success_criteria=bell_state_success
)
circuit_performance.add_job(job)

scanner = Scanner(circuit=circuit)
scanner.add_strategy(circuit_performance)
metrics_dict = scanner.calculate_metrics()

# Visualize with custom styling
config = PlotConfig(
    figsize=(14, 10),
    style="quantum",
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    save_format="png",
    dpi=150
)

visualizer = CircuitPerformanceVisualizer(
    metrics_dict, 
    output_dir="img/bell_state_analysis",
    config=config
)

# Create all plots
visualizer.plot_all(save=True, show=False)
```

## Data Format Requirements

### CircuitPerformanceVisualizer Data Format

The `CircuitPerformanceVisualizer` expects specific DataFrame structures:

#### Individual Jobs DataFrame (`"CircuitPerformance.individual_jobs"`)

Required columns:
- `job_id`: Unique identifier for each job
- `success_rate`: Success rate (0.0 to 1.0)
- `error_rate`: Error rate (0.0 to 1.0)
- `fidelity`: Quantum fidelity (0.0 to 1.0)
- `total_shots`: Total number of shots
- `successful_shots`: Number of successful shots

#### Aggregate DataFrame (`"CircuitPerformance.aggregate"`)

Required columns:
- `mean_success_rate`: Mean success rate across jobs
- `std_success_rate`: Standard deviation of success rate
- `min_success_rate`: Minimum success rate
- `max_success_rate`: Maximum success rate
- `total_trials`: Total number of trials across all jobs
- `fidelity`: Overall fidelity
- `error_rate`: Overall error rate

## Creating Custom Visualizers

You can extend the visualization system by creating custom visualizers:

```python
from qward.visualization.base import BaseVisualizer
import matplotlib.pyplot as plt

class MyCustomVisualizer(BaseVisualizer):
    def __init__(self, metrics_dict, output_dir="img", config=None):
        super().__init__(output_dir, config)
        self.metrics_dict = metrics_dict
    
    def create_custom_plot(self, save=False, show=True):
        """Create a custom plot for your specific needs."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Your custom plotting logic here
        # Access your data via self.metrics_dict
        
        ax.set_title("My Custom Analysis")
        ax.grid(self.config.grid)
        
        if save:
            self.save_plot(fig, "custom_analysis")
        if show:
            self.show_plot(fig)
        
        return fig

# Usage
custom_visualizer = MyCustomVisualizer(metrics_dict)
custom_visualizer.create_custom_plot(save=True)
```

## Integration with Jupyter Notebooks

QWARD visualizations work seamlessly in Jupyter notebooks:

```python
# In a Jupyter cell
%matplotlib inline

visualizer = CircuitPerformanceVisualizer(metrics_dict)

# Show plots inline
visualizer.plot_success_error_comparison(show=True, save=False)
visualizer.plot_fidelity_comparison(show=True, save=False)

# Create dashboard inline
dashboard = visualizer.create_dashboard(show=True, save=False)
```

## Styling and Themes

### Available Styles
- `"default"`: Matplotlib default
- `"seaborn"`: Seaborn style
- `"quantum"`: Custom quantum computing theme
- `"publication"`: Publication-ready plots

### Custom Color Palettes
```python
# Quantum-inspired colors
quantum_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Monochrome for publications
mono_colors = ["#000000", "#404040", "#808080", "#c0c0c0"]

config = PlotConfig(color_palette=quantum_colors)
```

## Performance Tips

1. **Batch Operations**: Use `plot_all()` for multiple plots to avoid repeated setup
2. **Output Formats**: Use PNG for quick previews, SVG for publications
3. **DPI Settings**: Use lower DPI (150) for quick analysis, higher (300+) for publications
4. **Memory Management**: Close figures explicitly in loops to prevent memory leaks

## Troubleshooting

### Common Issues

1. **Missing Data Keys**: Ensure your metrics dictionary contains the expected keys (`"CircuitPerformance.individual_jobs"` and `"CircuitPerformance.aggregate"`)
2. **Empty DataFrames**: Check that your metric calculations completed successfully
3. **Plot Not Showing**: Verify your matplotlib backend settings
4. **File Permissions**: Ensure write permissions for the output directory

### Debug Mode
```python
# Enable debug information
import logging
logging.basicConfig(level=logging.DEBUG)

visualizer = CircuitPerformanceVisualizer(metrics_dict)
```

## CircuitPerformanceVisualizer Methods

### Core Methods

- `plot_success_error_comparison(save=False, show=True)`: Creates bar charts comparing success vs error rates
- `plot_fidelity_comparison(save=False, show=True)`: Shows fidelity values across jobs
- `plot_shot_distribution(save=False, show=True)`: Displays shot distribution with detailed labels
- `plot_aggregate_summary(save=False, show=True)`: Summary of aggregate statistics
- `create_dashboard(save=False, show=True)`: Comprehensive 2x2 dashboard
- `plot_all(save=False, show=True)`: Generate all individual plots

### Utility Methods

- `save_plot(fig, filename)`: Save figure with configured format and DPI
- `show_plot(fig)`: Display figure with proper backend handling

For more examples and advanced usage, see the `qward/examples/visualization_demo.py` file.