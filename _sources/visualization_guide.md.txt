# QWARD Visualization Guide

## Overview

QWARD provides a comprehensive visualization system for quantum circuit metrics. The visualization module follows a clean, object-oriented architecture that makes it easy to create beautiful, informative plots for your quantum computing analysis.

## Architecture

The visualization system is built around a powerful and extensible architecture:

- **`PlotConfig`**: A dataclass holding all plot appearance and saving configurations
- **`VisualizationStrategy`**: An abstract base class that handles common functionality like output directory management, styling, data validation, and plot creation utilities
- **Individual Visualizers**: Three specialized visualizers for different metric types:
  - **`QiskitVisualizer`**: Circuit structure and instruction analysis
  - **`ComplexityVisualizer`**: Complexity analysis with radar charts and efficiency metrics
  - **`CircuitPerformanceVisualizer`**: Performance metrics with success rates and fidelity analysis
- **`Visualizer`**: A unified entry point that automatically detects available metrics and provides comprehensive visualization capabilities

```{mermaid}
classDiagram
    class VisualizationStrategy {
        <<abstract>>
        +output_dir: str
        +config: PlotConfig
        +save_plot(fig, filename)
        +show_plot(fig)
        +create_dashboard()*
        +plot_all()*
        +_validate_required_columns()
        +_extract_metrics_from_columns()
        +_create_bar_plot_with_labels()
        +_setup_plot_axes()
        +_finalize_plot()
    }
    
    class PlotConfig {
        +figsize: Tuple[int, int]
        +dpi: int
        +style: str
        +color_palette: List[str]
        +save_format: str
        +grid: bool
        +alpha: float
    }
    
    class QiskitVisualizer {
        +plot_circuit_structure()
        +plot_gate_distribution()
        +plot_instruction_metrics()
        +plot_circuit_summary()
        +create_dashboard()
        +plot_all()
    }
    
    class ComplexityVisualizer {
        +plot_gate_based_metrics()
        +plot_complexity_radar()
        +plot_quantum_volume_analysis()
        +plot_efficiency_metrics()
        +create_dashboard()
        +plot_all()
    }
    
    class CircuitPerformanceVisualizer {
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
        +plot_all()
    }
    
    class Visualizer {
        +register_strategy()
        +get_available_metrics()
        +visualize_metric()
        +create_dashboard()
        +visualize_all()
        +print_available_metrics()
    }

    VisualizationStrategy <|-- QiskitVisualizer
    VisualizationStrategy <|-- ComplexityVisualizer
    VisualizationStrategy <|-- CircuitPerformanceVisualizer
    VisualizationStrategy --> PlotConfig : uses
    Visualizer --> VisualizationStrategy : manages
    
    note for Visualizer "Unified entry point with auto-detection and comprehensive visualization"
```

## Quick Start

### Using the Unified Visualizer (Recommended)

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import Visualizer

# Create a quantum circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Run simulation for CircuitPerformanceMetrics
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)

# Create scanner with all metrics
scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
circuit_performance = CircuitPerformanceMetrics(circuit=circuit, job=job)
scanner.add_strategy(circuit_performance)

# Create unified visualizer
visualizer = Visualizer(scanner=scanner, output_dir="qward/examples/img")

# Option 1: Create comprehensive dashboards for all metrics
dashboards = visualizer.create_dashboard(save=True, show=False)
print(f"Created {len(dashboards)} dashboards")

# Option 2: Create all individual plots
all_plots = visualizer.visualize_all(save=True, show=False)
print(f"Created plots for {len(all_plots)} metric types")

# Option 3: Visualize specific metrics
qiskit_plots = visualizer.visualize_metric("QiskitMetrics", save=True, show=False)
complexity_plots = visualizer.visualize_metric("ComplexityMetrics", save=True, show=False)
```

### Using Individual Visualizers (Advanced)

```python
from qward.visualization import QiskitVisualizer, ComplexityVisualizer, CircuitPerformanceVisualizer

# Calculate metrics first
metrics_dict = scanner.calculate_metrics()

# Use QiskitVisualizer
qiskit_viz = QiskitVisualizer(
    metrics_dict={"QiskitMetrics": metrics_dict["QiskitMetrics"]},
    output_dir="qward/examples/img"
)
qiskit_figures = qiskit_viz.plot_all(save=True, show=False)

# Use ComplexityVisualizer
complexity_viz = ComplexityVisualizer(
    metrics_dict={"ComplexityMetrics": metrics_dict["ComplexityMetrics"]},
    output_dir="qward/examples/img"
)
complexity_figures = complexity_viz.plot_all(save=True, show=False)

# Use CircuitPerformanceVisualizer
circuit_perf_data = {k: v for k, v in metrics_dict.items() if k.startswith("CircuitPerformance")}
perf_viz = CircuitPerformanceVisualizer(
    metrics_dict=circuit_perf_data,
    output_dir="qward/examples/img"
)
perf_figures = perf_viz.plot_all(save=True, show=False)
```

## Examples and Usage Patterns

### 📁 Available Examples

QWARD provides comprehensive examples in the `qward/examples/` directory:

- **`example_visualizer.py`** - Complete workflow examples showing all features of the unified Visualizer
- **`visualization_demo.py`** - Focused demo of CircuitPerformanceVisualizer capabilities
- **`direct_strategy_example.py`** - Shows how to use visualization strategies directly for maximum control
- **`aer.py`** - Integration examples with Qiskit Aer simulator
- **`circuit_performance_demo.py`** - Circuit performance analysis examples
- **`visualization_quickstart.py`** - Quick start example with minimal setup

### 🎨 Two Approaches to Visualization

#### 1. Unified Visualizer (Recommended)

Use the `Visualizer` class for most cases - it automatically detects your metrics and creates appropriate visualizations:

```python
from qward.visualization import Visualizer

# From Scanner
visualizer = Visualizer(scanner=scanner)

# From custom data
visualizer = Visualizer(metrics_data=custom_metrics_dict)

# Create all visualizations
visualizer.visualize_all()
```

**Benefits:**
- Automatic metric detection
- Consistent styling across all plots
- Simple API for complex workflows
- Built-in error handling

#### 2. Direct Strategy Usage (Advanced)

Use individual strategies directly when you need fine-grained control:

```python
from qward.visualization import QiskitVisualizer, ComplexityVisualizer, CircuitPerformanceVisualizer

# Use specific strategies directly
qiskit_strategy = QiskitVisualizer(metrics_dict=qiskit_data, output_dir="custom_dir")
qiskit_strategy.plot_circuit_structure()
qiskit_strategy.create_dashboard()
```

**Benefits:**
- Fine-grained control over individual plots
- Custom configurations per strategy
- Integration with external data sources
- Flexible output formats and locations

### 🔧 Available Strategies

| Strategy | Metrics Type | Key Visualizations |
|----------|-------------|-------------------|
| `QiskitVisualizer` | QiskitMetrics | Circuit structure, gate distribution, instruction metrics |
| `ComplexityVisualizer` | ComplexityMetrics | Gate-based metrics, complexity radar, quantum volume |
| `CircuitPerformanceVisualizer` | CircuitPerformanceMetrics | Success rates, fidelity, shot distribution |

### 🎛️ Customization

#### Custom Plot Configuration

```python
from qward.visualization import PlotConfig

config = PlotConfig(
    figsize=(12, 8),
    style="quantum",
    color_palette=["#FF6B6B", "#4ECDC4", "#45B7D1"],
    save_format="pdf"
)

visualizer = Visualizer(scanner=scanner, config=config)
```

#### Custom Strategies

```python
from qward.visualization import VisualizationStrategy

class MyCustomStrategy(VisualizationStrategy):
    def create_dashboard(self, save=True, show=True):
        # Your custom visualization logic
        pass
    
    def plot_all(self, save=True, show=True):
        # Generate all plots for this strategy
        pass

# Register and use
visualizer.register_strategy("MyMetrics", MyCustomStrategy)
```

### 🚀 Running the Examples

```bash
# Run main visualizer examples
python qward/examples/example_visualizer.py

# Run CircuitPerformanceMetrics demo
python qward/examples/visualization_demo.py

# Run direct strategy examples
python qward/examples/direct_strategy_example.py

# Run Aer integration examples
python qward/examples/aer.py

# Quick start example
python qward/examples/visualization_quickstart.py
```

### 📊 Output

All examples save plots to `qward/examples/img/` by default. You'll find:

- **Dashboards**: Comprehensive multi-plot views
- **Individual plots**: Specific visualizations for detailed analysis
- **Multiple formats**: PNG (default), PDF, SVG support

### 💡 Best Practices

1. **Start with the unified Visualizer** - it handles most use cases automatically
2. **Use direct strategies** when you need specific plots or custom workflows
3. **Customize PlotConfig** for consistent styling across all visualizations
4. **Register custom strategies** to extend the system with your own visualizations
5. **Check the output directory** - all plots are saved with descriptive names
6. **Use meaningful output directories** - organize plots by analysis type or date

### 🔗 Key Benefits

- **Strategy Pattern**: Consistent with Scanner architecture
- **Auto-Detection**: Automatically finds and visualizes available metrics
- **Extensible**: Easy to add new visualization strategies
- **Flexible**: Use unified interface or direct strategies as needed
- **Customizable**: Full control over plot appearance and output formats

## Available Visualizers

### QiskitVisualizer

Visualizes circuit structure and instruction analysis from `QiskitMetrics`.

#### Individual Plots

1. **Circuit Structure**
   ```python
   qiskit_viz.plot_circuit_structure(save=True, show=True)
   ```
   Shows basic circuit metrics: depth, width, size, qubits, and classical bits.

2. **Gate Distribution**
   ```python
   qiskit_viz.plot_gate_distribution(save=True, show=True)
   ```
   Displays gate type analysis and instruction distribution as a pie chart.

3. **Instruction Metrics**
   ```python
   qiskit_viz.plot_instruction_metrics(save=True, show=True)
   ```
   Shows instruction-related metrics like connected components and nonlocal gates.

4. **Circuit Summary**
   ```python
   qiskit_viz.plot_circuit_summary(save=True, show=True)
   ```
   Displays derived metrics like gate density and parallelism.

#### Dashboard and All Plots

```python
# Create comprehensive dashboard
dashboard = qiskit_viz.create_dashboard(save=True, show=False)

# Generate all individual plots
all_figures = qiskit_viz.plot_all(save=True, show=False)
```

### ComplexityVisualizer

Visualizes complexity analysis from `ComplexityMetrics` with advanced charts and analysis.

#### Individual Plots

1. **Gate-Based Metrics**
   ```python
   complexity_viz.plot_gate_based_metrics(save=True, show=True)
   ```
   Shows gate counts, circuit depth, T-gates, and CNOT gates.

2. **Complexity Radar Chart**
   ```python
   complexity_viz.plot_complexity_radar(save=True, show=True)
   ```
   Creates a radar chart with normalized complexity indicators for quick visual assessment.

3. **Quantum Volume Analysis**
   ```python
   complexity_viz.plot_quantum_volume_analysis(save=True, show=True)
   ```
   Displays Quantum Volume estimation and contributing factors.

4. **Efficiency Metrics**
   ```python
   complexity_viz.plot_efficiency_metrics(save=True, show=True)
   ```
   Shows parallelism efficiency and circuit efficiency analysis.

#### Dashboard and All Plots

```python
# Create comprehensive dashboard
dashboard = complexity_viz.create_dashboard(save=True, show=False)

# Generate all individual plots
all_figures = complexity_viz.plot_all(save=True, show=False)
```

### CircuitPerformanceVisualizer

Visualizes performance metrics from `CircuitPerformance` with success rates, fidelity, and execution analysis.

#### Individual Plots

1. **Success vs Error Rate Comparison**
   ```python
   perf_viz.plot_success_error_comparison(save=True, show=True)
   ```
   Shows success and error rates across different jobs as a grouped bar chart.

2. **Fidelity Comparison**
   ```python
   perf_viz.plot_fidelity_comparison(save=True, show=True)
   ```
   Displays fidelity values for each job with value labels.

3. **Shot Distribution**
   ```python
   perf_viz.plot_shot_distribution(save=True, show=True)
   ```
   Shows the distribution of successful vs failed shots as stacked bars with detailed labels.

4. **Aggregate Summary**
   ```python
   perf_viz.plot_aggregate_summary(save=True, show=True)
   ```
   Provides a comprehensive summary of aggregate statistics across multiple jobs.

#### Dashboard and All Plots

```python
# Create comprehensive dashboard
dashboard = perf_viz.create_dashboard(save=True, show=False)

# Generate all individual plots
all_figures = perf_viz.plot_all(save=True, show=False)
```

## Plot Configuration

### PlotConfig Options

```python
from qward.visualization import PlotConfig

config = PlotConfig(
    figsize=(12, 8),           # Figure size in inches
    dpi=300,                   # Resolution for saved plots
    style="quantum",           # Plot style theme
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],  # Custom colors
    save_format="png",         # Output format: png, svg, pdf, eps
    grid=True,                 # Show grid lines
    alpha=0.8                  # Transparency level
)
```

### Available Styles

- `"default"`: Standard matplotlib style
- `"quantum"`: Custom quantum-themed style with clean backgrounds
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

## Advanced Examples

### Complete Analysis with Custom Configuration

```python
from qward.visualization import Visualizer, PlotConfig

# Create custom configuration
config = PlotConfig(
    figsize=(14, 10),
    style="quantum",
    dpi=150,
    save_format="svg",
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    alpha=0.8
)

# Create comprehensive analysis
scanner = Scanner(circuit=circuit)
scanner.add_metric(QiskitMetrics(circuit))
scanner.add_metric(ComplexityMetrics(circuit))
scanner.add_metric(CircuitPerformance(circuit=circuit, job=job))

# Use unified visualizer with custom config
visualizer = Visualizer(
    scanner=scanner, 
    config=config, 
    output_dir="comprehensive_analysis"
)

# Create all visualizations
dashboards = visualizer.create_dashboard(save=True, show=False)
all_plots = visualizer.visualize_all(save=True, show=False)

# Print summary
visualizer.print_available_metrics()
```

### Multiple Jobs Analysis

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Create multiple jobs with different noise levels
jobs = []
noise_levels = [0.0, 0.01, 0.05, 0.1]

for noise_level in noise_levels:
    if noise_level == 0.0:
        job = simulator.run(circuit, shots=1024)
    else:
        noise_model = NoiseModel()
        error = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        
        noisy_simulator = AerSimulator(noise_model=noise_model)
        job = noisy_simulator.run(circuit, shots=1024)
    
    jobs.append(job)

# Analyze with CircuitPerformance
circuit_performance = CircuitPerformance(circuit=circuit)
for job in jobs:
    circuit_performance.add_job(job)

scanner = Scanner(circuit=circuit)
scanner.add_metric(circuit_performance)
metrics_dict = scanner.calculate_metrics()

# Visualize noise effects
perf_viz = CircuitPerformanceVisualizer(
    {k: v for k, v in metrics_dict.items() if k.startswith("CircuitPerformance")},
    output_dir="noise_analysis"
)
dashboard = perf_viz.create_dashboard(save=True, show=True)
```

### Custom Success Criteria

```python
# Define custom success criteria for Bell state
def bell_state_success(outcome):
    """Success for Bell state: |00⟩ or |11⟩"""
    clean = outcome.replace(" ", "")
    return clean in ["00", "11"]

# Use custom criteria
circuit_performance = CircuitPerformance(
    circuit=circuit,
    job=job,
    success_criteria=bell_state_success
)

scanner = Scanner(circuit=circuit)
scanner.add_metric(QiskitMetrics(circuit))
scanner.add_metric(ComplexityMetrics(circuit))
scanner.add_metric(circuit_performance)

# Create visualizations with custom styling
config = PlotConfig(
    figsize=(16, 12),
    style="quantum",
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    save_format="png",
    dpi=150
)

visualizer = Visualizer(
    scanner=scanner,
    config=config,
    output_dir="bell_state_analysis"
)

# Create comprehensive analysis
dashboards = visualizer.create_dashboard(save=True, show=False)
all_plots = visualizer.visualize_all(save=True, show=False)
```

## Data Format Requirements

### QiskitMetricsVisualizer Data Format

Expects a DataFrame with QiskitMetrics columns:
- `basic_metrics.*`: Circuit structure metrics
- `instruction_metrics.*`: Gate and instruction analysis
- `scheduling_metrics.*`: Timing information

### ComplexityMetricsVisualizer Data Format

Expects a DataFrame with ComplexityMetrics columns:
- `gate_based_metrics.*`: Gate counts and circuit depth
- `entanglement_metrics.*`: Entanglement analysis
- `standardized_metrics.*`: Normalized complexity indicators
- `advanced_metrics.*`: Efficiency and parallelism metrics
- `quantum_volume.*`: Quantum Volume estimation

### CircuitPerformanceVisualizer Data Format

Expects specific DataFrame structures:

#### Individual Jobs DataFrame (`"CircuitPerformance.individual_jobs"`)
- `job_id`: Unique identifier for each job
- `success_rate`: Success rate (0.0 to 1.0)
- `error_rate`: Error rate (0.0 to 1.0)
- `fidelity`: Quantum fidelity (0.0 to 1.0)
- `total_shots`: Total number of shots
- `successful_shots`: Number of successful shots

#### Aggregate DataFrame (`"CircuitPerformance.aggregate"`)
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
        self.my_df = metrics_dict.get("MyCustomMetrics")
        
        if self.my_df is None:
            raise ValueError("'MyCustomMetrics' data not found in metrics_dict.")
    
    def create_plot(self):
        """Required by BaseVisualizer - creates default plot."""
        return self.plot_custom_analysis(save=False, show=False)
    
    def plot_custom_analysis(self, save=True, show=True, fig_ax_override=None):
        """Create a custom analysis plot."""
        fig, ax, is_override = self._setup_plot_axes(fig_ax_override)
        
        # Extract data using base class utilities
        custom_data = self._extract_metrics_from_columns(
            self.my_df, 
            ["metric1", "metric2", "metric3"],
            prefix_to_remove="custom_"
        )
        
        # Create plot using base class utilities
        self._create_bar_plot_with_labels(
            data=custom_data,
            ax=ax,
            title="My Custom Analysis",
            xlabel="Metrics",
            ylabel="Values",
            value_format="auto"
        )
        
        return self._finalize_plot(
            fig=fig,
            is_override=is_override,
            save=save,
            show=show,
            filename="custom_analysis"
        )
    
    def plot_all(self, save=True, show=True):
        """Generate all plots for this visualizer."""
        figures = []
        figures.append(self.plot_custom_analysis(save=save, show=show))
        return figures
    
    def create_dashboard(self, save=True, show=True):
        """Create a dashboard with all plots."""
        return self.plot_custom_analysis(save=save, show=show)

# Register with unified visualizer
visualizer = Visualizer(metrics_data=metrics_dict)
visualizer.register_visualizer("MyCustomMetrics", MyCustomVisualizer)

# Use custom visualizer
custom_plots = visualizer.visualize_metric("MyCustomMetrics", save=True)
```

## Integration with Jupyter Notebooks

QWARD visualizations work seamlessly in Jupyter notebooks:

```python
# In a Jupyter cell
%matplotlib inline

# Create visualizer
visualizer = Visualizer(scanner=scanner)

# Show plots inline
qiskit_plots = visualizer.visualize_metric("QiskitMetrics", show=True, save=False)
complexity_plots = visualizer.visualize_metric("ComplexityMetrics", show=True, save=False)

# Create dashboards inline
dashboards = visualizer.create_dashboard(show=True, save=False)
```

## Performance Tips

1. **Use Unified Visualizer**: The `Visualizer` class handles all metric types efficiently
2. **Batch Operations**: Use `visualize_all()` or `create_dashboard()` for multiple plots
3. **Output Formats**: Use PNG for quick previews, SVG for publications
4. **DPI Settings**: Use lower DPI (150) for quick analysis, higher (300+) for publications
5. **Memory Management**: The visualization system automatically handles figure cleanup

## Troubleshooting

### Common Issues

1. **Missing Data Keys**: Ensure your metrics dictionary contains the expected keys for each visualizer
2. **Empty DataFrames**: Check that your metric calculations completed successfully
3. **Plot Not Showing**: Verify your matplotlib backend settings
4. **File Permissions**: Ensure write permissions for the output directory

### Debug Information

```python
# Get summary of available metrics
visualizer = Visualizer(scanner=scanner)
visualizer.print_available_metrics()

# Get detailed metric summary
summary = visualizer.get_metric_summary()
print(summary)
```

## Complete Method Reference

### Visualizer (Unified Entry Point)

- `register_strategy(metric_name, strategy_class)`: Register custom strategies
- `get_available_metrics()`: Get list of available metrics for visualization
- `visualize_metric(metric_name, save=True, show=True)`: Create plots for specific metric
- `create_dashboard(save=True, show=True)`: Create dashboards for all metrics
- `visualize_all(save=True, show=True)`: Generate all individual plots
- `get_metric_summary()`: Get summary information about available metrics
- `print_available_metrics()`: Print detailed information about available visualizations

### Individual Visualizers

All individual visualizers share these common methods:
- `create_plot()`: Create the default plot (required by BaseVisualizer)
- `plot_all(save=True, show=True)`: Generate all individual plots
- `create_dashboard(save=True, show=True)`: Create comprehensive dashboard

Plus their specific plotting methods as detailed in each visualizer section above.

For more examples and advanced usage, see the `qward/examples/` directory, particularly:
- `example_visualizer.py`: Complete visualization examples
- `visualization_demo.py`: CircuitPerformance visualization demonstrations