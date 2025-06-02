# QWARD Visualization Guide

## Overview

QWARD provides a comprehensive visualization system for quantum circuit metrics with a modern, type-safe API. The visualization module follows a clean, object-oriented architecture that makes it easy to create beautiful, informative plots for your quantum computing analysis.

## New API Features (v0.9.0)

### 🎯 **Type-Safe Constants**
- **`Metrics`** constants: `Metrics.QISKIT`, `Metrics.COMPLEXITY`, `Metrics.CIRCUIT_PERFORMANCE`
- **`Plots`** constants: `Plots.QISKIT.CIRCUIT_STRUCTURE`, `Plots.COMPLEXITY.COMPLEXITY_RADAR`, etc.
- **IDE Autocompletion**: Full IntelliSense support for all plot names
- **Error Prevention**: Compile-time detection of typos in metric and plot names

### 🔍 **Rich Plot Metadata**
- **Plot Descriptions**: Detailed information about what each plot shows
- **Plot Types**: Categorized as bar charts, radar charts, line plots, etc.
- **Dependencies**: Information about required data columns
- **Categories**: Organized by analysis type (structure, performance, complexity)

### ⚡ **Granular Plot Control**
- **Single Plot Generation**: `generate_plot(metric, plot_name)`
- **Selected Plots**: `generate_plots({metric: [plot1, plot2]})`
- **All Plots**: `generate_plots({metric: None})`
- **Memory Efficient**: Default `save=False, show=False` for batch processing

### 🛡️ **Enhanced Error Handling**
- **Validation**: Automatic validation of metric and plot name combinations
- **Clear Messages**: Descriptive error messages for missing data or invalid requests
- **Graceful Fallbacks**: Robust handling of edge cases

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
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +save_plot(fig, filename)
        +show_plot(fig)
        +get_available_plots()* List[str]
        +get_plot_metadata(plot_name)* PlotMetadata
        +generate_plot(plot_name, save, show)* Figure
        +generate_plots(plot_names, save, show)* List[Figure]
        +generate_all_plots(save, show) List[Figure]
        +create_dashboard(save, show)*
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
    
    class PlotMetadata {
        +name: str
        +method_name: str
        +description: str
        +plot_type: PlotType
        +filename: str
        +dependencies: List[str]
        +category: str
    }
    
    class QiskitVisualizer {
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +generate_plot(plot_name, save, show)
        +get_available_plots() List[str]
        +get_plot_metadata(plot_name) PlotMetadata
        +plot_circuit_structure()
        +plot_gate_distribution()
        +plot_instruction_metrics()
        +plot_circuit_summary()
        +create_dashboard()
    }
    
    class ComplexityVisualizer {
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +generate_plot(plot_name, save, show)
        +get_available_plots() List[str]
        +get_plot_metadata(plot_name) PlotMetadata
        +plot_gate_based_metrics()
        +plot_complexity_radar()
        +plot_efficiency_metrics()
        +create_dashboard()
    }
    
    class CircuitPerformanceVisualizer {
        +PLOT_REGISTRY: Dict[str, PlotMetadata]
        +generate_plot(plot_name, save, show)
        +get_available_plots() List[str]
        +get_plot_metadata(plot_name) PlotMetadata
        +plot_success_error_comparison()
        +plot_fidelity_comparison()
        +plot_shot_distribution()
        +plot_aggregate_summary()
        +create_dashboard()
    }
    
    class Visualizer {
        +get_available_plots() Dict[str, List[str]]
        +get_plot_metadata(metric, plot_name) PlotMetadata
        +generate_plot(metric, plot_name, save, show) Figure
        +generate_plots(selections, save, show) Dict[str, List[Figure]]
        +create_dashboard(save, show) Dict[str, Figure]
        +register_strategy()
        +get_available_metrics()
        +get_metric_summary()
        +print_available_metrics()
    }

    VisualizationStrategy <|-- QiskitVisualizer
    VisualizationStrategy <|-- ComplexityVisualizer
    VisualizationStrategy <|-- CircuitPerformanceVisualizer
    VisualizationStrategy --> PlotConfig : uses
    VisualizationStrategy --> PlotMetadata : defines
    Visualizer --> VisualizationStrategy : manages
    
    note for VisualizationStrategy "Abstract base class with plot registry and metadata system"
    note for QiskitVisualizer "4 plots: circuit_structure, gate_distribution, instruction_metrics, circuit_summary"
    note for ComplexityVisualizer "3 plots: gate_based_metrics, complexity_radar, efficiency_metrics"
    note for CircuitPerformanceVisualizer "4 plots: success_error_comparison, fidelity_comparison, shot_distribution, aggregate_summary"
    note for Visualizer "Unified entry point with type-safe constants and granular control"
```

## Quick Start

### Using the New Type-Safe API (Recommended)

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qward import Scanner
from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
from qward.visualization import Visualizer
from qward.visualization.constants import Metrics, Plots

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

# NEW API: Generate specific plots with type-safe constants
selected_plots = visualizer.generate_plots({
    Metrics.QISKIT: [
        Plots.QISKIT.CIRCUIT_STRUCTURE,
        Plots.QISKIT.GATE_DISTRIBUTION
    ],
    Metrics.COMPLEXITY: [
        Plots.COMPLEXITY.COMPLEXITY_RADAR
    ]
}, save=True, show=False)

# NEW API: Generate all plots for specific metrics
all_qiskit_plots = visualizer.generate_plots({
    Metrics.QISKIT: None  # None = all plots
}, save=True, show=False)

# NEW API: Generate single plot
single_plot = visualizer.generate_plot(
    Metrics.CIRCUIT_PERFORMANCE, 
    Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON, 
    save=True, 
    show=False
)

# NEW API: Explore available plots and metadata
available_plots = visualizer.get_available_plots()
for metric_name, plot_names in available_plots.items():
    print(f"\n{metric_name} ({len(plot_names)} plots):")
    for plot_name in plot_names:
        metadata = visualizer.get_plot_metadata(metric_name, plot_name)
        print(f"  - {plot_name}: {metadata.description} ({metadata.plot_type.value})")

# Create comprehensive dashboards (unchanged)
dashboards = visualizer.create_dashboard(save=True, show=False)
print(f"Created {len(dashboards)} dashboards")
```

### Memory-Efficient Batch Processing

```python
# NEW: Default save=False, show=False for memory efficiency
for circuit_variant in circuit_variants:
    scanner = Scanner(circuit=circuit_variant, strategies=[QiskitMetrics, ComplexityMetrics])
    visualizer = Visualizer(scanner=scanner, output_dir=f"analysis_{circuit_variant.name}")
    
    # Generate all plots without displaying (memory efficient)
    all_plots = visualizer.generate_plots({
        Metrics.QISKIT: None,
        Metrics.COMPLEXITY: None
    })  # Default: save=False, show=False
    
    # Only save specific plots of interest
    important_plots = visualizer.generate_plots({
        Metrics.QISKIT: [Plots.QISKIT.CIRCUIT_STRUCTURE],
        Metrics.COMPLEXITY: [Plots.COMPLEXITY.COMPLEXITY_RADAR]
    }, save=True)
```

## Examples and Usage Patterns

### 📁 Available Examples

QWARD provides comprehensive examples in the `qward/examples/` directory:

- **`test_new_visualization_api.py`** - Comprehensive test suite demonstrating all new API features
- **`new_api_usage_example.py`** - Practical usage patterns with the new type-safe API
- **`example_visualizer.py`** - Complete workflow examples showing all features of the unified Visualizer
- **`visualization_demo.py`** - Focused demo of CircuitPerformanceVisualizer capabilities
- **`direct_strategy_example.py`** - Shows how to use visualization strategies directly for maximum control
- **`aer.py`** - Integration examples with Qiskit Aer simulator
- **`visualization_quickstart.py`** - Quick start example with minimal setup

### 🎨 Two Approaches to Visualization

#### 1. Unified Visualizer with Type-Safe Constants (Recommended)

Use the `Visualizer` class with constants for most cases - it provides type safety and prevents errors:

```python
from qward.visualization import Visualizer
from qward.visualization.constants import Metrics, Plots

# From Scanner
visualizer = Visualizer(scanner=scanner)

# NEW API: Type-safe plot generation
visualizer.generate_plots({
    Metrics.QISKIT: [Plots.QISKIT.CIRCUIT_STRUCTURE],
    Metrics.COMPLEXITY: None  # All plots
})

# NEW API: Explore available plots
available_plots = visualizer.get_available_plots()
metadata = visualizer.get_plot_metadata(Metrics.QISKIT, Plots.QISKIT.CIRCUIT_STRUCTURE)
print(f"Plot description: {metadata.description}")
```

**Benefits:**
- **Type Safety**: Constants prevent typos and provide IDE autocompletion
- **Rich Metadata**: Detailed information about each plot
- **Granular Control**: Generate exactly the plots you need
- **Memory Efficient**: Default save=False, show=False for batch processing
- **Error Prevention**: Validation of metric and plot combinations

#### 2. Direct Strategy Usage (Advanced)

Use individual strategies directly when you need fine-grained control:

```python
from qward.visualization import QiskitVisualizer
from qward.visualization.constants import Plots

# Use specific strategies directly with new API
qiskit_strategy = QiskitVisualizer(metrics_dict=qiskit_data, output_dir="custom_dir")

# NEW API: Generate specific plots
qiskit_strategy.generate_plot(Plots.QISKIT.CIRCUIT_STRUCTURE, save=True, show=False)

# NEW API: Get plot metadata
metadata = qiskit_strategy.get_plot_metadata(Plots.QISKIT.CIRCUIT_STRUCTURE)
print(f"Plot type: {metadata.plot_type.value}")
print(f"Dependencies: {metadata.dependencies}")

# Create dashboard (unchanged)
qiskit_strategy.create_dashboard(save=True, show=False)
```

**Benefits:**
- **Fine-grained control** over individual plots
- **Custom configurations** per strategy
- **Integration** with external data sources
- **Flexible output** formats and locations

### 🔧 Available Strategies

| Strategy | Metrics Type | Available Plots | Key Features |
|----------|-------------|----------------|--------------|
| `QiskitVisualizer` | QiskitMetrics | 4 plots | Circuit structure, gate distribution, instruction metrics, circuit summary |
| `ComplexityVisualizer` | ComplexityMetrics | 3 plots | Gate-based metrics, complexity radar, efficiency metrics |
| `CircuitPerformanceVisualizer` | CircuitPerformanceMetrics | 4 plots | Success rates, fidelity, shot distribution, aggregate summary |

### 🎛️ Plot Discovery and Metadata

#### Exploring Available Plots

```python
from qward.visualization.constants import Metrics, Plots

# Get all available plots
visualizer = Visualizer(scanner=scanner)
available_plots = visualizer.get_available_plots()

print("Available plots by metric:")
for metric_name, plot_names in available_plots.items():
    print(f"\n{metric_name} ({len(plot_names)} plots):")
    for plot_name in plot_names:
        metadata = visualizer.get_plot_metadata(metric_name, plot_name)
        print(f"  - {plot_name}")
        print(f"    Description: {metadata.description}")
        print(f"    Type: {metadata.plot_type.value}")
        print(f"    Category: {metadata.category}")
```

#### Using Constants for Type Safety

```python
# Type-safe constants prevent errors
from qward.visualization.constants import Metrics, Plots

# ✅ This works - IDE provides autocompletion
visualizer.generate_plot(Metrics.QISKIT, Plots.QISKIT.CIRCUIT_STRUCTURE)

# ❌ This would cause an error at runtime
# visualizer.generate_plot(Metrics.QISKIT, Plots.COMPLEXITY.COMPLEXITY_RADAR)  # Wrong combination!

# ✅ IDE autocompletion shows all available options
selected_plots = visualizer.generate_plots({
    Metrics.QISKIT: [
        Plots.QISKIT.CIRCUIT_STRUCTURE,    # IDE suggests these
        Plots.QISKIT.GATE_DISTRIBUTION,   # as you type
        Plots.QISKIT.INSTRUCTION_METRICS,
        Plots.QISKIT.CIRCUIT_SUMMARY
    ]
})
```

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

# Generate plots with custom styling
visualizer.generate_plots({
    Metrics.QISKIT: [Plots.QISKIT.CIRCUIT_STRUCTURE]
}, save=True, show=False)
```

#### Custom Strategies with Plot Registry

```python
from qward.visualization import VisualizationStrategy, PlotMetadata, PlotType

class MyCustomStrategy(VisualizationStrategy):
    # NEW: Define plot registry with metadata
    PLOT_REGISTRY = {
        "custom_plot": PlotMetadata(
            name="custom_plot",
            method_name="plot_custom_analysis",
            description="Custom analysis visualization",
            plot_type=PlotType.BAR_CHART,
            filename="custom_analysis",
            dependencies=["custom.metric"],
            category="custom"
        )
    }
    
    @classmethod
    def get_available_plots(cls):
        return list(cls.PLOT_REGISTRY.keys())
    
    @classmethod
    def get_plot_metadata(cls, plot_name):
        if plot_name not in cls.PLOT_REGISTRY:
            raise ValueError(f"Plot '{plot_name}' not found")
        return cls.PLOT_REGISTRY[plot_name]
    
    def generate_plot(self, plot_name, save=False, show=False):
        if plot_name not in self.PLOT_REGISTRY:
            raise ValueError(f"Plot '{plot_name}' not available")
        
        metadata = self.PLOT_REGISTRY[plot_name]
        method = getattr(self, metadata.method_name)
        return method(save=save, show=show)
    
    def plot_custom_analysis(self, save=False, show=False):
        # Your custom visualization logic
        fig, ax = plt.subplots(figsize=self.config.figsize)
        # ... plotting code ...
        
        if save:
            self.save_plot(fig, "custom_analysis")
        if show:
            self.show_plot(fig)
        return fig
    
    def create_dashboard(self, save=False, show=False):
        # Your custom dashboard logic
        return self.plot_custom_analysis(save=save, show=show)

# Register and use
visualizer.register_strategy("MyMetrics", MyCustomStrategy)
```

### 🚀 Running the Examples

```bash
# Run comprehensive new API test suite
python qward/examples/test_new_visualization_api.py

# Run practical usage examples with new API
python qward/examples/new_api_usage_example.py

# Run main visualizer examples (updated with new API)
python qward/examples/example_visualizer.py

# Run CircuitPerformanceMetrics demo (updated with new API)
python qward/examples/visualization_demo.py

# Run direct strategy examples (updated with new API)
python qward/examples/direct_strategy_example.py

# Run Aer integration examples (updated with new API)
python qward/examples/aer.py

# Quick start example (updated with new API)
python qward/examples/visualization_quickstart.py
```

### 📊 Output

All examples save plots to `qward/examples/img/` by default. You'll find:

- **Dashboards**: Comprehensive multi-plot views
- **Individual plots**: Specific visualizations for detailed analysis
- **Multiple formats**: PNG (default), PDF, SVG support
- **Organized structure**: Plots organized by metric type and analysis

### 💡 Best Practices

1. **Use type-safe constants** - Import `Metrics` and `Plots` from `qward.visualization.constants`
2. **Start with plot discovery** - Use `get_available_plots()` and `get_plot_metadata()` to explore
3. **Leverage granular control** - Generate only the plots you need with `generate_plots()`
4. **Use memory-efficient defaults** - Default `save=False, show=False` prevents unwanted files
5. **Customize PlotConfig** for consistent styling across all visualizations
6. **Register custom strategies** to extend the system with your own visualizations
7. **Check the output directory** - all plots are saved with descriptive names
8. **Use meaningful output directories** - organize plots by analysis type or date

### 🔗 Key Benefits

- **Type Safety**: Constants prevent typos and provide IDE autocompletion
- **Rich Metadata**: Detailed information about each plot's purpose and requirements
- **Granular Control**: Generate exactly the plots you need
- **Memory Efficient**: Default parameters optimized for batch processing
- **Strategy Pattern**: Consistent with Scanner architecture
- **Auto-Detection**: Automatically finds and visualizes available metrics
- **Extensible**: Easy to add new visualization strategies
- **Flexible**: Use unified interface or direct strategies as needed
- **Customizable**: Full control over plot appearance and output formats

## Available Visualizers

### QiskitVisualizer

Visualizes circuit structure and instruction analysis from `QiskitMetrics`.

#### Available Plots (New API)

```python
from qward.visualization.constants import Plots

# All available QiskitVisualizer plots:
Plots.QISKIT.CIRCUIT_STRUCTURE    # Basic circuit metrics
Plots.QISKIT.GATE_DISTRIBUTION    # Gate type analysis  
Plots.QISKIT.INSTRUCTION_METRICS  # Instruction-related metrics
Plots.QISKIT.CIRCUIT_SUMMARY      # Derived metrics summary
```

#### Individual Plots (New API)

1. **Circuit Structure**
   ```python
   # NEW API
   qiskit_viz.generate_plot(Plots.QISKIT.CIRCUIT_STRUCTURE, save=True, show=False)
   
   # Get plot metadata
   metadata = qiskit_viz.get_plot_metadata(Plots.QISKIT.CIRCUIT_STRUCTURE)
   print(f"Description: {metadata.description}")
   ```
   Shows basic circuit metrics: depth, width, size, qubits, and classical bits.

2. **Gate Distribution**
   ```python
   # NEW API
   qiskit_viz.generate_plot(Plots.QISKIT.GATE_DISTRIBUTION, save=True, show=False)
   ```
   Displays gate type analysis and instruction distribution as a pie chart.

3. **Instruction Metrics**
   ```python
   # NEW API
   qiskit_viz.generate_plot(Plots.QISKIT.INSTRUCTION_METRICS, save=True, show=False)
   ```
   Shows instruction-related metrics like connected components and nonlocal gates.

4. **Circuit Summary**
   ```python
   # NEW API
   qiskit_viz.generate_plot(Plots.QISKIT.CIRCUIT_SUMMARY, save=True, show=False)
   ```
   Displays derived metrics like gate density and parallelism.

#### Dashboard and All Plots (New API)

```python
# Create comprehensive dashboard (unchanged)
dashboard = qiskit_viz.create_dashboard(save=True, show=False)

# NEW API: Generate all individual plots
all_figures = qiskit_viz.generate_all_plots(save=True, show=False)

# NEW API: Generate selected plots
selected_figures = qiskit_viz.generate_plots([
    Plots.QISKIT.CIRCUIT_STRUCTURE,
    Plots.QISKIT.GATE_DISTRIBUTION
], save=True, show=False)

# NEW API: Get available plots and metadata
available_plots = qiskit_viz.get_available_plots()
for plot_name in available_plots:
    metadata = qiskit_viz.get_plot_metadata(plot_name)
    print(f"{plot_name}: {metadata.description}")
```

### ComplexityVisualizer

Visualizes complexity analysis from `ComplexityMetrics` with advanced charts and analysis.

#### Available Plots (New API)

```python
from qward.visualization.constants import Plots

# All available ComplexityVisualizer plots:
Plots.COMPLEXITY.GATE_BASED_METRICS  # Gate counts and depth analysis
Plots.COMPLEXITY.COMPLEXITY_RADAR    # Normalized complexity indicators
Plots.COMPLEXITY.EFFICIENCY_METRICS  # Parallelism and efficiency analysis
```

#### Individual Plots (New API)

1. **Gate-Based Metrics**
   ```python
   # NEW API
   complexity_viz.generate_plot(Plots.COMPLEXITY.GATE_BASED_METRICS, save=True, show=False)
   ```
   Shows gate counts, circuit depth, T-gates, and CNOT gates.

2. **Complexity Radar Chart**
   ```python
   # NEW API
   complexity_viz.generate_plot(Plots.COMPLEXITY.COMPLEXITY_RADAR, save=True, show=False)
   ```
   Creates a radar chart with normalized complexity indicators for quick visual assessment.

3. **Efficiency Metrics**
   ```python
   # NEW API
   complexity_viz.generate_plot(Plots.COMPLEXITY.EFFICIENCY_METRICS, save=True, show=False)
   ```
   Shows parallelism efficiency and circuit efficiency analysis.

#### Dashboard and All Plots (New API)

```python
# Create comprehensive dashboard (unchanged)
dashboard = complexity_viz.create_dashboard(save=True, show=False)

# NEW API: Generate all individual plots
all_figures = complexity_viz.generate_all_plots(save=True, show=False)

# NEW API: Generate selected plots
selected_figures = complexity_viz.generate_plots([
    Plots.COMPLEXITY.COMPLEXITY_RADAR,
    Plots.COMPLEXITY.EFFICIENCY_METRICS
], save=True, show=False)
```

### CircuitPerformanceVisualizer

Visualizes performance metrics from `CircuitPerformanceMetrics` with success rates, fidelity, and execution analysis.

#### Available Plots (New API)

```python
from qward.visualization.constants import Plots

# All available CircuitPerformanceVisualizer plots:
Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON  # Success vs error rates
Plots.CIRCUIT_PERFORMANCE.FIDELITY_COMPARISON       # Fidelity across jobs
Plots.CIRCUIT_PERFORMANCE.SHOT_DISTRIBUTION         # Successful vs failed shots
Plots.CIRCUIT_PERFORMANCE.AGGREGATE_SUMMARY         # Statistical summary
```

#### Individual Plots (New API)

1. **Success vs Error Rate Comparison**
   ```python
   # NEW API
   perf_viz.generate_plot(Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON, save=True, show=False)
   ```
   Shows success and error rates across different jobs as a grouped bar chart.

2. **Fidelity Comparison**
   ```python
   # NEW API
   perf_viz.generate_plot(Plots.CIRCUIT_PERFORMANCE.FIDELITY_COMPARISON, save=True, show=False)
   ```
   Displays fidelity values for each job with value labels.

3. **Shot Distribution**
   ```python
   # NEW API
   perf_viz.generate_plot(Plots.CIRCUIT_PERFORMANCE.SHOT_DISTRIBUTION, save=True, show=False)
   ```
   Shows the distribution of successful vs failed shots as stacked bars with detailed labels.

4. **Aggregate Summary**
   ```python
   # NEW API
   perf_viz.generate_plot(Plots.CIRCUIT_PERFORMANCE.AGGREGATE_SUMMARY, save=True, show=False)
   ```
   Provides a comprehensive summary of aggregate statistics across multiple jobs.

#### Dashboard and All Plots (New API)

```python
# Create comprehensive dashboard (unchanged)
dashboard = perf_viz.create_dashboard(save=True, show=False)

# NEW API: Generate all individual plots
all_figures = perf_viz.generate_all_plots(save=True, show=False)

# NEW API: Generate selected plots
selected_figures = perf_viz.generate_plots([
    Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON,
    Plots.CIRCUIT_PERFORMANCE.FIDELITY_COMPARISON
], save=True, show=False)
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

### Complete Analysis with Custom Configuration and New API

```python
from qward.visualization import Visualizer, PlotConfig
from qward.visualization.constants import Metrics, Plots

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
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
scanner.add_strategy(CircuitPerformanceMetrics(circuit=circuit, job=job))

# Use unified visualizer with custom config
visualizer = Visualizer(
    scanner=scanner, 
    config=config, 
    output_dir="comprehensive_analysis"
)

# NEW API: Create all visualizations with type-safe constants
dashboards = visualizer.create_dashboard(save=True, show=False)

# NEW API: Generate specific plots for analysis
analysis_plots = visualizer.generate_plots({
    Metrics.QISKIT: [
        Plots.QISKIT.CIRCUIT_STRUCTURE,
        Plots.QISKIT.GATE_DISTRIBUTION
    ],
    Metrics.COMPLEXITY: [
        Plots.COMPLEXITY.COMPLEXITY_RADAR,
        Plots.COMPLEXITY.EFFICIENCY_METRICS
    ],
    Metrics.CIRCUIT_PERFORMANCE: None  # All plots
}, save=True, show=False)

# NEW API: Explore what was created
print("Generated visualizations:")
for metric_name, figures in analysis_plots.items():
    print(f"  {metric_name}: {len(figures)} plots")
    
    # Get metadata for each plot
    available_plots = visualizer.get_available_plots(metric_name)
    for plot_name in available_plots[metric_name]:
        metadata = visualizer.get_plot_metadata(metric_name, plot_name)
        print(f"    - {plot_name}: {metadata.description}")

# Print summary
visualizer.print_available_metrics()
```

### Multiple Jobs Analysis with New API

```python
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qward.visualization.constants import Metrics, Plots

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

# Analyze with CircuitPerformanceMetrics
circuit_performance = CircuitPerformanceMetrics(circuit=circuit)
for job in jobs:
    circuit_performance.add_job(job)

scanner = Scanner(circuit=circuit)
scanner.add_strategy(circuit_performance)
metrics_dict = scanner.calculate_metrics()

# NEW API: Visualize noise effects with granular control
perf_viz = CircuitPerformanceVisualizer(
    {k: v for k, v in metrics_dict.items() if k.startswith("CircuitPerformance")},
    output_dir="noise_analysis"
)

# Generate specific plots to analyze noise impact
noise_analysis_plots = perf_viz.generate_plots([
    Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON,
    Plots.CIRCUIT_PERFORMANCE.FIDELITY_COMPARISON
], save=True, show=False)

# Create comprehensive dashboard
dashboard = perf_viz.create_dashboard(save=True, show=False)

# NEW API: Get plot metadata to understand what was generated
for plot_name in perf_viz.get_available_plots():
    metadata = perf_viz.get_plot_metadata(plot_name)
    print(f"{plot_name}: {metadata.description} ({metadata.plot_type.value})")
```

### Custom Success Criteria with New API

```python
from qward.visualization.constants import Metrics, Plots

# Define custom success criteria for Bell state
def bell_state_success(outcome):
    """Success for Bell state: |00⟩ or |11⟩"""
    clean = outcome.replace(" ", "")
    return clean in ["00", "11"]

# Use custom criteria
circuit_performance = CircuitPerformanceMetrics(
    circuit=circuit,
    job=job,
    success_criteria=bell_state_success
)

scanner = Scanner(circuit=circuit)
scanner.add_strategy(QiskitMetrics(circuit))
scanner.add_strategy(ComplexityMetrics(circuit))
scanner.add_strategy(circuit_performance)

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

# NEW API: Create comprehensive analysis with type-safe constants
dashboards = visualizer.create_dashboard(save=True, show=False)

# NEW API: Generate specific plots for Bell state analysis
bell_analysis = visualizer.generate_plots({
    Metrics.QISKIT: [Plots.QISKIT.CIRCUIT_STRUCTURE],
    Metrics.COMPLEXITY: [Plots.COMPLEXITY.COMPLEXITY_RADAR],
    Metrics.CIRCUIT_PERFORMANCE: [
        Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON,
        Plots.CIRCUIT_PERFORMANCE.FIDELITY_COMPARISON
    ]
}, save=True, show=False)

# NEW API: Explore the analysis results
print("Bell State Analysis Results:")
for metric_name, figures in bell_analysis.items():
    print(f"\n{metric_name}:")
    available_plots = visualizer.get_available_plots(metric_name)
    for plot_name in available_plots[metric_name]:
        if any(plot_name in generated_plots for generated_plots in bell_analysis.values()):
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            print(f"  ✅ {plot_name}: {metadata.description}")
```

### Circuit Comparison Workflow with New API

```python
from qward.visualization.constants import Metrics, Plots

# Compare different circuit implementations
circuits = {
    "basic_bell": create_bell_circuit(),
    "optimized_bell": create_optimized_bell_circuit(),
    "noisy_bell": create_bell_circuit_with_noise()
}

comparison_results = {}

for circuit_name, circuit in circuits.items():
    # Analyze each circuit
    scanner = Scanner(circuit=circuit, strategies=[QiskitMetrics, ComplexityMetrics])
    
    # Create visualizer for this circuit
    visualizer = Visualizer(
        scanner=scanner, 
        output_dir=f"comparison/{circuit_name}"
    )
    
    # NEW API: Generate comparison plots
    comparison_plots = visualizer.generate_plots({
        Metrics.QISKIT: [
            Plots.QISKIT.CIRCUIT_STRUCTURE,
            Plots.QISKIT.GATE_DISTRIBUTION
        ],
        Metrics.COMPLEXITY: [
            Plots.COMPLEXITY.GATE_BASED_METRICS,
            Plots.COMPLEXITY.COMPLEXITY_RADAR
        ]
    }, save=True, show=False)
    
    comparison_results[circuit_name] = comparison_plots

# NEW API: Generate summary report
print("Circuit Comparison Summary:")
for circuit_name in circuits.keys():
    print(f"\n{circuit_name}:")
    
    # Get metrics for comparison
    scanner = Scanner(circuit=circuits[circuit_name], strategies=[QiskitMetrics, ComplexityMetrics])
    qiskit_metrics = QiskitMetrics(circuits[circuit_name])
    complexity_metrics = ComplexityMetrics(circuits[circuit_name])
    
    qiskit_schema = qiskit_metrics.get_metrics()
    complexity_schema = complexity_metrics.get_metrics()
    
    print(f"  Depth: {qiskit_schema.basic_metrics.depth}")
    print(f"  Gate Count: {complexity_schema.gate_based_metrics.gate_count}")
    print(f"  Complexity Score: {complexity_schema.derived_metrics.weighted_complexity:.3f}")
```

## Data Format Requirements

### QiskitVisualizer Data Format

Expects a DataFrame with QiskitMetrics columns:
- `basic_metrics.*`: Circuit structure metrics (depth, width, size, qubits, classical bits)
- `instruction_metrics.*`: Gate and instruction analysis (connected components, nonlocal gates)
- `scheduling_metrics.*`: Timing information (if available)

### ComplexityVisualizer Data Format

Expects a DataFrame with ComplexityMetrics columns:
- `gate_based_metrics.*`: Gate counts and circuit depth
- `entanglement_metrics.*`: Entanglement analysis
- `standardized_metrics.*`: Normalized complexity indicators
- `advanced_metrics.*`: Efficiency and parallelism metrics
- `derived_metrics.*`: Weighted complexity scores

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

You can extend the visualization system by creating custom visualizers with the new plot registry system:

```python
from qward.visualization.base import VisualizationStrategy, PlotMetadata, PlotType
import matplotlib.pyplot as plt

class MyCustomVisualizer(VisualizationStrategy):
    # NEW: Define plot registry with rich metadata
    PLOT_REGISTRY = {
        "custom_analysis": PlotMetadata(
            name="custom_analysis",
            method_name="plot_custom_analysis",
            description="Custom quantum circuit analysis visualization",
            plot_type=PlotType.BAR_CHART,
            filename="custom_analysis",
            dependencies=["custom.metric1", "custom.metric2"],
            category="analysis"
        ),
        "custom_comparison": PlotMetadata(
            name="custom_comparison",
            method_name="plot_custom_comparison",
            description="Comparative analysis of custom metrics",
            plot_type=PlotType.LINE_PLOT,
            filename="custom_comparison",
            dependencies=["custom.metric1", "custom.metric3"],
            category="comparison"
        )
    }
    
    def __init__(self, metrics_dict, output_dir="img", config=None):
        super().__init__(output_dir, config)
        self.metrics_dict = metrics_dict
        self.my_df = metrics_dict.get("MyCustomMetrics")
        
        if self.my_df is None:
            raise ValueError("'MyCustomMetrics' data not found in metrics_dict.")
    
    @classmethod
    def get_available_plots(cls):
        """NEW API: Return list of available plot names."""
        return list(cls.PLOT_REGISTRY.keys())
    
    @classmethod
    def get_plot_metadata(cls, plot_name):
        """NEW API: Return metadata for a specific plot."""
        if plot_name not in cls.PLOT_REGISTRY:
            raise ValueError(f"Plot '{plot_name}' not found in registry")
        return cls.PLOT_REGISTRY[plot_name]
    
    def generate_plot(self, plot_name, save=False, show=False):
        """NEW API: Generate a specific plot by name."""
        if plot_name not in self.PLOT_REGISTRY:
            raise ValueError(f"Plot '{plot_name}' not available")
        
        metadata = self.PLOT_REGISTRY[plot_name]
        method = getattr(self, metadata.method_name)
        return method(save=save, show=show)
    
    def generate_plots(self, plot_names, save=False, show=False):
        """NEW API: Generate multiple specific plots."""
        figures = []
        for plot_name in plot_names:
            fig = self.generate_plot(plot_name, save=save, show=show)
            figures.append(fig)
        return figures
    
    def generate_all_plots(self, save=False, show=False):
        """NEW API: Generate all available plots."""
        return self.generate_plots(self.get_available_plots(), save=save, show=show)
    
    def plot_custom_analysis(self, save=False, show=False):
        """Create a custom analysis plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Extract data using base class utilities
        custom_data = self._extract_metrics_from_columns(
            self.my_df, 
            ["custom.metric1", "custom.metric2"],
            prefix_to_remove="custom."
        )
        
        # Create plot using base class utilities
        self._create_bar_plot_with_labels(
            data=custom_data,
            ax=ax,
            title="Custom Analysis",
            xlabel="Metrics",
            ylabel="Values",
            value_format="auto"
        )
        
        if save:
            self.save_plot(fig, "custom_analysis")
        if show:
            self.show_plot(fig)
        return fig
    
    def plot_custom_comparison(self, save=False, show=False):
        """Create a custom comparison plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Custom comparison logic here
        # ...
        
        if save:
            self.save_plot(fig, "custom_comparison")
        if show:
            self.show_plot(fig)
        return fig
    
    def create_dashboard(self, save=False, show=False):
        """Create a dashboard with all plots."""
        # Create a comprehensive dashboard
        fig, axes = plt.subplots(2, 1, figsize=(self.config.figsize[0], self.config.figsize[1] * 2))
        
        # Use existing plot methods with axis override
        self.plot_custom_analysis(save=False, show=False)
        self.plot_custom_comparison(save=False, show=False)
        
        if save:
            self.save_plot(fig, "custom_dashboard")
        if show:
            self.show_plot(fig)
        return fig

# Register with unified visualizer
visualizer = Visualizer(metrics_data=metrics_dict)
visualizer.register_strategy("MyCustomMetrics", MyCustomVisualizer)

# NEW API: Use custom visualizer with type-safe approach
custom_plots = visualizer.generate_plots({
    "MyCustomMetrics": ["custom_analysis", "custom_comparison"]
}, save=True, show=False)

# NEW API: Explore custom visualizer metadata
available_plots = visualizer.get_available_plots("MyCustomMetrics")
for plot_name in available_plots["MyCustomMetrics"]:
    metadata = visualizer.get_plot_metadata("MyCustomMetrics", plot_name)
    print(f"{plot_name}: {metadata.description} ({metadata.plot_type.value})")
```

## Integration with Jupyter Notebooks

QWARD visualizations work seamlessly in Jupyter notebooks with the new API:

```python
# In a Jupyter cell
%matplotlib inline

from qward.visualization.constants import Metrics, Plots

# Create visualizer
visualizer = Visualizer(scanner=scanner)

# NEW API: Show specific plots inline with type-safe constants
qiskit_plots = visualizer.generate_plots({
    Metrics.QISKIT: [Plots.QISKIT.CIRCUIT_STRUCTURE]
}, show=True, save=False)

complexity_plots = visualizer.generate_plots({
    Metrics.COMPLEXITY: [Plots.COMPLEXITY.COMPLEXITY_RADAR]
}, show=True, save=False)

# NEW API: Create dashboards inline
dashboards = visualizer.create_dashboard(show=True, save=False)

# NEW API: Explore available plots interactively
available_plots = visualizer.get_available_plots()
for metric_name, plot_names in available_plots.items():
    print(f"\n{metric_name} plots:")
    for plot_name in plot_names:
        metadata = visualizer.get_plot_metadata(metric_name, plot_name)
        print(f"  - {plot_name}: {metadata.description}")
```

## Performance Tips

1. **Use Type-Safe Constants**: Import `Metrics` and `Plots` for IDE autocompletion and error prevention
2. **Leverage Memory-Efficient Defaults**: Default `save=False, show=False` prevents unwanted file creation
3. **Use Granular Control**: Generate only the plots you need with `generate_plots()`
4. **Batch Operations**: Use `generate_plots()` with multiple selections for efficiency
5. **Output Formats**: Use PNG for quick previews, SVG for publications
6. **DPI Settings**: Use lower DPI (150) for quick analysis, higher (300+) for publications
7. **Plot Discovery**: Use `get_available_plots()` to explore before generating
8. **Memory Management**: The visualization system automatically handles figure cleanup

## Troubleshooting

### Common Issues

1. **Missing Data Keys**: Ensure your metrics dictionary contains the expected keys for each visualizer
2. **Empty DataFrames**: Check that your metric calculations completed successfully
3. **Plot Not Showing**: Verify your matplotlib backend settings
4. **File Permissions**: Ensure write permissions for the output directory
5. **Invalid Plot Names**: Use constants from `qward.visualization.constants` to prevent typos
6. **Metric/Plot Mismatch**: Verify plot names are valid for the specific metric type

### Debug Information

```python
# NEW API: Get comprehensive information about available visualizations
visualizer = Visualizer(scanner=scanner)

# Print available metrics and plots
visualizer.print_available_metrics()

# Get detailed metric summary
summary = visualizer.get_metric_summary()
print(summary)

# NEW API: Explore specific metric plots
available_plots = visualizer.get_available_plots()
for metric_name, plot_names in available_plots.items():
    print(f"\n{metric_name}:")
    for plot_name in plot_names:
        try:
            metadata = visualizer.get_plot_metadata(metric_name, plot_name)
            print(f"  ✅ {plot_name}: {metadata.description}")
            print(f"     Type: {metadata.plot_type.value}")
            print(f"     Dependencies: {metadata.dependencies}")
        except Exception as e:
            print(f"  ❌ {plot_name}: Error - {e}")

# NEW API: Test plot generation
try:
    test_plot = visualizer.generate_plot(
        Metrics.QISKIT, 
        Plots.QISKIT.CIRCUIT_STRUCTURE, 
        save=False, 
        show=False
    )
    print("✅ Plot generation successful")
    plt.close(test_plot)  # Clean up
except Exception as e:
    print(f"❌ Plot generation failed: {e}")
```

### Error Handling Best Practices

```python
from qward.visualization.constants import Metrics, Plots

# Robust error handling with new API
try:
    # Check if metric is available
    available_metrics = visualizer.get_available_metrics()
    if Metrics.QISKIT not in available_metrics:
        print("QiskitMetrics not available")
        return
    
    # Check if plot is available for this metric
    available_plots = visualizer.get_available_plots(Metrics.QISKIT)
    if Plots.QISKIT.CIRCUIT_STRUCTURE not in available_plots[Metrics.QISKIT]:
        print("Circuit structure plot not available")
        return
    
    # Generate plot safely
    plot = visualizer.generate_plot(
        Metrics.QISKIT, 
        Plots.QISKIT.CIRCUIT_STRUCTURE, 
        save=True, 
        show=False
    )
    print("✅ Plot generated successfully")
    
except Exception as e:
    print(f"❌ Error generating plot: {e}")
    # Log additional debug information
    print(f"Available metrics: {list(available_metrics)}")
    print(f"Available plots: {available_plots}")
```

## Complete Method Reference

### Visualizer (Unified Entry Point) - New API

#### Core Methods
- `get_available_plots(metric_name=None)`: Get available plots for all metrics or specific metric
- `get_plot_metadata(metric_name, plot_name)`: Get detailed metadata for a specific plot
- `generate_plot(metric_name, plot_name, save=False, show=False)`: Generate single plot
- `generate_plots(selections, save=False, show=False)`: Generate multiple selected plots
- `create_dashboard(save=False, show=False)`: Create dashboards for all metrics
- `get_available_metrics()`: Get list of available metrics for visualization
- `get_metric_summary()`: Get summary information about available metrics
- `print_available_metrics()`: Print detailed information about available visualizations
- `register_strategy(metric_name, strategy_class)`: Register custom visualization strategies

#### Usage Examples
```python
# Get all available plots
all_plots = visualizer.get_available_plots()

# Get plots for specific metric
qiskit_plots = visualizer.get_available_plots(Metrics.QISKIT)

# Get plot metadata
metadata = visualizer.get_plot_metadata(Metrics.QISKIT, Plots.QISKIT.CIRCUIT_STRUCTURE)

# Generate single plot
single_plot = visualizer.generate_plot(Metrics.QISKIT, Plots.QISKIT.CIRCUIT_STRUCTURE)

# Generate selected plots
selected_plots = visualizer.generate_plots({
    Metrics.QISKIT: [Plots.QISKIT.CIRCUIT_STRUCTURE, Plots.QISKIT.GATE_DISTRIBUTION],
    Metrics.COMPLEXITY: None  # All plots
})
```

### Individual Visualizers - New API

All individual visualizers (`QiskitVisualizer`, `ComplexityVisualizer`, `CircuitPerformanceVisualizer`) share these common methods:

#### Class Methods
- `get_available_plots()`: Return list of available plot names
- `get_plot_metadata(plot_name)`: Return metadata for specific plot

#### Instance Methods
- `generate_plot(plot_name, save=False, show=False)`: Generate specific plot by name
- `generate_plots(plot_names, save=False, show=False)`: Generate multiple specific plots
- `generate_all_plots(save=False, show=False)`: Generate all available plots
- `create_dashboard(save=False, show=False)`: Create comprehensive dashboard

#### Legacy Methods (Still Available)
- Individual plot methods (e.g., `plot_circuit_structure()`, `plot_complexity_radar()`)
- These methods are still available but the new `generate_plot()` API is recommended

### Constants Reference

```python
from qward.visualization.constants import Metrics, Plots

# Metric Constants
Metrics.QISKIT                    # "QiskitMetrics"
Metrics.COMPLEXITY               # "ComplexityMetrics"  
Metrics.CIRCUIT_PERFORMANCE      # "CircuitPerformance"

# QiskitMetrics Plot Constants
Plots.QISKIT.CIRCUIT_STRUCTURE   # "circuit_structure"
Plots.QISKIT.GATE_DISTRIBUTION   # "gate_distribution"
Plots.QISKIT.INSTRUCTION_METRICS # "instruction_metrics"
Plots.QISKIT.CIRCUIT_SUMMARY     # "circuit_summary"

# ComplexityMetrics Plot Constants
Plots.COMPLEXITY.GATE_BASED_METRICS  # "gate_based_metrics"
Plots.COMPLEXITY.COMPLEXITY_RADAR    # "complexity_radar"
Plots.COMPLEXITY.EFFICIENCY_METRICS  # "efficiency_metrics"

# CircuitPerformanceMetrics Plot Constants
Plots.CIRCUIT_PERFORMANCE.SUCCESS_ERROR_COMPARISON  # "success_error_comparison"
Plots.CIRCUIT_PERFORMANCE.FIDELITY_COMPARISON       # "fidelity_comparison"
Plots.CIRCUIT_PERFORMANCE.SHOT_DISTRIBUTION         # "shot_distribution"
Plots.CIRCUIT_PERFORMANCE.AGGREGATE_SUMMARY         # "aggregate_summary"
```

## Migration from Old API

### Quick Migration Guide

**Old API:**
```python
# Old methods (deprecated)
visualizer.visualize_all()
visualizer.visualize_metric("QiskitMetrics")
strategy.plot_all()
strategy.plot_circuit_structure()
```

**New API:**
```python
# New type-safe methods
from qward.visualization.constants import Metrics, Plots

visualizer.generate_plots({Metrics.QISKIT: None, Metrics.COMPLEXITY: None})
visualizer.generate_plots({Metrics.QISKIT: None})
strategy.generate_all_plots()
strategy.generate_plot(Plots.QISKIT.CIRCUIT_STRUCTURE)
```

### Benefits of Migration

1. **Type Safety**: Constants prevent typos and provide IDE autocompletion
2. **Rich Metadata**: Detailed information about each plot
3. **Granular Control**: Generate exactly the plots you need
4. **Memory Efficiency**: Better defaults for batch processing
5. **Error Prevention**: Validation of metric and plot combinations
6. **Future-Proof**: New features will be added to the new API

For more examples and advanced usage, see the `qward/examples/` directory, particularly:
- **`test_new_visualization_api.py`**: Comprehensive test suite for new API
- **`new_api_usage_example.py`**: Practical usage patterns
- **`example_visualizer.py`**: Complete visualization examples
- **`visualization_demo.py`**: CircuitPerformanceVisualizer demonstrations