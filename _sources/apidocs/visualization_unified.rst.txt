Unified Visualizer
==================

The `Visualizer` class is the main entry point for creating visualizations in QWARD. It automatically detects available metric types and orchestrates the appropriate visualization strategies to create comprehensive analysis dashboards.

**Key Features:**
- **Automatic Detection**: Identifies available metric types and selects appropriate visualizers
- **Unified Interface**: Single class handles all visualization needs
- **Comprehensive Output**: Creates both individual plots and combined dashboards
- **Flexible Configuration**: Supports custom plot configurations and output directories

**Supported Metric Types:**
- **QiskitMetrics**: Basic circuit properties and instruction analysis
- **ComplexityMetrics**: Multi-dimensional circuit complexity analysis
- **CircuitPerformanceMetrics**: Performance analysis with custom success criteria

**Usage Patterns:**

**Basic Usage:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qward import Scanner
    from qward.metrics import QiskitMetrics, ComplexityMetrics
    from qward.visualization import Visualizer
    
    # Create and analyze circuit
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    results = scanner.calculate_metrics()
    
    # Create unified visualizations
    visualizer = Visualizer(results, output_dir="analysis_plots")
    
    # Generate all visualizations
    all_plots = visualizer.create_all_visualizations(save=True, show=False)

**Advanced Configuration:**

.. code-block:: python

    from qward.visualization import Visualizer
    from qward.visualization.base import PlotConfig
    
    # Custom plot configuration
    config = PlotConfig(
        figsize=(14, 10),
        dpi=200,
        style="quantum",
        color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        save_format="svg",
        grid=True,
        alpha=0.85
    )
    
    # Create visualizer with custom configuration
    visualizer = Visualizer(
        metrics_dict=results,
        output_dir="custom_plots",
        config=config
    )
    
    # Generate customized visualizations
    dashboards = visualizer.create_dashboards(save=True, show=False)
    individual_plots = visualizer.create_individual_plots(save=True, show=False)

**Performance Analysis Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qward import Scanner
    from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
    from qward.visualization import Visualizer
    
    # Create and execute circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1000)
    
    # Define success criteria
    def bell_state_success(result: str) -> bool:
        return result.replace(" ", "") in ["00", "11"]
    
    # Comprehensive analysis
    scanner = Scanner(circuit=circuit, job=job)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    scanner.add_strategy(CircuitPerformanceMetrics(
        circuit=circuit, 
        job=job, 
        success_criteria=bell_state_success
    ))
    results = scanner.calculate_metrics()
    
    # Create comprehensive visualizations
    visualizer = Visualizer(results, output_dir="comprehensive_analysis")
    
    # Generate complete analysis suite
    complete_analysis = visualizer.create_all_visualizations(save=True, show=False)
    
    print("Generated visualizations:")
    for metric_type, plots in complete_analysis.items():
        print(f"  {metric_type}: {len(plots)} plots")

Main Visualizer Class
----------------------

.. automodule:: qward.visualization.visualizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.visualization.visualizer.Visualizer
   :members:
   :undoc-members:
   :show-inheritance: 