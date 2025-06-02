Visualization Strategies
========================

QWARD provides specialized visualization strategies for different types of quantum circuit analysis. Each visualizer is designed to create meaningful plots for specific metric categories while maintaining a consistent interface.

QiskitVisualizer
----------------

The `QiskitVisualizer` creates visualizations for basic quantum circuit properties extracted by `QiskitMetrics`. It focuses on fundamental circuit characteristics and instruction analysis.

**Visualization Types:**
- **Basic Metrics Overview**: Circuit depth, size, qubits, and classical bits
- **Instruction Analysis**: Gate type distribution and frequency analysis
- **Scheduling Information**: Circuit timing and parallelization insights
- **Comprehensive Dashboard**: Combined view of all QiskitMetrics data

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qward import Scanner
    from qward.metrics import QiskitMetrics
    from qward.visualization import QiskitVisualizer
    
    # Create and analyze circuit
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(QiskitMetrics(circuit))
    results = scanner.calculate_metrics()
    
    # Create visualizations
    visualizer = QiskitVisualizer(results, output_dir="qiskit_plots")
    
    # Generate dashboard
    dashboard = visualizer.create_dashboard(save=True, show=False)
    
    # Generate individual plots
    all_plots = visualizer.plot_all(save=True, show=False)

.. automodule:: qward.visualization.qiskit_metrics_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.visualization.qiskit_metrics_visualizer.QiskitVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

ComplexityVisualizer
--------------------

The `ComplexityVisualizer` creates sophisticated visualizations for circuit complexity analysis from `ComplexityMetrics`. It provides insights into various complexity dimensions and their relationships.

**Visualization Types:**
- **Gate-Based Analysis**: Gate counts, T-gates, CNOT gates, multi-qubit ratios
- **Entanglement Metrics**: Entangling gate density and circuit width analysis
- **Standardized Metrics**: Circuit volume, gate density, and Clifford ratios
- **Advanced Analysis**: Parallelism factors and circuit efficiency
- **Derived Complexity**: Weighted complexity scores and comparative analysis
- **Comprehensive Dashboard**: Multi-dimensional complexity overview

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qward import Scanner
    from qward.metrics import ComplexityMetrics
    from qward.visualization import ComplexityVisualizer
    
    # Create complex circuit
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.t(0)
    circuit.t(1)
    
    scanner = Scanner(circuit=circuit)
    scanner.add_strategy(ComplexityMetrics(circuit))
    results = scanner.calculate_metrics()
    
    # Create complexity visualizations
    visualizer = ComplexityVisualizer(results, output_dir="complexity_plots")
    
    # Generate comprehensive analysis
    dashboard = visualizer.create_dashboard(save=True, show=False)
    all_plots = visualizer.plot_all(save=True, show=False)

.. automodule:: qward.visualization.complexity_metrics_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.visualization.complexity_metrics_visualizer.ComplexityVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

CircuitPerformanceVisualizer
----------------------------

The `CircuitPerformanceVisualizer` creates performance analysis visualizations from `CircuitPerformanceMetrics`. It handles both single job and multiple job scenarios with support for custom success criteria.

**Visualization Types:**
- **Success Analysis**: Success rates, error rates, and performance trends
- **Fidelity Analysis**: Quantum fidelity measurements and distributions
- **Statistical Analysis**: Entropy, uniformity, and concentration metrics
- **Multi-Job Comparison**: Comparative analysis across multiple executions
- **Performance Dashboard**: Comprehensive performance overview

**Key Features:**
- **Flexible Data Handling**: Automatically detects single vs. multiple job scenarios
- **Custom Success Criteria**: Visualizes user-defined success conditions
- **Statistical Insights**: Advanced statistical analysis of quantum measurements
- **Performance Trends**: Identifies patterns in circuit execution performance

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qward import Scanner
    from qward.metrics import CircuitPerformanceMetrics
    from qward.visualization import CircuitPerformanceVisualizer
    
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
    
    scanner = Scanner(circuit=circuit, job=job)
    scanner.add_strategy(CircuitPerformanceMetrics(
        circuit=circuit, 
        job=job, 
        success_criteria=bell_state_success
    ))
    results = scanner.calculate_metrics()
    
    # Create performance visualizations
    visualizer = CircuitPerformanceVisualizer(results, output_dir="performance_plots")
    
    # Generate performance analysis
    dashboard = visualizer.create_dashboard(save=True, show=False)
    all_plots = visualizer.plot_all(save=True, show=False)

.. automodule:: qward.visualization.circuit_performance_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.visualization.circuit_performance_visualizer.CircuitPerformanceVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: 