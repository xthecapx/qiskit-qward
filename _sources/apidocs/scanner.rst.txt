Scanner
=======

The Scanner class is the main entry point for analyzing quantum circuits in QWARD. It orchestrates metric calculation and automatically converts schema objects to DataFrames for analysis and visualization.

**Key Features:**
- **Unified Interface**: Works seamlessly with all metric calculators
- **Automatic Conversion**: Converts schema objects to DataFrames via `to_flat_dict()`
- **Flexible Input**: Accepts circuits, jobs, and metric calculator instances or classes
- **DataFrame Output**: Returns dictionary of pandas DataFrames for analysis

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qward import Scanner
    from qward.metrics import QiskitMetrics, ComplexityMetrics, CircuitPerformanceMetrics
    
    # Create circuit and execute
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1000)
    
    # Create scanner and add metrics
    scanner = Scanner(circuit=circuit, job=job)
    scanner.add_strategy(QiskitMetrics(circuit))
    scanner.add_strategy(ComplexityMetrics(circuit))
    scanner.add_strategy(CircuitPerformanceMetrics(circuit=circuit, job=job))
    
    # Calculate metrics (returns DataFrames)
    results = scanner.calculate_metrics()
    
    # Access DataFrames for analysis
    print("Available metrics:", list(results.keys()))
    print("QiskitMetrics shape:", results["QiskitMetrics"].shape)
    print("ComplexityMetrics shape:", results["ComplexityMetrics"].shape)

**Integration with Schema API:**

The Scanner automatically detects when metric calculators return schema objects and converts them to flat dictionaries for DataFrame creation, while preserving all the benefits of schema validation.

.. code-block:: python

    # Schema objects are automatically converted
    # QiskitMetrics.get_metrics() → QiskitMetricsSchema → DataFrame
    # ComplexityMetrics.get_metrics() → ComplexityMetricsSchema → DataFrame
    # CircuitPerformanceMetrics.get_metrics() → CircuitPerformanceSchema → DataFrame

API Reference
-------------

.. automodule:: qward.scanner
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.scanner.Scanner
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: 