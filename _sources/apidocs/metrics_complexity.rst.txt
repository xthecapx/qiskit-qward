Complexity Metrics
==================

ComplexityMetrics calculates comprehensive circuit complexity analysis based on research literature using the unified schema-based API.

**Key Features:**
- **Unified API**: Single `get_metrics()` method returns `ComplexityMetricsSchema`
- **Research-Based**: Implements metrics from "Character Complexity: A Novel Measure for Quantum Circuit Analysis" by D. Shami
- **Comprehensive Analysis**: Gate-based, entanglement, standardized, advanced, and derived metrics
- **Type Safety**: Full validation with IDE support

**Metric Categories:**
- **Gate-Based Metrics**: Gate counts, T-count, CNOT count, multi-qubit ratios
- **Entanglement Metrics**: Entangling gate density, entangling width
- **Standardized Metrics**: Circuit volume, gate density, Clifford ratio
- **Advanced Metrics**: Parallelism factor, circuit efficiency
- **Derived Metrics**: Weighted complexity scores

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qward.metrics import ComplexityMetrics
    
    # Create circuit
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    
    # Analyze with type-safe API
    complexity_metrics = ComplexityMetrics(circuit)
    metrics = complexity_metrics.get_metrics()  # Returns ComplexityMetricsSchema
    
    # Access validated complexity data
    print(f"Gate count: {metrics.gate_based_metrics.gate_count}")
    print(f"T-gate count: {metrics.gate_based_metrics.t_count}")
    print(f"Circuit volume: {metrics.standardized_metrics.circuit_volume}")
    print(f"Parallelism factor: {metrics.advanced_metrics.parallelism_factor:.3f}")

API Reference
-------------

.. automodule:: qward.metrics.complexity_metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.metrics.complexity_metrics.ComplexityMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: 