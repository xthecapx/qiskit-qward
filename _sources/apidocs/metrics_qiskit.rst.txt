Qiskit Metrics
==============

QiskitMetrics extracts basic circuit properties directly from QuantumCircuit objects using the unified schema-based API.

**Key Features:**
- **Unified API**: Single `get_metrics()` method returns `QiskitMetricsSchema`
- **Type Safety**: Full IDE support with automatic validation
- **Comprehensive Coverage**: Basic metrics, instruction analysis, and scheduling information

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qward.metrics import QiskitMetrics
    
    # Create circuit
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Analyze with type-safe API
    qiskit_metrics = QiskitMetrics(circuit)
    metrics = qiskit_metrics.get_metrics()  # Returns QiskitMetricsSchema
    
    # Access validated data
    print(f"Depth: {metrics.basic_metrics.depth}")
    print(f"Gate count: {metrics.basic_metrics.size}")
    print(f"Qubits: {metrics.basic_metrics.num_qubits}")

API Reference
-------------

.. automodule:: qward.metrics.qiskit_metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.metrics.qiskit_metrics.QiskitMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: 