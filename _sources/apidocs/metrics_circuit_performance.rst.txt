Circuit Performance Metrics
============================

CircuitPerformanceMetrics analyzes quantum circuit execution performance using the unified schema-based API.

**Key Features:**
- **Unified API**: Single `get_metrics()` method returns `CircuitPerformanceSchema`
- **Custom Success Criteria**: User-defined success conditions for flexible analysis
- **Multi-Job Support**: Handles both single job and multiple job analysis
- **Type Safety**: Full validation with automatic constraint checking

**Metric Categories:**
- **Success Metrics**: Success rate, error rate, successful shots analysis
- **Fidelity Metrics**: Quantum fidelity between measured and expected distributions
- **Statistical Metrics**: Entropy, uniformity, concentration analysis

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qward.metrics import CircuitPerformanceMetrics
    
    # Create and execute circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1000)
    
    # Define custom success criteria
    def bell_state_success(result: str) -> bool:
        clean_result = result.replace(" ", "")
        return clean_result in ["00", "11"]  # |00⟩ or |11⟩ states
    
    # Analyze with type-safe API
    circuit_performance = CircuitPerformanceMetrics(
        circuit=circuit, 
        job=job, 
        success_criteria=bell_state_success
    )
    metrics = circuit_performance.get_metrics()  # Returns CircuitPerformanceSchema
    
    # Access validated performance data
    print(f"Success rate: {metrics.success_metrics.success_rate:.3f}")
    print(f"Error rate: {metrics.success_metrics.error_rate:.3f}")
    print(f"Fidelity: {metrics.fidelity_metrics.fidelity:.3f}")
    print(f"Entropy: {metrics.statistical_metrics.entropy:.3f}")

API Reference
-------------

.. automodule:: qward.metrics.circuit_performance
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.metrics.circuit_performance.CircuitPerformanceMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: 