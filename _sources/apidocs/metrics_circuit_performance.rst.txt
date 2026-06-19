Circuit Performance Metrics
============================

FidelityMetrics analyzes quantum circuit execution performance using the unified schema-based API.

**Key Features:**
- **Unified API**: Single `get_metrics()` method returns `FidelitySchema`
- **Custom Success Criteria**: User-defined success conditions for flexible analysis
- **Multi-Job Support**: Handles both single job and multiple job analysis
- **Type Safety**: Full validation with automatic constraint checking

**Metric Categories:**
- **Success Metrics**: Success rate, error rate, successful shots analysis
- **Statistical Metrics**: Entropy, uniformity, concentration analysis

**Usage Example:**

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qward import FidelityMetrics
    
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
    circuit_performance = FidelityMetrics(
        circuit=circuit, 
        job=job, 
        success_criteria=bell_state_success
    )
    metrics = circuit_performance.get_metrics()  # Returns FidelitySchema

    # Access validated fidelity data
    print(f"Success rate: {metrics.success_rate:.3f}")
    print(f"DSR: {metrics.dsr:.3f}")
    print(f"Hellinger fidelity: {metrics.hellinger_fidelity:.3f}")

API Reference
-------------

.. automodule:: qward.metrics.fidelity_metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.metrics.fidelity_metrics.FidelityMetrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members: 
