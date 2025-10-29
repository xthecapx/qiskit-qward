Metrics Base & Types
====================

Base Metric Calculator Class
-----------------------------

The `MetricCalculator` class is the abstract base class for all metric calculators in QWARD. It defines the unified interface that all metric implementations follow.

**Key Features:**
- **Unified Interface**: All metric calculators implement `get_metrics()` returning schema objects
- **Type Safety**: Schema-based validation with automatic error checking
- **Extensible**: Easy to create custom metric calculators

**Usage Pattern:**

.. code-block:: python

    from qward.metrics.base_metric import MetricCalculator
    from qward.metrics.types import MetricsType, MetricsId
    from pydantic import BaseModel
    
    class MyCustomMetricsSchema(BaseModel):
        custom_value: float
        circuit_signature: str
    
    class MyCustomMetric(MetricCalculator):
        def _get_metric_type(self) -> MetricsType:
            return MetricsType.PRE_RUNTIME
        
        def _get_metric_id(self) -> MetricsId:
            return MetricsId.QISKIT  # Or define new ID
        
        def is_ready(self) -> bool:
            return self.circuit is not None
        
        def get_metrics(self) -> MyCustomMetricsSchema:
            # Custom calculation logic
            return MyCustomMetricsSchema(
                custom_value=self.circuit.depth() * self.circuit.num_qubits,
                circuit_signature=f"{self.circuit.num_qubits}q_{self.circuit.depth()}d"
            )

API Reference
-------------

.. automodule:: qward.metrics.base_metric
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.metrics.base_metric.MetricCalculator
   :members:
   :undoc-members:
   :show-inheritance:

Metric Enums
------------

.. automodule:: qward.metrics.types
   :members:
   :undoc-members:

.. autoclass:: qward.metrics.types.MetricsId
   :members:
   :undoc-members:

.. autoclass:: qward.metrics.types.MetricsType
   :members:
   :undoc-members: 