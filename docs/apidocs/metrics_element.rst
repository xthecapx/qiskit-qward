=============================
Element Metrics (Circuit Quality)
=============================

.. automodule:: qward.metrics.element_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
Element metrics cuantifican la distribución y control de compuertas, uso de oráculos y mediciones para evaluar comprensibilidad y estructura lógica de un circuito.

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   ElementMetrics

Key Ratios
----------
- ``percent_single_gates``: Proporción de compuertas de un solo qubit.
- ``percent_q_in_cnot`` / ``percent_q_in_toff``: Penetración estructural de interacciones controladas.
- ``percent_q_in_or``: Cobertura de oráculos sobre qubits.

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import ElementMetrics

   qc = QuantumCircuit(3)
   qc.h(0); qc.cx(0,1); qc.ccx(0,1,2); qc.measure_all()

   metrics = ElementMetrics(qc).get_metrics()
   print(metrics.no_gates, metrics.percent_single_gates)
