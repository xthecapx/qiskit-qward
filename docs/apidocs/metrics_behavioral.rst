=============================
Behavioral Metrics (Execution Pattern)
=============================

.. automodule:: qward.metrics.behavioral_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
Caracterizan patrones de ejecución potencial (pre-runtime): profundidad normalizada, comunicación, camino crítico de interacciones de dos qubits, medición mid‑circuit, liveness y paralelismo.

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   BehavioralMetrics

Metric Notes
------------
- ``normalized_depth``: profundidad tras transpilar a base canónica ``['rx','ry','rz','cx']``.
- ``program_communication``: grado promedio normalizado del grafo de interacción.
- ``critical_depth``: fracción de operaciones de 2 qubits en el camino crítico.
- ``liveness``: actividad relativa qubit-tiempo.
- ``parallelism``: (ng/d - 1)/(n - 1) acotado a [0,1].

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import BehavioralMetrics

   qc = QuantumCircuit(3)
   qc.cx(0,1); qc.h(2); qc.cx(1,2); qc.measure_all()
   bm = BehavioralMetrics(qc).get_metrics()
   print(bm.normalized_depth, bm.parallelism)
