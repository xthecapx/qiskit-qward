=============================
Structural Metrics (LOC / Halstead / Shape)
=============================

.. automodule:: qward.metrics.structural_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
Reúne métricas de líneas de código cuánticas (ϕ1–ϕ6), Halstead adaptado y métricas estructurales (ancho, profundidad, densidad, tamaño) para caracterizar complejidad y esfuerzo potencial.

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   StructuralMetrics

Selected Formulas
-----------------
- Halstead Volume ``V = N * log2(n)`` donde ``N`` longitud del programa y ``n`` vocabulario.
- Dificultad ``D = (n1/2) * (N2/n2)`` adaptada a operadores/operandos cuánticos.
- Densidad promedio: promedio de operaciones por capa del DAG.

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import StructuralMetrics

   qc = QuantumCircuit(2)
   qc.h(0); qc.cx(0,1); qc.measure_all()
   sm = StructuralMetrics(qc).get_metrics()
   print(sm.depth, sm.volume, sm.effort)
