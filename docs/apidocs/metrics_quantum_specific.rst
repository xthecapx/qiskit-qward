=============================
Quantum Specific Metrics (Magic / Coherence / Sensitivity)
=============================

.. automodule:: qward.metrics.quantum_specific_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
Métricas que enfocan propiedades intrínsecas al potencial de ventaja cuántica: superposición inicial (%SpposQ), magia (no-Cliffordness), coherencia generada, sensibilidad estructural y entanglement-ratio.

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   QuantumSpecificMetrics

Implementation Notes
--------------------
- Algunas métricas (magic, coherence, sensitivity) requieren ``torch``; si no está instalado retornan 0 con aviso.
- ``entanglement_ratio`` = interacciones de 2 qubits / total de operaciones.

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import QuantumSpecificMetrics

   qc = QuantumCircuit(2)
   qc.h(0); qc.cx(0,1); qc.measure_all()
   qm = QuantumSpecificMetrics(qc).get_metrics()
   print(qm.spposq_ratio, qm.entanglement_ratio)
