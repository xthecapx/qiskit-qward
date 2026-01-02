===========================
Element Metrics (Gate-Level)
===========================

.. automodule:: qward.metrics.element_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
The *ElementMetrics* component provides low-level, gate– and qubit–oriented
metrics that describe the fundamental building blocks of a quantum circuit.
These metrics quantify:

- Distribution of Pauli, single-qubit, controlled, and multi-qubit gates
- Oracle usage patterns (simple and controlled)
- Qubit activation and interaction patterns
- CNOT- and Toffoli-related qubit involvement
- Measurement and ancilla utilization

They represent a core metric family in QWARD, directly aligned with the
definition of elementary circuit properties proposed by Cruz-Lemus *et al.*
(QUATIC 2021).

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   ElementMetrics

Selected Formulas
-----------------
- **Total Pauli gates**: ``t_no_p = no_p_x + no_p_y + no_p_z``
- **Single-qubit gates**:
  ``t_no_sqg = no_h + t_no_p + no_other_sg + t_no_csqg``
- **Percentage of single-qubit gates**:
  ``percent_single_gates = t_no_sqg / no_gates`` (if ``no_gates > 0``)
- **Superposition ratio** (qubits whose first gate is ``H``):
  ``percent_sppos_q = (# qubits with initial H) / total_qubits``  
- **Oracle qubit ratio**:
  ``percent_q_in_or = (# qubits affected by oracles) / total_qubits``
- **CNOT/Toffoli target statistics**:  
  - ``avg_cnot`` = average number of times a qubit is target of a CNOT  
  - ``max_cnot`` = maximum among qubits  
  - (analogous definitions for Toffoli)

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import ElementMetrics

   qc = QuantumCircuit(3)
   qc.h(0); qc.cx(0, 1); qc.ccx(0, 1, 2)
   qc.measure_all()

   em = ElementMetrics(qc).get_metrics()
   print(em.no_p_x, em.no_cnot, em.no_toff, em.percent_q_in_or)

References
----------
- J. A. Cruz-Lemus, L. A. Marcelo, and M. Piattini, "Towards a set of metrics for 
quantum circuits understandability," in *Quality of Information and Communications 
Technology. QUATIC 2021 (Communications in Computer and Information Science, vol. 1439)
*, A. C. R. Paiva, A. R. Cavalli, P. Ventura Martins, and R. Pérez-Castillo, Eds. Cham:
 Springer, 2021, pp. 238–253. doi: 10.1007/978-3-030-85347-1_18.