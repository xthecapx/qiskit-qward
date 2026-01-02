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
Quantum-specific metrics differ fundamentally from structural or gate-level
metrics. Instead of counting operations or analyzing connectivity, they evaluate
properties rooted in the circuit’s action on quantum states or operators.
These properties correlate strongly with classical simulability, potential
quantum advantage, and the physical difficulty of implementing or maintaining
a computation.

Practical Approximation Strategy
--------------------------------
The original definitions of many of these metrics (especially *magic*,
*sensitivity*, and *coherence*) require solving optimization problems over
exponentially large operator spaces. In their exact form, these computations
are intractable for all but the smallest circuits due to:

* the exponential dimension of the operator Hilbert space,
* the need to optimize over arbitrary density matrices or observables,
* the requirement to evaluate full Pauli expansions with \( O(4^n) \) terms.

Because of this, **QWARD implements practical, efficient approximations**:

1. **Gradient-ascent optimization**  
   Instead of optimizing over all possible states or operators, parameters are
   restricted (e.g., diagonal density matrices, low-weight Pauli operators),
   and PyTorch is used to perform differentiable optimization.

2. **Restricted Pauli basis**  
   Instead of expanding operators over all Pauli strings up to weight \( n \),
   only low-weight Paulis (weight 1–2) are considered, following the empirical
   observation that these capture most relevant interactions in typical circuits.

3. **Proxy quantities**  
   For example, the magic metric uses off-diagonal imaginary components as a
   proxy for non-stabilizerness; coherence uses the \( L^1 \)-norm of
   off-diagonal elements; and sensitivity considers the influence difference
   before and after conjugation under the circuit unitary.

4. **Measurement removal**  
   Since measurement breaks unitarity and invalidates many theoretical
   quantities, all metrics are computed on the *unitary portion* of the
   circuit, obtained by stripping out measurement, barrier, and reset nodes.

These approximations preserve **interpretability, scalability, and relative
comparability** across circuits, while avoiding exponential computational cost.


Available Metrics
-----------------
The class computes the following quantum-specific metrics:

* ``spposq_ratio`` — Percentage of qubits whose first nontrivial operation is a
  Hadamard gate, measuring early creation of superposition.

* ``magic`` — Proxy for circuit non-stabilizerness, approximated via gradient
  ascent over diagonal density matrices.

* ``coherence`` — Approximated \( L^1 \)-coherence of output states under the
  circuit, optimized over diagonal input states.

* ``sensitivity`` — Approximate circuit sensitivity (CiS), computed as the
  change in influence of a restricted Pauli operator under conjugation.

* ``entanglement_ratio`` — Ratio of two-qubit interactions to total
  non-measurement gates.


Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import QuantumSpecificMetrics

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   qsm = QuantumSpecificMetrics(qc)
   metrics = qsm.get_metrics()

   print(metrics.magic, metrics.coherence, metrics.entanglement_ratio)


References
----------
[1] K. Bu, R. J. Garcia, A. Jaffe, D. E. Koh, and L. Li,
       *"Complexity of quantum circuits via sensitivity, magic, and coherence,"*
       Communications in Mathematical Physics, vol. 405, no. 7, 2024.
       doi:10.1007/s00220-024-05030-6.

[2] T. Tomesh et al., *"SupermarQ: A scalable quantum benchmark suite,"*
       IEEE HPCA 2022. doi:10.1109/HPCA53966.2022.00050.

[3] J. A. Cruz-Lemus et al.,
       *"Towards a set of metrics for quantum circuits understandability,"*
       QUATIC 2021, CCIS vol. 1439, Springer, 2021.
       doi:10.1007/978-3-030-85347-1_18.