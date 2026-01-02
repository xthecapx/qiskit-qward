===========================
Behavioral Metrics (Execution Dynamics)
===========================

.. automodule:: qward.metrics.behavioral_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
The *BehavioralMetrics* module provides execution-oriented metrics derived from
the static structure of a quantum circuit. Unlike structural or element metrics,
which characterize complexity and gate composition, behavioral metrics aim to
capture **how the circuit behaves as a computational process**.

These metrics approximate dynamic execution features without running the circuit,
following methods from Lubinski *et al.* (2023) and Tomesh *et al.* (2022).
The following behavioral dimensions are computed:

1. **Normalized Depth**  
   Depth of the circuit after transpilation to a canonical gate set
   {``rx``, ``ry``, ``rz``, ``cx``}, normalized through basis unification.

2. **Program Communication**  
   Communication intensity derived from the *interaction graph*, reflecting how
   frequently qubits must exchange information (two-qubit operations).

3. **Critical-Depth**  
   Ratio of two-qubit operations on the DAG’s critical path to the total number of 
   two-qubit operations, measuring how strongly entangling gates define execution time.

4. **Measurement**  
   Fraction of circuit layers involving measurement or reset operations, with emphasis
   on *mid-circuit measurement*.

5. **Liveness**  
   Ratio of active qubit-time slots across the circuit, estimating concurrency and
   qubit usage over time.

6. **Parallelism (Cross-Talk Susceptibility)**  
   A normalized metric derived from the ratio between gate count, depth, and qubit
   count, approximating susceptibility to cross-talk and parallel-execution density.

These metrics complement QWARD’s structural and element-level metrics by capturing
temporal, interactive, and execution-critical behavior.

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   BehavioralMetrics

Selected Formulas
-----------------
**Normalized Depth**  
Depth after transpilation to basis ``{rx, ry, rz, cx}``.

**Program Communication**  
Based on interaction graph ``G``:
``C = Σ d(qᵢ) / (N(N−1))``  
where ``d(qᵢ)`` is the degree of qubit ``i`` and ``N`` is the number of qubits.

**Critical-Depth**  
``D = ned / ne``  
where ``ned`` = two-qubit gates on the critical path,  
``ne`` = total two-qubit gates.

**Measurement Ratio**  
``M = l_mcm / d``  
where ``l_mcm`` = layers containing measurement/reset,  
``d`` = total layers (circuit depth).

**Liveness**  
``L = Σ Aᵢⱼ / (n · d)``  
where ``Aᵢⱼ = 1`` if qubit ``i`` is active in layer ``j``.

**Parallelism**  
``P = (ng/d − 1) / (n − 1)``  
where ``ng`` = number of gates excluding barrier/measure/reset.

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import BehavioralMetrics

   qc = QuantumCircuit(3)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   metrics = BehavioralMetrics(qc).get_metrics()

   print("Normalized Depth:", metrics.normalized_depth)
   print("Program Communication:", metrics.program_communication)
   print("Critical Depth:", metrics.critical_depth)
   print("Liveness:", metrics.liveness)

Notes
-----
- All behavioral metrics are *pre-runtime* and require only static analysis.  
- Normalized depth depends on successful transpilation to the canonical basis.  
- Program communication and critical-depth are graph-theoretic and depend on the
  circuit’s DAG (Directed Acyclic Graph) representation.  
- Liveness ignores barrier operations, treating them as non-informative for execution
  activity.  
- Parallelism is clamped to the ``[0, 1]`` interval.

References
----------
- [1] T. Lubinski et al., "Application-Oriented Performance Benchmarks for 
Quantum Computing," in IEEE Transactions on Quantum Engineering, vol. 4, 
pp. 1-32, 2023, Art no. 3100332, doi: 10.1109/TQE.2023.3253761.

- [2] T. Tomesh, P. Gokhale, V. Omole, G. S. Ravi, K. N. Smith, J. Viszlai, 
X.-C. Wu, N. Hardavellas, M. R. Martonosi y F. T. Chong, “SupermarQ: A scalable 
quantum benchmark suite,” in Proc. 2022 IEEE International Symposium on 
High-Performance Computer Architecture (HPCA), 2022, doi: 
10.1109/HPCA53966.2022.00050.