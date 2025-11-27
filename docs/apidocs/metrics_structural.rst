=============================
Structural Metrics
=============================

.. automodule:: qward.metrics.structural_metrics
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__

Overview
--------
The Structural Metrics module provides a unified analysis of the structural
properties of a quantum circuit. It integrates three complementary metric
families:

1. **Quantum Lines-of-Code (ϕ1–ϕ6)** as proposed in quantum software engineering  
   (e.g., total LOC, quantum-only LOC, measurement LOC, number of qubits, and
   gate-type diversity).

2. **Halstead Metrics adapted to quantum circuits**, where quantum gates act as
   *operators* and qubits/classical bits/parameters act as *operands*. These
   metrics estimate program vocabulary, program length, volume, difficulty, and
   effort—offering an early indication of cognitive complexity before execution.

3. **Circuit Shape Metrics**, computed from the circuit DAG, which quantify:
   - **Width** (number of qubits),
   - **Depth** (longest sequential chain of operations),
   - **Size** (total operations),
   - **Maximum and average density** (number of operations per DAG layer).

Together, these metrics capture the structural, topological, and cognitive
complexity of a quantum program, enabling static analysis, comparison of circuit
architectures, and complexity-aware optimizations.

Autosummary
-----------
.. autosummary::
   :toctree: generated/

   StructuralMetrics

Selected Formulas
-----------------
The following equations summarize key computed metrics:

**Halstead Vocabulary**
``n = n₁ + n₂``  
where ``n₁`` = number of unique operators,  
and ``n₂`` = number of unique operands.

**Halstead Program Length**
``N = N₁ + N₂``  
where ``N₁`` = total operators,  
and ``N₂`` = total operands.

**Halstead Estimated Length**
``N̂ = n₁ * log₂(n₁) + n₂ * log₂(n₂)``

**Halstead Volume**
``V = N * log₂(n)``

**Halstead Difficulty**
``D = (n₁ / 2) * (N₂ / n₂)``

**Halstead Effort**
``E = D * V``

**Circuit Density**
- ``max_dens`` = maximum number of operations in any DAG layer  
- ``avg_dens`` = average number of operations across all layers

Usage Example
-------------
.. code-block:: python

   from qiskit import QuantumCircuit
   from qward.metrics import StructuralMetrics

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   metrics = StructuralMetrics(qc).get_metrics()

   print("Depth:", metrics.depth)
   print("Halstead Volume:", metrics.volume)
   print("Estimated Effort:", metrics.effort)
   print("Average Density:", metrics.avg_dens)

Notes
-----
- Structural metrics are **pre-runtime**; they do not require circuit execution.
- Measurements and classical operations are included as operands but are treated
  differently from quantum gates in LOC and Halstead categories.
- Density metrics depend on the DAG structure; circuits with higher parallelism
  exhibit lower depth but potentially higher per-layer operation counts.

References
----------
- [1]J. Zhao, “Some Size and Structure Metrics for Quantum Software.” 2021. 
[Online]. Available: https://arxiv.org/abs/2103.08815