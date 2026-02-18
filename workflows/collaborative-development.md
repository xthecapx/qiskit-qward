# Collaborative Development Workflow

This document defines how the QWARD agent team collaborates on quantum computing tasks.

## Team Roles

| Agent | Role | Primary Focus |
|-------|------|---------------|
| **quantum-research-lead** | Research Lead | Strategy, coordination, risk assessment |
| **quantum-computing-researcher** | Researcher | Algorithm design, Hamiltonians, proofs |
| **python-architect** | Python Dev | Code architecture, OOP, library compliance |
| **quantum-data-scientist** | Data Scientist | Analysis, visualization, benchmarking |

> Note: The **python-architect** also serves as **Quantum Dev** for Qiskit circuit implementation.

---

## Workflow Phases

### Phase 1: Ideation (Lead + Researcher)

**Participants**: quantum-research-lead, quantum-computing-researcher

**Process**:
1. **Lead** identifies the problem domain (e.g., "Optimize a logistics network")
2. **Lead** assesses feasibility and quantum advantage potential
3. **Researcher** determines suitable quantum algorithm (QAOA, VQE, Grover's, etc.)
4. **Researcher** provides initial complexity analysis and resource estimates

**Outputs**:
- Problem statement with success criteria
- Recommended quantum approach with justification
- Preliminary resource requirements (qubits, depth, shots)

**Handoff**: Approved problem statement → Phase 2

---

### Phase 2: Theoretical Design (Researcher)

**Participants**: quantum-computing-researcher

**Process**:
1. Draft circuit structure and gate sequence
2. Formulate cost Hamiltonian or success function
3. Define mathematical requirements and constraints
4. Specify measurement strategy and post-processing
5. Prove correctness and analyze complexity

**Outputs**:
- Circuit diagram/pseudocode
- Hamiltonian in Pauli string notation: H = Σᵢ cᵢ Pᵢ
- Success criteria with mathematical definition
- Complexity analysis (gate count, depth, qubit requirements)

**Handoff**: Theoretical specification → Phase 3

---

### Phase 3: Implementation (Python Dev + Quantum Dev)

**Participants**: python-architect (dual role)

**Quantum Development Tasks**:
1. Implement circuit in Qiskit using current APIs
2. Test against Aer noise models (IBM Heron, Rigetti Ankaa presets)
3. Optimize transpilation (use optimization_level=3 for production)
4. Validate circuit matches theoretical specification

**Python Development Tasks**:
1. Design class structure following OOP best practices
2. Implement classical optimization loop (SciPy, custom optimizers)
3. Ensure library compliance (see Project Standards below)
4. Write unit tests matching class hierarchy

**Outputs**:
- Working Qiskit implementation
- Clean class architecture with proper abstractions
- Test suite with noise model validation
- Documentation with usage examples

**Handoff**: Tested implementation → Phase 4

---

### Phase 4: Execution & Analysis (Data Scientist)

**Participants**: quantum-data-scientist

**Process**:
1. Execute on simulator or real backend via QWARD executor
2. Analyze measurement count histograms
3. Calculate statistical significance and confidence intervals
4. Visualize convergence (for variational algorithms)
5. Characterize noise impact on results
6. Benchmark against classical baseline

**Outputs**:
- Results summary with uncertainty quantification
- Visualizations (histograms, Bloch spheres, loss landscapes)
- Noise characterization report
- Performance comparison vs. classical

**Handoff**: Analysis report → Phase 5

---

### Phase 5: Review (All Agents)

**Participants**: All team members

**Process**:
1. **Data Scientist** presents results and analysis
2. **All** evaluate if success criteria are met
3. If fidelity is too low:
   - **Researcher** analyzes noise impact on algorithm
   - **Python Dev** optimizes gate depth or adjusts parameters
   - Return to Phase 3 or 4 as needed
4. **Lead** synthesizes findings and documents conclusions

**Decision Points**:
- Success criteria met → Document and close
- Fidelity too low → Iterate (optimize circuit or adjust noise mitigation)
- Fundamental limitation → Lead assesses alternative approaches

**Outputs**:
- Final report with conclusions
- Lessons learned for agent memory
- Updated documentation

---

## Project Standards

All developer agents must follow these project standards:

### Code Style (`.pylintrc`)
- Max line length: 100 characters
- Max arguments: 15, Max locals: 25
- Ignored directories: `CVS`, `examples`
- See full config: [.pylintrc](../.pylintrc)

### Available Libraries (`requirements.qward.txt`)
```
qiskit==2.1.2
qiskit-aer==0.17.1
qiskit-ibm-runtime==0.41.1
qbraid[braket]==0.11.0
qiskit-braket-provider>=0.11.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
statsmodels
torch==2.8.0
```

### QWARD Executor Patterns
Use `qward.algorithms.QuantumCircuitExecutor` for:
- `simulate()` - Local simulation with optional noise
- `run_ibm()` - IBM Quantum hardware (batch mode)
- `run_qbraid()` - Rigetti via qBraid

### Noise Model Presets
```python
from qward.algorithms import get_preset_noise_config
# Available: "IBM-HERON-R1", "IBM-HERON-R2", "IBM-HERON-R3", "RIGETTI-ANKAA3"
```

---

## Communication Protocol

### Agent Memory
Each agent maintains persistent memory in `.claude/agent-memory/<agent-name>/`:
- Record discoveries, patterns, and decisions
- Share findings relevant to other team members
- Document lessons learned after each workflow completion

### Handoff Format
When passing work between phases, include:
1. **Status**: What was accomplished
2. **Artifacts**: Files created/modified
3. **Next Steps**: What the receiving agent should do
4. **Blockers**: Any issues requiring attention

---

## Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│                    COLLABORATIVE WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: IDEATION                                              │
│  ┌──────────────┐    ┌────────────────┐                         │
│  │ Research Lead │───▶│   Researcher   │                         │
│  │ (problem)     │    │ (algorithm)    │                         │
│  └──────────────┘    └───────┬────────┘                         │
│                              │                                   │
│                              ▼                                   │
│  Phase 2: THEORETICAL DESIGN                                    │
│  ┌────────────────────────────────────┐                         │
│  │         Researcher                  │                         │
│  │ (circuit, Hamiltonian, proofs)     │                         │
│  └───────────────┬────────────────────┘                         │
│                  │                                               │
│                  ▼                                               │
│  Phase 3: IMPLEMENTATION                                        │
│  ┌────────────────────────────────────┐                         │
│  │      Python Architect               │                         │
│  │ (Qiskit circuit + class structure) │                         │
│  └───────────────┬────────────────────┘                         │
│                  │                                               │
│                  ▼                                               │
│  Phase 4: EXECUTION & ANALYSIS                                  │
│  ┌────────────────────────────────────┐                         │
│  │       Data Scientist                │                         │
│  │ (run, analyze, visualize)          │                         │
│  └───────────────┬────────────────────┘                         │
│                  │                                               │
│                  ▼                                               │
│  Phase 5: REVIEW                                                │
│  ┌────────────────────────────────────┐                         │
│  │          All Agents                 │                         │
│  │ (evaluate, iterate if needed)      │                         │
│  └────────────────────────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
