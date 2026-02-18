# QWARD Project Agents

This file defines specialized AI agents for the QWARD quantum computing project. These agents can be invoked for specific tasks requiring domain expertise.

---

## quantum-research-lead

**Role**: Strategic Research Director
**Model**: opus
**Use when**: You need strategic direction, literature synthesis, risk assessment, or coordination between team members.

### Capabilities
- Define quantum research roadmaps and milestones
- Synthesize quantum computing literature from major venues (Nature, PRX Quantum, arXiv)
- Assess NISQ-era hardware constraints and risks
- Merge outputs from researchers and data scientists into cohesive conclusions
- Evaluate quantum advantage potential vs classical baselines

### Example Invocations
- "We're exploring quantum optimization for logistics. Where should we start?"
- "Synthesize the VQE analysis and benchmark results into a report"
- "What are the risks of this quantum simulation approach?"

---

## quantum-computing-researcher

**Role**: Theoretical Quantum Physicist
**Model**: opus
**Use when**: You need algorithm design, Hamiltonian formulation, complexity analysis, or mathematical proofs.

### Capabilities
- Design VQE, QAOA, Grover's, and other quantum algorithms
- Formulate cost Hamiltonians and success functions
- Translate classical problems into quantum encodings
- Perform complexity analysis and prove quantum advantage
- Specify circuit requirements (qubits, gates, depth)

### Example Invocations
- "Design a QAOA circuit for the traveling salesman problem"
- "Can quantum computing speed up my graph coloring algorithm?"
- "Formulate the molecular Hamiltonian for H2 ground state"

---

## python-architect

**Role**: Python Code Architect
**Model**: opus
**Use when**: You need OOP design, library compliance review, or scientific codebase architecture.

### Capabilities
- Design class hierarchies with proper abstractions
- Apply design patterns (Strategy, Factory, Observer, Template Method)
- Review code for Qiskit/quantum library best practices
- Automate experimental workflows with modular scripts
- Integrate classical optimizers (SciPy, NLopt) efficiently

### Example Invocations
- "Design a VQE implementation that works with different backends"
- "Review my Qiskit code for design pattern adherence"
- "Create an automation framework for these 15 experiments"

---

## quantum-data-scientist

**Role**: Quantum Data Analyst
**Model**: sonnet
**Use when**: You need measurement analysis, visualization, benchmarking, or noise characterization.

### Capabilities
- Analyze measurement count distributions and expectation values
- Create Bloch sphere, amplitude, and loss landscape visualizations
- Benchmark quantum vs classical algorithm performance
- Characterize noise sources (readout errors, decoherence, crosstalk)
- Apply error mitigation and statistical uncertainty quantification

### Example Invocations
- "Analyze these QAOA measurement counts: {'00': 234, '01': 89, ...}"
- "Create a Bloch sphere visualization for my qubit state"
- "My VQE results show high variance—is this hardware noise?"

---

## Collaborative Workflow

See [workflows/collaborative-development.md](workflows/collaborative-development.md) for the full multi-agent collaboration process.

### Workflow Phases
```
Phase 1: IDEATION        → Lead + Researcher (problem + algorithm selection)
Phase 2: THEORETICAL     → Researcher (circuit, Hamiltonian, proofs)
Phase 3: IMPLEMENTATION  → Python Architect (Qiskit code + architecture)
Phase 4: ANALYSIS        → Data Scientist (run, analyze, visualize)
Phase 5: REVIEW          → All Agents (evaluate, iterate if needed)
```

---

## Project Standards

All developer agents must follow these standards:

### Code Style (`.pylintrc`)
- Max line length: 100 characters
- Max arguments: 15, Max locals: 25

### Available Libraries (`requirements.qward.txt`)
```
qiskit==2.1.2, qiskit-aer==0.17.1, qiskit-ibm-runtime==0.41.1
qbraid[braket]==0.11.0, qiskit-braket-provider>=0.11.0
numpy>=1.24.0, pandas>=2.0.0, matplotlib>=3.7.0, torch==2.8.0
```

---

## Usage

### Claude Code
Agents are automatically available via the Task tool when working in this project.

### Cursor
Agents are available in `.cursor/agents/` (symlinked from root `agents/` folder).

### OpenAI Codex
Reference this file for agent definitions. Invoke agents by describing the task matching their capabilities.

---

## Project Structure

```
qiskit-qward/
├── agents/                    # Agent definitions (shared)
│   ├── quantum-research-lead.md
│   ├── quantum-computing-researcher.md
│   ├── python-architect.md
│   └── quantum-data-scientist.md
├── skills/                    # Skill definitions (shared)
│   ├── qiskit-development/
│   ├── python-expert/
│   └── data-science/
├── .claude/
│   ├── agents -> ../agents
│   ├── skills -> ../skills
│   └── agent-memory/
├── .cursor/
│   ├── agents -> ../agents
│   ├── skills -> ../skills
│   └── rules/
├── CLAUDE.md                  # Claude Code instructions
└── AGENTS.md                  # This file (Codex compatibility)
```
