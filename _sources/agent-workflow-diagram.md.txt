# Quantum Development Agent Workflow

This diagram shows how the multi-agent system collaborates across the 7 workflow phases defined in `workflows/collaborative-development.md`.

## Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM DEVELOPMENT WORKFLOW                                  │
│                                                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────────┐  │
│   │                        quantum-research-lead                                   │  │
│   │                           (ORCHESTRATOR)                                       │  │
│   │    • Coordinates all phases  • Enforces TDD  • Makes iteration decisions      │  │
│   └───────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┘  │
│           │                 │                 │                 │                     │
│           ▼                 ▼                 ▼                 ▼                     │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐           │
│   │Phase 1:       │ │Phase 4:       │ │Phase 6:       │ │Phase 7:       │           │
│   │IDEATION       │ │IMPLEMENTATION │ │REVIEW         │ │DOCUMENTATION  │           │
│   │               │ │               │ │               │ │               │           │
│   │Lead+Researcher│ │  Architect    │ │  All Agents   │ │Tech. Writer   │           │
│   └───────┬───────┘ └───────▲───────┘ └───────▲───────┘ └───────────────┘           │
│           │                 │                 │                                      │
│           ▼                 │                 │                                      │
│   ┌───────────────┐         │         ┌───────────────┐                             │
│   │Phase 2:       │         │         │Phase 5:       │                             │
│   │THEORY         │         │         │ANALYSIS       │                             │
│   │               │         │         │               │                             │
│   │  Researcher   │         │         │Data Scientist │─────────────────────────────┘
│   └───────┬───────┘         │         └───────▲───────┘
│           │                 │                 │
│           ▼                 │                 │
│   ┌───────────────┐         │                 │
│   │Phase 3:       │─────────┘                 │
│   │TEST DESIGN    │                           │
│   │    (TDD)      │                           │
│   │               │                           │
│   │Test Engineer  │                           │
│   └───────────────┘                           │
│                                               │
└───────────────────────────────────────────────┴───────────────────────────────────────
```

## Detailed Agent Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│                              TDD-ENFORCED DEVELOPMENT FLOW                              │
│                                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │  research-  │     │  quantum-   │     │    test-    │     │   python-   │          │
│   │    lead     │◄───►│  researcher │────►│  engineer   │────►│  architect  │          │
│   │             │     │             │     │             │     │             │          │
│   │ Orchestrates│     │ Designs     │     │ Writes tests│     │ Implements  │          │
│   │ Coordinates │     │ Hamiltonians│     │ BEFORE code │     │ to pass     │          │
│   │ Decides     │     │ Proofs      │     │ (TDD)       │     │ all tests   │          │
│   └──────┬──────┘     └─────────────┘     └─────────────┘     └──────┬──────┘          │
│          │                                                          │                  │
│          │            ┌─────────────┐     ┌─────────────┐            │                  │
│          │            │  technical- │     │  quantum-   │            │                  │
│          └───────────►│   writer    │◄────┤data-scientist◄───────────┘                  │
│                       │             │     │             │                               │
│           Final       │ LaTeX docs  │     │ Analyzes    │    Tested                     │
│           handoff     │ Papers      │     │ Visualizes  │    implementation             │
│                       │ Reports     │     │ Benchmarks  │                               │
│                       └─────────────┘     └─────────────┘                               │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Communication Model

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                         │
│                         DIRECT CLARIFICATION + LEAD OVERSIGHT                           │
│                                                                                         │
│                              ┌─────────────────────┐                                    │
│                              │   research-lead     │                                    │
│                              │                     │                                    │
│                              │  • Phase transitions│                                    │
│                              │  • Decisions        │                                    │
│                              │  • Unblocking       │                                    │
│                              │  • NOT a router     │                                    │
│                              └──────────┬──────────┘                                    │
│                                         │                                               │
│                    escalate blockers &  │  oversight                                    │
│                    decisions only       │  (observes, doesn't bottleneck)               │
│                                         │                                               │
│  ┌──────────────────────────────────────┼───────────────────────────────────────────┐  │
│  │                                      │                                           │  │
│  │  ┌─────────────┐    ┌─────────────┐  │  ┌─────────────┐    ┌─────────────┐      │  │
│  │  │  researcher │◄──►│ test-engineer│◄─┼─►│  architect  │◄──►│data-scientist│      │  │
│  │  └──────┬──────┘    └─────────────┘  │  └──────┬──────┘    └──────┬──────┘      │  │
│  │         │                            │         │                  │             │  │
│  │         │         DIRECT             │         │      DIRECT      │             │  │
│  │         │      CLARIFICATION         │         │   CLARIFICATION  │             │  │
│  │         │                            │         │                  │             │  │
│  │         │    ┌─────────────────┐     │         │                  │             │  │
│  │         └───►│technical-writer │◄────┼─────────┴──────────────────┘             │  │
│  │              │                 │     │                                          │  │
│  │              │ Can ask ANY     │     │                                          │  │
│  │              │ agent directly  │     │                                          │  │
│  │              └─────────────────┘     │                                          │  │
│  │                                      │                                          │  │
│  │            AGENTS CAN TALK DIRECTLY FOR CLARIFICATIONS                          │  │
│  └──────────────────────────────────────┴──────────────────────────────────────────┘  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Communication Rules

| Type | Channel | Lead Involved? |
|------|---------|----------------|
| **Phase Handoffs** | Sequential flow | Yes (approves) |
| **Clarifications** | Direct agent-to-agent | No |
| **Decisions** | Through Lead | Yes (decides) |
| **Blockers** | Escalate to Lead | Yes (unblocks) |
| **Scope Changes** | Through Lead | Yes (approves) |

### Examples

```
✅ DIRECT (no Lead):
   Architect → Researcher: "Can you clarify the Hamiltonian encoding?"
   Tech Writer → Data Scientist: "Can you send me the convergence plot data?"
   Test Engineer → Researcher: "What edge cases should I test for?"

⬆️ ESCALATE TO LEAD:
   Architect: "Hardware limits make this design infeasible, need new approach"
   Data Scientist: "Results show algorithm doesn't converge, should we pivot?"
   Test Engineer: "Requirements conflict with what's testable"
```

## TDD Workflow Detail

```
                                 TEST-DRIVEN DEVELOPMENT

    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │   RESEARCHER    │          │  TEST ENGINEER  │          │    ARCHITECT    │
    │                 │          │                 │          │                 │
    │  Theoretical    │   ───►   │  Writes tests   │   ───►   │  Implements     │
    │  specification  │          │  that FAIL      │          │  until tests    │
    │                 │          │  initially      │          │  PASS           │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
           │                            │                            │
           │                            │                            │
           ▼                            ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │ • Circuit design│          │ • pytest tests  │          │ • Qiskit code   │
    │ • Hamiltonian   │          │ • Fixtures      │          │ • Classes       │
    │ • Success math  │          │ • Edge cases    │          │ • All tests ✓   │
    │ • Complexity    │          │ • Assertions    │          │ • Documentation │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
```

## Handoff Artifacts

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│Researcher → Test Eng.│  │Test Eng. → Architect │  │Architect → Data Sci. │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ • Circuit diagram    │  │ • pytest test files  │  │ • Tested code        │
│ • Hamiltonian        │  │ • Fixtures           │  │ • All tests passing  │
│ • Success criteria   │  │ • Expected behaviors │  │ • Execution ready    │
│ • Complexity bounds  │  │ • Edge cases         │  │ • QWARD integration  │
│ • Edge cases to test │  │ • Failing tests (red)│  │ • Test results       │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐
│Data Sci. → Lead      │  │Lead → Tech. Writer   │
├──────────────────────┤  ├──────────────────────┤
│ • Results + errors   │  │ • Approved findings  │
│ • Visualizations     │  │ • All agent outputs  │
│ • Noise analysis     │  │ • Key conclusions    │
│ • Baseline compare   │  │ • Figures to include │
└──────────────────────┘  └──────────────────────┘
```

## Iteration Loops

```
                            FAILURE DIAGNOSIS

        Tests failing or low fidelity detected?
                        │
                        ▼
              ┌─────────────────────┐
              │   Lead evaluates    │
              │   root cause        │
              └─────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Theoretical?    Test Design?    Implementation?
        │               │               │
        ▼               ▼               ▼
   ┌────────┐      ┌────────┐      ┌────────┐
   │Researcher│      │Test Eng│      │Architect│
   │ revises │      │ updates│      │ fixes   │
   │ theory  │      │ tests  │      │ code    │
   └────────┘      └────────┘      └────────┘
```

## Phase Summary

| Phase | Agent(s) | Purpose | Output |
|-------|----------|---------|--------|
| 1. Ideation | Lead + Researcher | Define problem, assess quantum advantage | Problem spec, feasibility |
| 2. Theory | Researcher | Circuit design, Hamiltonian formulation | Algorithm design |
| 3. Test Design | Test Engineer | Write tests from specs (TDD - red phase) | Failing pytest tests |
| 4. Implementation | Architect | Qiskit code to pass all tests (TDD - green) | Working, tested code |
| 5. Analysis | Data Scientist | Run experiments, analyze results | Metrics, visualizations |
| 6. Review | All | Evaluate success, iterate or approve | Go/no-go decision |
| 7. Documentation | Technical Writer | LaTeX papers, final reports | Publication-ready docs |

## Agent Summary

| Agent | Model | Role | Color |
|-------|-------|------|-------|
| `quantum-research-lead` | opus | Orchestrator | red |
| `quantum-computing-researcher` | opus | Theorist | blue |
| `test-engineer` | sonnet | TDD Specialist | green |
| `python-architect` | opus | Developer | green |
| `quantum-data-scientist` | sonnet | Analyst | yellow |
| `technical-writer` | sonnet | Documenter | magenta |

## Agent Files

- `agents/quantum-research-lead.md` - Orchestrator with full workflow knowledge
- `agents/quantum-computing-researcher.md` - Theorist with handoff to Test Engineer
- `agents/test-engineer.md` - TDD specialist, writes tests before implementation
- `agents/python-architect.md` - Developer, implements to pass tests
- `agents/quantum-data-scientist.md` - Analyst with QWARD tools
- `agents/technical-writer.md` - LaTeX documentation specialist

## Related Files

- `workflows/collaborative-development.md` - Full workflow specification
- `CLAUDE.md` - Project-level agent coordination instructions
