# Quantum Research Lead - Agent Memory

## Project: QML Data Encoding Research
- Location: `qward/examples/papers/qml-data-encoding/`
- Branch: `team-qml`
- Plan: `plan.md` (7 phases, multi-agent workflow)

## Phase 1 Decisions (2026-02-19)
- Selected 8 datasets: Iris, Wine, Cancer, MNIST (benchmark) + Credit Fraud, NSL-KDD, HAR, Heart Disease (real-world)
- 5 encoding methods: Basis, Amplitude, Angle, IQP, Data Re-uploading
- 4 preprocessing levels: None, MinMax, Z-score, PCA+MinMax
- Fixed controls: VQC with RealAmplitudes reps=2, COBYLA maxiter=200, 1024 shots
- Primary qubit targets: 4 and 8 qubits (after PCA when needed)
- IQP is most NISQ-constrained (O(n^2) CX gates); limit to 4-6 features on hardware noise
- Angle encoding is most NISQ-friendly (no 2Q gates)
- Simulation-first approach; noise study on top configurations only

## Key Literature Findings
- Encoding choice > model architecture for QML performance (Cerezo 2022)
- Exponential concentration kills generic quantum kernels (Thanasilp 2024)
- Expressibility-trainability trade-off constrains encoding design (Holmes 2022)
- No existing data-to-encoding mapping framework (our primary gap)

## Phase 6 Synthesis (2026-02-19)
- **Sparsity-encoding correlation**: rho=0.90, p=0.002 (strongest quantitative result)
- **PCA neutralizes IQP**: Angle >= IQP in 6/8 datasets after PCA (validates Phase 2 theory)
- **Classical outperforms quantum**: t=-15.07, p<0.001 at 4-qubit scale (expected)
- **Encoding gap**: 18.7% on Iris, W=0.556 (large effect, underpowered Friedman test)
- **Decision**: No iteration needed; proceed to Phase 7
- **Key limitation**: 58% exclusion rate (67/160 valid configs), low statistical power

## Hypothesis Verdicts
- H1 (data structure -> encoding): NOT SUPPORTED (power issue, not effect issue)
- H2 (preprocessing reduces resources): PARTIALLY SUPPORTED
- H3 (domain-specific advantages): SUPPORTED (strongest hypothesis)
- H4 (standard preprocessing inadequate): PARTIALLY SUPPORTED

## Technical Notes
- WebSearch tool was unavailable during Phase 1; relied on training knowledge
- Token budget tracking required in `reports/token_budget.md`
- Use `uv run` not `python` per CLAUDE.md
- QWARD noise presets: IBM-HERON-R1/R2/R3, RIGETTI-ANKAA3
- Qiskit 2.1 requires `.decompose()` after parameter binding for Aer compatibility
- Total project cost through Phase 6: ~$33.01
