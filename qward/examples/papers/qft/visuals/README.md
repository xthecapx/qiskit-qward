# QFT QPU Visualization Guide

This folder contains graphs generated from QPU execution results for the Quantum Fourier Transform (QFT). By default, the visualizer auto-loads provider folders under `data/`, using `data/qpu/raw` (IBM) and `data/qpu/aws` (AWS). Use these plots to see **when results stop being useful** as circuit size or depth increases.

## How to generate the graphs

From the `qft` directory:

```bash
python qft_qpu_visuals.py
```

Optional: filter or override paths

```bash
python qft_qpu_visuals.py --provider aws
python qft_qpu_visuals.py --data-dir data/qpu/raw --out-dir visuals
```

- `--provider aws` reads `data/qpu/aws` (and legacy `data/aws/raw` if present).
- `--provider qpu` reads only `data/qpu/raw`.
- `--data-dir` keeps legacy behavior and overrides provider discovery.

---

## Graph descriptions

### 1. `qft_success_rate_vs_qubits.png`

**What it shows:** Mean success rate (with error bars) vs number of qubits, one point per config (labeled by num_qubits and config_id).

**Reference lines:**

- **Random guess (1/2^n):** Classical random probability for the correct output (roundtrip: one valid state; period detection: multiple peaks).
- **30% and 50% thresholds:** Horizontal lines at 0.30 and 0.50.

**How to read it:** As qubit count increases, QPU success typically drops. When the curve approaches the random-guess line, **results are no longer useful**.

---

### 2. `qft_success_rate_vs_transpiled_depth.png`

**What it shows:** Mean success rate vs **transpiled circuit depth** (max over optimization levels). Each point is labeled with its config_id.

**How to read it:** Deeper circuits see more noise. The plot shows **where useful results stop**: when success rate falls and stays low (~30% or near random), that depth regime is not useful on this hardware.

---

### 3. `qft_advantage_ratio_vs_qubits.png`

**What it shows:** **Quantum advantage ratio** (success rate ÷ random chance) per config. Green bars: ratio > 2 (quantum advantage). Red bars: ratio ≤ 2.

**Reference line:** 2× random (horizontal line at 2.0).

**How to read it:** Red bars indicate configs where the QPU is **not giving useful advantage**; green bars indicate configs that still show meaningful advantage.

---

### 4. `qft_threshold_pass_by_config.png`

**What it shows:** For each config, whether the **best run** (across optimization levels) passes the 30%, 50%, 70%, or 90% success-rate thresholds. Bar = 1 (Pass) or 0 (Fail).

**How to read it:** Quick view of which configs still meet each quality bar. When all thresholds fail, that config is **no longer giving useful results**.

---

### 5. `qft_success_by_optimization_level.png`

**What it shows:** Success rate per config, broken down by **transpiler optimization level** (0, 1, 2, 3). Grouped bars per config.

**How to read it:** Compare optimization levels. If every level is low for a config, that circuit size/depth is too large for useful QPU results.

---

## Summary: when do results stop being useful?

- **Success rate** near or below **random chance** → no advantage.
- **Advantage ratio** &lt; 2 → not useful.
- **All threshold bars (30/50/70/90%)** fail → config not useful.
- **Success rate vs transpiled depth** drops and stays low → that depth range is beyond useful QPU performance.

Use these plots together to decide which QFT configs and sizes are worth running on the QPU and where to stop.
