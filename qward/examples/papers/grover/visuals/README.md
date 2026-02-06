# Grover QPU Visualization Guide

This folder contains graphs generated from **IBM QPU execution results** for Grover's algorithm. The data is loaded from `data/qpu/raw/*.json`. Use these plots to see **when results stop being useful** as circuit size or depth increases.

## How to generate the graphs

From the `grover` directory:

```bash
python grover_qpu_visuals.py
```

Optional: specify custom paths

```bash
python grover_qpu_visuals.py --data-dir data/qpu/raw --out-dir visuals
```

---

## Graph descriptions

### 1. `grover_success_rate_vs_qubits.png`

**What it shows:** Mean success rate (with error bars) vs circuit size, one point per config (labeled by num_qubits and config_id).

**Reference lines:**
- **Theoretical (no noise):** Ideal success probability from Grover’s formula.
- **Random guess:** Classical random search probability (1/2^n for one marked state).
- **30% and 50% thresholds:** Horizontal lines at 0.30 and 0.50.

**How to read it:** As qubit count increases, QPU success usually drops toward (or below) random chance. When the QPU curve approaches or crosses the random-guess line, **results are no longer useful** (no quantum advantage).

---

### 2. `grover_success_rate_vs_transpiled_depth.png`

**What it shows:** Mean success rate vs **transpiled circuit depth** (max over optimization levels). Each point is labeled with its config_id.

**How to read it:** Deeper circuits accumulate more noise. You see **where useful results stop**: when success rate falls and stays below ~30% or near random, that depth/size regime is not useful on this hardware.

---

### 3. `grover_advantage_ratio_vs_qubits.png`

**What it shows:** **Quantum advantage ratio** (success rate ÷ random chance) per config. Green bars: ratio > 2 (quantum advantage). Red bars: ratio ≤ 2.

**Reference line:** 2× random (horizontal line at 2.0). Above this, the algorithm is doing better than 2× random search.

**How to read it:** Configs with red bars are **not giving useful advantage**; green bars indicate configs where the QPU still shows meaningful advantage.

---

### 4. `grover_threshold_pass_by_config.png`

**What it shows:** For each config, whether the **best run** (across optimization levels) passes the 30%, 50%, 70%, or 90% success-rate thresholds. Bar = 1 (Pass) or 0 (Fail).

**How to read it:** Lets you quickly see which configs still meet each quality bar. When all thresholds fail (all zeros), that config is **no longer giving useful results**.

---

### 5. `grover_success_by_optimization_level.png`

**What it shows:** Success rate per config, broken down by **transpiler optimization level** (0, 1, 2, 3). Grouped bars per config.

**How to read it:** Compare optimization levels: often level 2 or 3 gives better success by reducing gate count/depth. If every level is low for a config, that circuit is too large/noisy for useful results.

---

## Summary: when do results stop being useful?

- **Success rate** near or below **random chance** → no advantage.
- **Advantage ratio** &lt; 2 → not useful.
- **All threshold bars (30/50/70/90%)** fail → config not useful.
- **Success rate vs transpiled depth** drops and flattens low → that depth range is beyond useful QPU performance.

Use these plots together to decide which Grover configs and sizes are worth running on the QPU and where to stop.
