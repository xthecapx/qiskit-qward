# BV scaling simulator baseline (2-8 measured bits)

Only `manifest.json`, the noise models and `analysis/` are kept. The 864 raw
records (131 MB) were removed because they are reproducible: rerunning with the
same seed gives identical counts and predictions (verified 2026-10-29; only the
capture timestamp differs).

Regenerate into a fresh directory, from the repository root:

```bash
uv run -m qward.examples.papers.bv.bv_calibration_simulator \
  --scope scaling --profiles IDEAL NOMINAL STRESS \
  --shots 2048 --runs 3 --seed 20261026 \
  --output-dir /tmp/baseline_20261026_scaling_m2
```

`analysis/*.csv` keep the original `source` paths to `raw/`; those files exist
only after regeneration.
