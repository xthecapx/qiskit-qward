# QAQS: predict circuit quality and cost before running

`qward.qaqs` predicts how well a circuit will run on **IBM Fez** and **IBM Kingston**, and what
the job will cost, from the circuit alone. It computes the circuit's *footprint* (pre-runtime
metrics of the logical and the compiled circuit) and applies frozen models trained on 2,409
real executions. A runnable demo is in
[`qward/examples/qaqs_predict_quality_and_cost.ipynb`](https://github.com/xthecapx/qiskit-qward/blob/main/qward/examples/qaqs_predict_quality_and_cost.ipynb).

```python
from qiskit_ibm_runtime.fake_provider import FakeFez
from qward.qaqs import predict

result = predict(circuit, FakeFez())     # or a real backend: service.backend("ibm_fez")
result.footprint                         # a.* / b.* metrics, as in the thesis dataset
result.quality["ibm_fez/success"]        # value, model, validation R², verdict, warnings
result.cost                              # billed seconds, resource units, whole seconds charged
result.quality_table()                   # all predictions as rows for pandas
```

## Install

```bash
pip install qiskit-qward              # footprint and predictions
pip install "qiskit-qward[torch]"     # also the quantum-specific metrics (PyTorch)
```

Inference needs NumPy only. The models are plain JSON inside the package (about 4 MB), never
pickles, so installing or loading QWARD does not execute model code.

## What is predicted

| Output | Models | Notes |
| --- | --- | --- |
| Success rate, Hellinger fidelity, TVD, DSR | one per target for `ibm_fez` and `ibm_kingston` | circuit footprint only (no calibration data) |
| Billed QPU time and resource units (1 RU = 1 minute) | one model for every chip | validated at 1,024 shots |

Each prediction carries the score of its model on circuits it had never seen. Success rate,
Hellinger fidelity and TVD scored R² 0.90 to 0.94 on the training chip in the thesis' new-run
test (some intervals straddle the 0.80 threshold and are reported as *inconclusive*). DSR scored
0.05 to 0.29 and is flagged as unreliable.

## Reading the warnings

`predict` warns instead of guessing when:

* the circuit has more two-qubit gates, depth or qubits than anything in training. Predictions
  for much larger circuits were not reliable in validation;
* a model input is outside the range seen in training;
* quantum-specific metrics are missing (circuit wider than 8 qubits, dynamic, or PyTorch
  absent). They are computed up to 8 qubits (a few seconds at 6, over a minute at 8);
* the backend is fake, so the compiled depth can differ from the real chip by up to ~12%;
* the cost is estimated for a shot count other than 1,024 (an extrapolation of IBM's usage
  formula, not a validated result);
* the circuit has no barrier. One is added before the final measurements, because every
  training circuit had one and the footprint is sensitive to it (`add_barrier=False` disables
  this).

A circuit that compiles to no gates (for example a transform followed by its own inverse, with no
barrier between them) is refused with an explanation.

## How it was validated

The footprint code reproduces the values stored with 20 real validation circuits exactly, and
NumPy inference matches scikit-learn / XGBoost to 4e-7. The full evidence, including the
pre-registered new-run test on Fez, Kingston and the unseen Marrakesh, is in the thesis
repository (`report/ml_estimation/`). `tests/test_qaqs.py` checks all of it offline.

## Limits

* Two chips and one cost model only; Marrakesh was a validation target, not a model.
* Cost is validated at 1,024 shots.
* The models describe execution on IBM Heron chips in November 2026 and do not know a chip's
  current calibration.
