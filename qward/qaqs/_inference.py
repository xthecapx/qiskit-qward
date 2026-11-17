"""Pure-NumPy inference for the frozen QAQS models.

The models were trained with scikit-learn and XGBoost and exported to plain JSON (tree
arrays, support vectors, medians). Inference here needs only NumPy, so QWARD stays light and
loading a model never executes pickled code.

``tests/test_qaqs.py`` checks every prediction against the original scikit-learn / XGBoost
output on stored feature rows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def _impute(X: np.ndarray, statistics: Sequence[float]) -> np.ndarray:
    """Replace NaN with the training medians (``SimpleImputer(strategy="median")``)."""
    out = np.array(X, dtype=float, copy=True)
    fill = np.asarray(statistics, dtype=float)
    mask = np.isnan(out)
    if mask.any():
        out[mask] = np.broadcast_to(fill, out.shape)[mask]
    return out


def _tree_values(tree: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    """Evaluate one binary tree on every row; ``x[feature] <= threshold`` goes left."""
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])
    feature = np.asarray(tree["feature"])
    threshold = np.asarray(tree["threshold"], dtype=float)
    value = np.asarray(tree["value"], dtype=float)
    node = np.zeros(len(X), dtype=int)
    active = left[node] != -1
    while active.any():
        idx = np.flatnonzero(active)
        n = node[idx]
        go_left = X[idx, feature[n]] <= threshold[n]
        node[idx] = np.where(go_left, left[n], right[n])
        active = left[node] != -1
    return value[node]


def _xgb_tree_values(tree: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    """XGBoost goes left when ``x < threshold`` (strict), unlike scikit-learn's ``<=``.

    XGBoost works in float32 end to end, and the JSON thresholds are shortest-round-trip
    decimals of float32 values, so both sides must be compared as float32.
    """
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])
    feature = np.asarray(tree["feature"])
    threshold = np.asarray(tree["threshold"], dtype=np.float32)
    value = np.asarray(tree["value"], dtype=float)
    node = np.zeros(len(X), dtype=int)
    active = left[node] != -1
    while active.any():
        idx = np.flatnonzero(active)
        n = node[idx]
        node[idx] = np.where(X[idx, feature[n]] < threshold[n], left[n], right[n])
        active = left[node] != -1
    return value[node]


def predict_quality(model: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    """Raw (unclipped) prediction of one quality model for rows ``X`` in feature order."""
    pre = model["preprocessing"]
    Z = _impute(X, pre["imputer"])
    if "scaler" in pre:
        Z = (Z - np.asarray(pre["scaler"]["mean"])) / np.asarray(pre["scaler"]["scale"])
    est = model["estimator"]
    kind = model["kind"]
    if kind == "random_forest":
        # scikit-learn casts inputs to float32 before walking trees; do the same.
        Z32 = Z.astype(np.float32).astype(float)
        return np.mean([_tree_values(t, Z32) for t in est["trees"]], axis=0)
    if kind == "xgboost":
        Z32 = Z.astype(np.float32)
        total = np.zeros(len(Z32))
        for tree in est["trees"]:
            total += _xgb_tree_values(tree, Z32)
        return est["base_score"] + total
    if kind == "svr_rbf":
        sv = np.asarray(est["support_vectors"], dtype=float)
        sq = (Z**2).sum(1)[:, None] + (sv**2).sum(1)[None, :] - 2.0 * Z @ sv.T
        k = np.exp(-est["gamma"] * np.maximum(sq, 0.0))
        return k @ np.asarray(est["dual_coef"], dtype=float) + est["intercept"]
    raise ValueError(f"Unknown QAQS model kind: {kind}")


def predict_cost(model: Mapping[str, Any], X: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Billed seconds = physical model + gradient-boosted residual (see the cost model card)."""
    features: List[str] = model["features"]
    phys = model["physical"]
    shots = np.asarray(X["shots"], dtype=float)
    duration = np.asarray(X["duration_s"], dtype=float)
    twirl = np.asarray(X["twirling"], dtype=float)
    dur = np.where(np.isnan(duration), phys["durationFill"], duration)
    design = np.c_[np.ones(len(shots)), dur, dur * twirl, twirl]
    per_shot = design @ np.asarray(phys["shotCoef"], dtype=float)
    circuit = shots * per_shot
    physical = phys["overhead"] + phys["overheadTwirling"] * twirl + circuit
    matrix = np.column_stack([np.asarray(X[f], dtype=float) for f in features])
    Z = _impute(matrix, model["preprocessing"]["imputer"])
    est = model["estimator"]
    residual = np.full(len(Z), est["init"])
    for tree in est["trees"]:
        residual += est["learningRate"] * _tree_values(tree, Z)
    return {"billed_seconds": physical + residual, "physical": physical, "circuit_seconds": circuit}
