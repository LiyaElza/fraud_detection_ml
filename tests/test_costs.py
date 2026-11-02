import numpy as np
import pandas as pd

from src.costs import COST_FN, COST_FP, compute_cost, optimize_threshold, sweep_thresholds


def test_compute_cost_known_confusion_matrix():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])  # tp=1, fn=1, fp=2, tn=2
    c = compute_cost(y_true, y_pred)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 2, 2)
    assert c["total_cost"] == 1 * COST_FN + 2 * COST_FP


def test_compute_cost_handles_missing_class_in_predictions():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    c = compute_cost(y_true, y_pred)
    assert c["fn"] == 1 and c["fp"] == 0 and c["total_cost"] == COST_FN


def test_optimize_threshold_separable_data_has_zero_cost():
    y = np.array([0] * 50 + [1] * 10)
    proba = np.array([0.05] * 50 + [0.95] * 10)
    thr, sweep = optimize_threshold(y, proba)
    assert isinstance(sweep, pd.DataFrame)
    assert 0.05 < thr <= 0.95
    assert sweep.loc[sweep["threshold"] == thr, "total_cost"].iloc[0] == 0


def test_expensive_misses_push_threshold_down():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 50)
    proba = np.concatenate([rng.beta(2, 8, 500), rng.beta(5, 3, 50)])
    thr_cheap_fn, _ = optimize_threshold(y, proba, cost_fn=10, cost_fp=10)
    thr_costly_fn, _ = optimize_threshold(y, proba, cost_fn=1000, cost_fp=10)
    assert thr_costly_fn <= thr_cheap_fn


def test_sweep_has_one_row_per_threshold():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8])
    sweep = sweep_thresholds(y, proba, thresholds=[0.3, 0.5, 0.7])
    assert len(sweep) == 3
