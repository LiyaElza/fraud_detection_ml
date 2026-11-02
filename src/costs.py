"""Business-cost logic and threshold selection.

Costs are assumptions, not measured values: a missed fraud costs COST_FN and a
blocked legitimate transaction costs COST_FP. Change them here to re-run the
analysis under different assumptions.
"""
import numpy as np
import pandas as pd

COST_FN = 100  # missed fraud: the institution absorbs the loss
COST_FP = 10   # blocked legitimate transaction: customer friction


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tn, fp, fn, tp


def compute_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "total_cost": fn * cost_fn + fp * cost_fp,
    }


def sweep_thresholds(y_true, proba, thresholds=None, cost_fn=COST_FN, cost_fp=COST_FP):
    """Total cost at each threshold. Returns a DataFrame."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        c = compute_cost(y_true, (proba >= t).astype(int), cost_fn, cost_fp)
        rows.append({"threshold": float(t), "total_cost": c["total_cost"], "fn": c["fn"], "fp": c["fp"]})
    return pd.DataFrame(rows)


def optimize_threshold(y_true, proba, thresholds=None, cost_fn=COST_FN, cost_fp=COST_FP):
    """Pick the cost-minimising threshold. Call this on VALIDATION data only.

    Ties are broken towards the threshold closest to 0.5 so the choice is stable.
    """
    df = sweep_thresholds(y_true, proba, thresholds, cost_fn, cost_fp)
    best_cost = df["total_cost"].min()
    tied = df[df["total_cost"] == best_cost].copy()
    tied["dist"] = (tied["threshold"] - 0.5).abs()
    best = tied.sort_values("dist").iloc[0]
    return float(best["threshold"]), df
