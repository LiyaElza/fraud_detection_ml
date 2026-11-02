"""Evaluate all trained models.

For each model the threshold is chosen on the VALIDATION split and the TEST
split is scored once. Reports cost at the default 0.5 and at the tuned
threshold, plus recall, precision and PR-AUC, and a bootstrap confidence
interval on the cost saving from tuning (positive = tuned threshold is cheaper).

Run:  python src/evaluate.py --data data/creditcard.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.costs import COST_FN, COST_FP, compute_cost, optimize_threshold  # noqa: E402
from src.preprocessing import SEED, load_data, prepare  # noqa: E402

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def per_sample_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    return fn * cost_fn + fp * cost_fp


def bootstrap_saving(y_true, pred_default, pred_tuned, n_boot=1000, seed=SEED):
    """95% bootstrap CI for (cost at default threshold) - (cost at tuned threshold)."""
    diff = per_sample_cost(y_true, pred_default) - per_sample_cost(y_true, pred_tuned)
    rng = np.random.default_rng(seed)
    n = len(diff)
    sums = np.array([rng.choice(diff, size=n, replace=True).sum() for _ in range(n_boot)])
    return float(diff.sum()), [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def evaluate_models(models, X_val, y_val, X_test, y_test, n_boot=1000):
    results = {}
    for name, model in models.items():
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        threshold, sweep = optimize_threshold(y_val, p_val)
        pred_default = (p_test >= 0.5).astype(int)
        pred_tuned = (p_test >= threshold).astype(int)

        saving, ci = bootstrap_saving(y_test, pred_default, pred_tuned, n_boot=n_boot)
        c_default = compute_cost(y_test, pred_default)
        c_tuned = compute_cost(y_test, pred_tuned)

        results[name] = {
            "threshold_chosen_on_validation": round(threshold, 4),
            "pr_auc": float(average_precision_score(y_test, p_test)),
            "default_0.5": {
                **c_default,
                "recall": float(recall_score(y_test, pred_default, zero_division=0)),
                "precision": float(precision_score(y_test, pred_default, zero_division=0)),
            },
            "tuned": {
                **c_tuned,
                "recall": float(recall_score(y_test, pred_tuned, zero_division=0)),
                "precision": float(precision_score(y_test, pred_tuned, zero_division=0)),
            },
            "cost_saving_from_tuning": saving,
            "cost_saving_95ci": ci,
            "validation_sweep": sweep,
        }
    return results


def to_markdown(results):
    lines = [
        "| Model | PR-AUC | Threshold | Cost @0.5 | Cost @tuned | Recall @tuned | Precision @tuned | Saving (95% CI) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['pr_auc']:.3f} | {r['threshold_chosen_on_validation']:.3f} "
            f"| ${r['default_0.5']['total_cost']:,} | ${r['tuned']['total_cost']:,} "
            f"| {r['tuned']['recall']:.3f} | {r['tuned']['precision']:.3f} "
            f"| ${r['cost_saving_from_tuning']:,.0f} "
            f"([{r['cost_saving_95ci'][0]:,.0f}, {r['cost_saving_95ci'][1]:,.0f}]) |"
        )
    lines.append("")
    lines.append(
        f"Costs assume ${COST_FN} per missed fraud and ${COST_FP} per false alarm. "
        "Threshold chosen on the validation split; test split scored once."
    )
    return "\n".join(lines)


def plot_sweep(sweep, threshold, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.plot(sweep["threshold"], sweep["total_cost"], lw=2, color="crimson")
    plt.axvline(threshold, color="green", ls="--", label=f"Chosen on validation: {threshold:.3f}")
    plt.axvline(0.5, color="steelblue", ls=":", label="Default: 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Validation cost ($)")
    plt.title("Threshold vs business cost (validation split)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    splits = prepare(load_data(args.data))
    models = {
        n: joblib.load(models_dir / f"{n}.pkl")
        for n in MODEL_NAMES
        if (models_dir / f"{n}.pkl").exists()
    }
    if not models:
        sys.exit("No trained models found. Run src/train.py first.")

    results = evaluate_models(
        models, splits["X_val"], splits["y_val"], splits["X_test"], splits["y_test"], args.n_boot
    )

    if "xgboost" in results:
        thr = results["xgboost"]["threshold_chosen_on_validation"]
        (models_dir / "optimal_threshold.txt").write_text(str(thr))
        plot_sweep(results["xgboost"]["validation_sweep"], thr, models_dir / "threshold_vs_cost.png")

    serialisable = {
        k: {kk: vv for kk, vv in v.items() if kk != "validation_sweep"} for k, v in results.items()
    }
    (models_dir / "metrics.json").write_text(json.dumps(serialisable, indent=2))
    table = to_markdown(results)
    (models_dir / "results.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
