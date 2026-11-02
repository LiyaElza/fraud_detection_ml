"""SHAP explanations for the XGBoost model.

Run:  python src/explain.py --data data/creditcard.csv
Saves models/shap_summary.png, models/shap_importance.csv and
models/shap_explainer.pkl (used by the Streamlit app).
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import SEED, load_data, prepare  # noqa: E402


def get_shap_values(model, X_sample):
    import shap

    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_sample)


def mean_abs_importance(shap_values, feature_names):
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_summary(shap_values, X_sample, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Top features driving fraud predictions")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def explain_single(explainer, transaction, feature_names):
    """Per-feature SHAP values for one transaction (a pandas Series)."""
    row = transaction.values.reshape(1, -1)
    sv = explainer.shap_values(row)[0]
    return (
        pd.DataFrame(
            {"feature": feature_names, "shap_value": sv, "abs_impact": np.abs(sv)}
        )
        .sort_values("abs_impact", ascending=False)
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    splits = prepare(load_data(args.data))
    model = joblib.load(models_dir / "xgboost.pkl")

    X_sample = splits["X_val"].sample(
        min(args.n_samples, len(splits["X_val"])), random_state=SEED
    )
    explainer, shap_values = get_shap_values(model, X_sample)
    plot_summary(shap_values, X_sample, models_dir / "shap_summary.png")
    importance = mean_abs_importance(shap_values, list(X_sample.columns))
    importance.to_csv(models_dir / "shap_importance.csv", index=False)
    joblib.dump(explainer, models_dir / "shap_explainer.pkl")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
