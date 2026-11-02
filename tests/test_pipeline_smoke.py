"""End-to-end smoke test on small synthetic data (no Kaggle file needed)."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.evaluate import evaluate_models, to_markdown
from src.preprocessing import prepare
from src.train import train_all


def make_fake_creditcard(n=6000, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=10, weights=[0.97, 0.03],
        class_sep=1.5, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.default_rng(seed)
    df["Time"] = np.arange(n)
    df["Amount"] = rng.gamma(2.0, 50.0, size=n)
    df["Class"] = y
    return df


def test_train_and_evaluate_smoke():
    s = prepare(make_fake_creditcard())
    models = train_all(s["X_train"], s["y_train"], s["X_val"], s["y_val"])
    assert set(models) == {"logistic_regression", "random_forest", "xgboost"}

    results = evaluate_models(
        models, s["X_val"], s["y_val"], s["X_test"], s["y_test"], n_boot=50
    )
    for r in results.values():
        assert 0.0 <= r["pr_auc"] <= 1.0
        assert 0.0 < r["threshold_chosen_on_validation"] < 1.0
        lo, hi = r["cost_saving_95ci"]
        assert lo <= hi
    assert "| xgboost |" in to_markdown(results)
