"""Train the three models on the training split.

Logistic regression and random forest use class_weight="balanced". XGBoost uses
scale_pos_weight and early stopping on the validation split (PR-AUC).
Run:  python src/train.py --data data/creditcard.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import SEED, class_distribution, load_data, prepare  # noqa: E402


def train_all(X_train, y_train, X_val, y_val, seed=SEED):
    """Return a dict of fitted models keyed by name."""
    models = {}

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    models["logistic_regression"] = lr.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    models["random_forest"] = rf.fit(X_train, y_train)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models["xgboost"] = xgb
    return models


def save_artifacts(models, scaler, out_dir="models"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, out / f"{name}.pkl")
    joblib.dump(scaler, out / "amount_scaler.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    df = load_data(args.data)
    print("Class distribution:", json.dumps(class_distribution(df)))
    splits = prepare(df)
    print({k: v.shape[0] for k, v in splits.items() if k.startswith("X_")})

    models = train_all(
        splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"]
    )
    print("XGBoost best iteration:", models["xgboost"].best_iteration)
    save_artifacts(models, splits["scaler"], args.models_dir)
    print(f"Saved models and scaler to {args.models_dir}/")


if __name__ == "__main__":
    main()
