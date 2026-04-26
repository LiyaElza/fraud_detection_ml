from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COST_FN = 100  # missed fraud — bank absorbs the loss
COST_FP = 10   # blocked legit transaction — customer friction

def compute_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * COST_FN) + (fp * COST_FP)

    print(f"True Negatives  (correct legit)  : {tn:,}")
    print(f"False Positives (blocked legit)  : {fp:,}  → ${fp * COST_FP:,}")
    print(f"False Negatives (missed fraud)   : {fn:,}  → ${fn * COST_FN:,}")
    print(f"True Positives  (caught fraud)   : {tp:,}")
    print(f"\nTotal Business Cost: ${total_cost:,}")

    return total_cost, fn, fp

def compute_cost_silent(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return (fn * COST_FN) + (fp * COST_FP), fn, fp

def optimize_threshold(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)
    results = []

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        cost, fn, fp = compute_cost_silent(y_test, y_pred)
        results.append({"threshold": t, "total_cost": cost, "fn": fn, "fp": fp})

    df = pd.DataFrame(results)
    best = df.loc[df["total_cost"].idxmin()]

    print(f"Optimal threshold : {best['threshold']:.3f}")
    print(f"Total cost        : ${best['total_cost']:,.0f}")
    print(f"False Negatives   : {int(best['fn'])}")
    print(f"False Positives   : {int(best['fp'])}")

    return best["threshold"], df

def plot_threshold_vs_cost(threshold_df, optimal_threshold):
    plt.figure(figsize=(10, 5))
    plt.plot(threshold_df["threshold"], threshold_df["total_cost"], color="crimson", lw=2)
    plt.axvline(optimal_threshold, color="green", linestyle="--", lw=2, label=f"Optimal: {optimal_threshold:.3f}")
    plt.axvline(0.5, color="steelblue", linestyle=":", lw=2, label="Default: 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Total Business Cost ($)")
    plt.title("Threshold vs Business Cost")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/threshold_vs_cost.png")
    plt.show()
    print("Saved to models/threshold_vs_cost.png")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import joblib
    from src.preprocessing import load_data, drop_time, scale_amount, split_data

    df = load_data("data/creditcard.csv")
    df = drop_time(df)
    df, scaler = scale_amount(df)
    X_train, X_test, y_train, y_test = split_data(df)

    xgb = joblib.load("models/xgboost.pkl")
    y_pred = xgb.predict(X_test)

    print("XGBoost at default threshold (0.5):")
    compute_cost(y_test, y_pred)

    print("\nOptimizing threshold...")
    optimal_threshold, threshold_df = optimize_threshold(xgb, X_test, y_test)

    plot_threshold_vs_cost(threshold_df, optimal_threshold)

    with open("models/optimal_threshold.txt", "w") as f:
        f.write(str(round(float(optimal_threshold), 4)))
    print(f"Saved optimal threshold: {optimal_threshold:.4f}")