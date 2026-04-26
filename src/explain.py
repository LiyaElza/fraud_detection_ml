import shap
import matplotlib.pyplot as plt
import joblib

def get_shap_values(model, X_sample):
    print(f"Computing SHAP values for {len(X_sample):,} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print("Done.")
    return explainer, shap_values

def plot_summary(shap_values, X_sample):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Top Features Driving Fraud Predictions")
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", bbox_inches="tight")
    plt.show()
    print("Saved to models/shap_summary.png")

def explain_single(explainer, transaction, feature_names):
    import pandas as pd
    import numpy as np

    row = transaction.values.reshape(1, -1)
    sv = explainer.shap_values(row)[0]

    result = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "abs_impact": np.abs(sv)
    }).sort_values("abs_impact", ascending=False)

    print(result.head(10).to_string(index=False))
    return result

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.preprocessing import load_data, drop_time, scale_amount, split_data

    df = load_data("data/creditcard.csv")
    df = drop_time(df)
    df, scaler = scale_amount(df)
    X_train, X_test, y_train, y_test = split_data(df)

    xgb = joblib.load("models/xgboost.pkl")

    # Use a sample of 2000 rows — computing on full dataset is slow
    X_sample = X_train.sample(2000, random_state=42)

    explainer, shap_values = get_shap_values(xgb, X_sample)
    plot_summary(shap_values, X_sample)

    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("Saved to models/shap_explainer.pkl")

    # explain one fraud transaction
    fraud_idx = y_test[y_test == 1].index[0]
    transaction = X_test.loc[fraud_idx]
    print("\nExplaining one fraud transaction:")
    explain_single(explainer, transaction, list(X_test.columns))


    # Print a real fraud transaction's feature values
    fraud_idx = y_test[y_test == 1].index[0]
    print("Feature values for a real fraud transaction:")
    print(X_test.loc[fraud_idx].to_dict())