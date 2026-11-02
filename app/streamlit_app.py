import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(".")
from src.explain import explain_single  # noqa: E402


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    scaler = joblib.load("models/amount_scaler.pkl")
    with open("models/optimal_threshold.txt") as f:
        threshold = float(f.read())
    return model, explainer, scaler, threshold


model, explainer, scaler, THRESHOLD = load_artifacts()

st.title("Fraud Detection System")
st.caption(f"XGBoost · decision threshold {THRESHOLD:.3f} (chosen on a validation split)")

st.subheader("Enter transaction details")
amount = st.number_input("Transaction amount ($)", min_value=0.0, value=100.0, step=10.0)

st.markdown("**PCA features (V1-V28)**")
cols = st.columns(4)
features = {}
for i, v in enumerate(f"V{j}" for j in range(1, 29)):
    features[v] = cols[i % 4].number_input(v, value=0.0, format="%.3f", step=0.1)

if st.button("Analyze transaction", type="primary", use_container_width=True):
    features["Amount"] = scaler.transform(pd.DataFrame({"Amount": [amount]}))[0][0]
    row = pd.Series(features)
    proba = float(model.predict_proba(row.values.reshape(1, -1))[0][1])
    is_fraud = proba >= THRESHOLD

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud probability", f"{proba * 100:.1f}%")
    c2.metric("Decision", "FRAUD" if is_fraud else "LEGITIMATE")
    c3.metric("Risk level", "LOW" if proba < 0.3 else "MEDIUM" if proba < 0.6 else "HIGH")

    st.subheader("Why this prediction?")
    top = explain_single(explainer, row, list(features.keys())).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["crimson" if v > 0 else "steelblue" for v in top["shap_value"]]
    ax.barh(top["feature"][::-1], top["shap_value"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (red = toward fraud, blue = toward legitimate)")
    ax.set_title("Top 10 contributing features")
    plt.tight_layout()
    st.pyplot(fig)
