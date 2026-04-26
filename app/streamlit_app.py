import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
import matplotlib.pyplot as plt

sys.path.append(".")
from src.explain import explain_single

# ── Load artifacts ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("models/xgboost.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    scaler    = joblib.load("models/amount_scaler.pkl")
    threshold = float(open("models/optimal_threshold.txt").read())
    return model, explainer, scaler, threshold

model, explainer, scaler, THRESHOLD = load_artifacts()

# ── Page ────────────────────────────────────────────────────────
st.title("💳 Fraud Detection System")
st.caption(f"XGBoost · Optimal threshold: {THRESHOLD:.3f}")

st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)

st.markdown("**PCA Features (V1–V28)**")
cols = st.columns(4)
features = {}
for i, v in enumerate([f"V{j}" for j in range(1, 29)]):
    features[v] = cols[i % 4].number_input(v, value=0.0, format="%.3f", step=0.1)

# ── Predict ─────────────────────────────────────────────────────
if st.button("Analyze Transaction", type="primary", use_container_width=True):
    scaled_amount = scaler.transform([[amount]])[0][0]
    features["Amount"] = scaled_amount

    row = pd.Series(features)
    X = row.values.reshape(1, -1)

    proba = model.predict_proba(X)[0][1]
    is_fraud = proba >= THRESHOLD

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{proba*100:.1f}%")
    c2.metric("Decision", "FRAUD" if is_fraud else "LEGITIMATE")
    if proba < 0.3:
        c3.metric("Risk Level", "LOW")
    elif proba < 0.6:
        c3.metric("Risk Level", "MEDIUM")
    else:
        c3.metric("Risk Level", "HIGH")

    # SHAP explanation
    st.subheader("Why this prediction?")
    explanation = explain_single(explainer, row, list(features.keys()))
    top = explanation.head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["crimson" if v > 0 else "steelblue" for v in top["shap_value"]]
    ax.barh(top["feature"][::-1], top["shap_value"][::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value  (red = toward fraud, blue = toward legit)")
    ax.set_title("Top 10 Contributing Features")
    plt.tight_layout()
    st.pyplot(fig)