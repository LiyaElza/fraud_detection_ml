# Cost-Sensitive Fraud Detection

Credit card fraud detection that evaluates models by **business cost** as well as PR-AUC, with the decision threshold chosen on a validation split. Built with XGBoost, scikit-learn baselines, SHAP explanations and a Streamlit app, with unit tests and CI.

## Problem

The public dataset ([Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) has 284,807 European card transactions from September 2013, of which 492 (0.17%) are fraud. A model that predicts "legitimate" every time scores 99.83% accuracy and catches nothing, so models are compared on recall, precision, PR-AUC and an explicit cost.

## Method

- **Split:** stratified 60/20/20 train / validation / test (170,883 / 56,962 / 56,962 transactions). The `Amount` scaler is fitted on the training split only.
- **Models:** logistic regression and random forest (`class_weight="balanced"`), and XGBoost (`scale_pos_weight`, early stopping on validation PR-AUC).
- **Cost model (assumed, not measured):** a missed fraud costs $100 and a false alarm costs $10. Edit `COST_FN` and `COST_FP` in `src/costs.py` to test other assumptions.
- **Threshold:** swept over 0.01-0.99 on the **validation** split; the test split is scored once.
- **Uncertainty:** a bootstrap 95% confidence interval on the cost saving from threshold tuning. The test split holds only about 100 fraud cases, so small differences in cost between models can be noise.
- **Explainability:** SHAP values for the top fraud drivers and per-transaction explanations. Features V1-V28 are anonymised PCA components, so they cannot be mapped to business meaning.

## Results (test split, scored once)

| Model | PR-AUC | Threshold | Cost @0.5 | Cost @tuned | Recall @tuned | Precision @tuned | Saving (95% CI) |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.719 | 0.990 | $15,100 | $2,080 | 0.857 | 0.553 | $13,020 ([12,220, 13,870]) |
| random_forest | 0.853 | 0.070 | $2,630 | $1,490 | 0.888 | 0.690 | $1,140 ([450, 1,960]) |
| xgboost | 0.882 | 0.410 | $1,700 | $1,710 | 0.837 | 0.882 | -$10 ([-30, 0]) |

Costs assume $100 per missed fraud and $10 per false alarm.

**What the results show**

- **XGBoost ranks fraud best.** It has the highest PR-AUC (0.882) and the highest precision (0.882), so it raises the fewest false alarms.
- **Threshold tuning matters for the weaker-calibrated models, not for XGBoost.** It cut logistic regression's cost from $15,100 to $2,080, because `class_weight="balanced"` inflates its probabilities. For XGBoost the default 0.5 was already close to optimal, and the tuned threshold changed cost by -$10 (95% CI -$30 to $0), so tuning did not help there.
- **Random forest has the lowest cost under these assumed costs** ($1,490), with higher recall (0.888) at lower precision (0.690). The $220 gap to XGBoost was not tested for significance, so with about 100 fraud cases in the test split it could be noise.
- **The best model may depend on the cost ratio.** XGBoost is more precise and random forest has higher recall, so a higher cost per false alarm would probably favour XGBoost and a higher cost per missed fraud random forest. This has not been tested; changing `COST_FN` and `COST_FP` and re-running `evaluate.py` would show it.
- **Top SHAP features for XGBoost:** V14, V4, V12, V11, V10 (mean absolute SHAP; the features are anonymised PCA components).

An earlier version of this repo chose the threshold on the test set, so its numbers are not comparable with these.

## Run it

```bash
pip install -r requirements.txt
# download creditcard.csv from Kaggle into data/
python src/train.py --data data/creditcard.csv
python src/evaluate.py --data data/creditcard.csv
python src/explain.py --data data/creditcard.csv
streamlit run app/streamlit_app.py   # local app
pytest -q
```

## Layout

```
src/preprocessing.py  split + leakage-free scaling
src/costs.py          cost function and threshold selection
src/train.py          three models, saved to models/
src/evaluate.py       validation-tuned threshold, test metrics, bootstrap CI
src/explain.py        SHAP summary, importance table, explainer for the app
app/streamlit_app.py  scoring app with per-transaction explanation
tests/                unit tests and an end-to-end smoke test on synthetic data
.github/workflows/    CI: pytest on every push
```

## Limitations

- The costs are illustrative assumptions.
- One random split; no time-based validation, although fraud patterns drift over time.
- Test-set differences between models are not significance-tested, and the test split holds only about 100 fraud cases.
- Features are anonymised, so explanations are statistical rather than business-readable.
- The Streamlit app runs locally and is not hosted.