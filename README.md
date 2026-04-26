# Cost Sensitive Fraud Detection System

A machine learning system that detects credit card fraud by minimizing financial loss. The project is built with XGBoost, threshold optimization and SHAP explainability. An interactive app is built using Streamlit to use the trained model to detect fraud cases.

## Problem Statement

Here we are using the public dataset that consists of credit card transactions in September 2013 by European cardholders. The key problem lies in the representation of fraud cases in the dataset. Out of 284807 transactions in the dataset, only 492 are fraud cases. That is, just 0.17%.

Here, a model which falsely predicts all the transactions as "legitimate" can still acheive 99.83% accuracy without catching any fraud cases. Hence, we are using a business cost function as metrics of the machine learning system rather than accuracy.

## Key Conecpts

### Cost-Sensitive Learning

The model is evaluated based on financial loss, not accuracy. Here, missing a fraud costs more than a false alarm.

#### Cost Model 

False Positives - Legitimate transaction blocked - Cost $10
False Negatives - Fraud transaction missed causing loss for the financial institution - $100

Total Cost = (False Negatives * $100) + (False Positives * $10)


### Threshold Optimization

Instead of using, the default 0.5 cutoff for legitimate/fraud trade-off, we sweep all thresholds and pick the one which minimizes the business cost

### Shap Explainability

Every prediction made by the model comes with an explaination about what features in what level influenced the final decision.

### Handle Class imbalance

For baseline Logistic Regression and Random Forest models, SMOTE technique and for XGBoost, it built-in scale_pos_weight method are used to handle class imbalance.



## Key Results

### Threshold 0.5 (default)

False Negatives - 15
False Positives - 13
Total Cost - $1630

### Threshold 0.246 (optimized)

False Negatives - 14
False Positives - 21
Total Cost - $1610

### Top fraud-driving features (SHAP analysis results):

V14, V4, V12, V10, V17 (in descending order)


## How to Run

### Download Data

Download the file creditcard.csv from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and store it in the folder data/

### Install depependencies

pip install -r requirements.txt

### Train the models

python src/train.py

### Evaluation and threshold optimization

python src/evaluate.py

### SHAP explanations

python src/explain.py

### Launch Streamlit App

streamlit run app/streamlit_app.py









