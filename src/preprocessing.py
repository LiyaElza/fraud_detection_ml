"""Data loading, splitting and scaling.

The split is train / validation / test (60/20/20, stratified). Anything that is
fitted (the Amount scaler) is fitted on the training split only, and the
decision threshold is tuned on the validation split. The test split is only
used for the final report.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_data(path):
    return pd.read_csv(path)


def class_distribution(df):
    counts = df["Class"].value_counts()
    return {
        "legitimate": int(counts.get(0, 0)),
        "fraud": int(counts.get(1, 0)),
        "fraud_rate_pct": float(counts.get(1, 0) / len(df) * 100),
    }


def drop_time(df):
    return df.drop(columns=["Time"], errors="ignore")


def split_data(df, seed=SEED, val_size=0.2, test_size=0.2):
    """Stratified train/validation/test split."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_fraction_of_rest = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction_of_rest, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_amount_scaler(X_train):
    """Fit the Amount scaler on the training split only (no leakage)."""
    return StandardScaler().fit(X_train[["Amount"]])


def apply_amount_scaler(X, scaler):
    X = X.copy()
    X["Amount"] = scaler.transform(X[["Amount"]]).ravel()
    return X


def prepare(df, seed=SEED):
    """Drop Time, split, scale. Returns a dict of splits plus the fitted scaler."""
    df = drop_time(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=seed)
    scaler = fit_amount_scaler(X_train)
    return {
        "X_train": apply_amount_scaler(X_train, scaler),
        "X_val": apply_amount_scaler(X_val, scaler),
        "X_test": apply_amount_scaler(X_test, scaler),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
    }
