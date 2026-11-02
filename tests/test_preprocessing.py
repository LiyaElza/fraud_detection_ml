import numpy as np
import pandas as pd
import pytest

from src.preprocessing import apply_amount_scaler, drop_time, fit_amount_scaler, prepare, split_data


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.arange(n)
    df["Amount"] = rng.gamma(2.0, 50.0, size=n)
    df["Class"] = (rng.random(n) < 0.02).astype(int)
    return df


def test_drop_time_removes_column(fake_df):
    assert "Time" not in drop_time(fake_df).columns


def test_split_sizes_and_no_overlap(fake_df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(drop_time(fake_df))
    n = len(fake_df)
    assert len(X_tr) + len(X_va) + len(X_te) == n
    assert abs(len(X_te) / n - 0.2) < 0.01
    assert abs(len(X_va) / n - 0.2) < 0.01
    assert not set(X_tr.index) & set(X_va.index)
    assert not set(X_tr.index) & set(X_te.index)
    assert not set(X_va.index) & set(X_te.index)


def test_split_is_stratified(fake_df):
    _, _, _, y_tr, y_va, y_te = split_data(drop_time(fake_df))
    rate = fake_df["Class"].mean()
    for y in (y_tr, y_va, y_te):
        assert abs(y.mean() - rate) < 0.005


def test_scaler_fit_on_train_only(fake_df):
    X_tr, X_va, _, _, _, _ = split_data(drop_time(fake_df))
    scaler = fit_amount_scaler(X_tr)
    assert scaler.mean_[0] == pytest.approx(X_tr["Amount"].mean())
    scaled_train = apply_amount_scaler(X_tr, scaler)
    assert scaled_train["Amount"].mean() == pytest.approx(0.0, abs=1e-8)
    # validation is scaled with the training statistics, so its mean is not forced to 0
    scaled_val = apply_amount_scaler(X_va, scaler)
    assert scaled_val["Amount"].mean() != pytest.approx(0.0, abs=1e-12)


def test_prepare_returns_expected_keys_and_columns(fake_df):
    s = prepare(fake_df)
    assert {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "scaler"} <= set(s)
    assert list(s["X_train"].columns)[-1] == "Amount"
    assert "Time" not in s["X_train"].columns
