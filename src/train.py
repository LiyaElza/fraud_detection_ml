import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Done.")
    return model

def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Done.")
    return model

def train_xgboost(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"Training XGBoost with scale_pos_weight={spw:.2f}...")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=spw, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Done.")
    return model

def save_model(model, name):
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    print(f"Saved to {path}")

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.preprocessing import load_data, drop_time, scale_amount, split_data, apply_smote

    df = load_data("data/creditcard.csv")
    df = drop_time(df)
    df, scaler = scale_amount(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    lr = train_logistic_regression(X_train_sm, y_train_sm)
    rf = train_random_forest(X_train_sm, y_train_sm)
    xgb = train_xgboost(X_train, y_train)

    save_model(lr, "logistic_regression")
    save_model(rf, "random_forest")
    save_model(xgb, "xgboost")