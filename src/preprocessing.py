import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def load_data(path):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    return df

def class_distribution(df):
    counts = df["Class"].value_counts()
    total = len(df)
    print(f"Legitimate: {counts[0]:,}")
    print(f"Fraud:      {counts[1]:,}")
    print(f"Fraud rate: {counts[1]/total*100:.4f}%")


def drop_time(df):
    df = df.drop(columns=["Time"])
    print("Dropped 'Time' column.")
    return df

def scale_amount(df):
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    joblib.dump(scaler, "models/amount_scaler.pkl")
    print("Scaled 'Amount' column.")
    print(f"Amount mean after scaling: {df['Amount'].mean():.4f}")
    print(f"Amount std after scaling:  {df['Amount'].std():.4f}")
    return df, scaler

def split_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
    print(f"Test fraud rate:  {y_test.mean()*100:.4f}%")

    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    print(f"Before SMOTE → Fraud: {y_train.sum()}, Total: {len(y_train)}")

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    print(f"After  SMOTE → Fraud: {y_resampled.sum()}, Total: {len(y_resampled)}")
    print(f"New fraud rate: {y_resampled.mean()*100:.2f}%")

    return X_resampled, y_resampled

if __name__ == "__main__":
    df = load_data("data/creditcard.csv")
    class_distribution(df)
    # print(f"\nMissing values:\n{df.isnull().sum().sum()}")
    # print(f"\nAmount stats:\n{df['Amount'].describe()}")
    df = drop_time(df)
    df, scaler = scale_amount(df)
    # print(f"\nFinal shape: {df.shape}")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
