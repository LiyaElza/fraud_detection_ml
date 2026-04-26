import sys
sys.path.append(".")

import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import load_data

df = load_data("data/creditcard.csv")

# Plot 1: class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["Class"].value_counts().plot.bar(ax=axes[0], color=["steelblue", "crimson"])
axes[0].set_title("Class Distribution")
axes[0].set_xticklabels(["Legitimate", "Fraud"], rotation=0)
axes[0].set_ylabel("Count")

# Plot 2: amount distribution by class
df[df["Class"] == 0]["Amount"].hist(bins=50, ax=axes[1], alpha=0.6, label="Legitimate", color="steelblue")
df[df["Class"] == 1]["Amount"].hist(bins=50, ax=axes[1], alpha=0.6, label="Fraud", color="crimson")
axes[1].set_title("Transaction Amount by Class")
axes[1].set_xlabel("Amount ($)")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebooks/eda_plot.png")
plt.show()
print("Plot saved.")

print("Fraud amount stats:")
print(df[df["Class"] == 1]["Amount"].describe())

print("\nLegit amount stats:")
print(df[df["Class"] == 0]["Amount"].describe())