from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

print("Rows, Cols:", df.shape)
print("Class distribution (%):\n", df["Churn"].value_counts(normalize=True) * 100)
print("Missing % by column:\n", df.isna().mean() * 100)

# Plot 1: churn distribution
counts = df["Churn"].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(counts.index, counts.values)
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
ax.set_title("Churn Distribution (Yes vs No)")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            str(v), ha="center", va="bottom")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot1_churn_distribution.png", dpi=200)
plt.clf()

# Plot 2: correlation heatmap with numeric labels
cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]
corr = df[cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")

ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        value = corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                color="black", fontsize=12)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot2_corr_heatmap.png", dpi=200)
plt.clf()