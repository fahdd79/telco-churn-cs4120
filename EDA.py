import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("plots").mkdir(exist_ok=True)

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

counts = df["Churn"].value_counts()
plt.bar(counts.index, counts.values)
plt.title("Churn Distribution (Yes vs No)")
plt.xlabel("Churn")
plt.ylabel("Count")
for i, v in enumerate(counts.values):
    plt.text(i, v, str(v), ha="center", va="bottom")
plt.tight_layout()
plt.savefig("plots/plot1_churn_distribution.png", dpi=200)
plt.clf()

cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]
corr = df[cols].corr().values
fig, ax = plt.subplots()
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(cols)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig("plots/plot2_corr_heatmap.png", dpi=200)
plt.clf()

print("Rows, Cols:", df.shape)
print("Class distribution (%):\n", df["Churn"].value_counts(normalize=True).rename("proportion")*100)
print("Missing % by column:\n", df.isna().mean()*100)