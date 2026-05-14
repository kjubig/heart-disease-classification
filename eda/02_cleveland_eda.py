"""
02_cleveland_eda.py
Szczegółowa EDA zbioru treningowego – Cleveland.
Zapisuje wykresy do eda/plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("datasets/cleveland.csv")

NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CAT_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

FEATURE_LABELS = {
    "age":      "Wiek (lata)",
    "trestbps": "Ciśnienie spoczynkowe (mmHg)",
    "chol":     "Cholesterol (mg/dl)",
    "thalach":  "Max tętno wysiłkowe (udm/min)",
    "oldpeak":  "Obniżenie odcinka ST przy wysiłku",
    "ca":       "Liczba zwężonych naczyń wieńcowych (0–3)",
}

# ---------------------------------------------------------------------------
# 1. Statystyki opisowe
# ---------------------------------------------------------------------------

print("=" * 60)
print("CLEVELAND – STATYSTYKI OPISOWE")
print("=" * 60)
print(df[NUM_FEATURES].describe().round(2))

print("\n--- Korelacja z targetem (|r| malejąco) ---")
corr = df.corr(numeric_only=True)["target"].drop("target")
print(corr.abs().sort_values(ascending=False).round(3))

# ---------------------------------------------------------------------------
# 2. Boxploty cech numerycznych – zdrowi vs chorzy
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle("Rozkład cech numerycznych: zdrowi (0) vs chorzy (1)", fontsize=13, fontweight="bold")

for ax, col in zip(axes.flat, NUM_FEATURES):
    groups = [df[df["target"] == 0][col].dropna(),
              df[df["target"] == 1][col].dropna()]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#4C9BE8")
    bp["boxes"][1].set_facecolor("#E8604C")
    ax.set_xticklabels(["Zdrowy (0)", "Chory (1)"])
    ax.set_title(FEATURE_LABELS.get(col, col), fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_boxplots_cleveland.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Zapisano: eda/plots/04_boxplots_cleveland.png")

# ---------------------------------------------------------------------------
# 3. Macierz korelacji
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df.drop(columns=CAT_FEATURES).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, mask=mask, ax=ax, linewidths=0.5,
            annot_kws={"size": 9})
ax.set_title("Macierz korelacji – Cleveland (cechy numeryczne)", fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Zapisano: eda/plots/05_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 4. Wartości odstające (IQR)
# ---------------------------------------------------------------------------

print("\n--- Wartości odstające (metoda IQR) ---")
for col in NUM_FEATURES:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"  {col:12s}: {n_out:2d} outlierów  |  zakres: [{low:.1f}, {high:.1f}]  |  min={df[col].min()}, max={df[col].max()}")

# ---------------------------------------------------------------------------
# 5. Cechy kategoryczne vs target
# ---------------------------------------------------------------------------

print("\n--- Cechy kategoryczne: % chorych wg wartości ---")
for col in ["cp", "thal", "exang", "sex"]:
    tbl = df.groupby(col)["target"].agg(["mean", "count"]).round(3)
    tbl["mean"] = (tbl["mean"] * 100).round(1).astype(str) + "%"
    print(f"\n  {col}:\n{tbl.to_string()}")

# ---------------------------------------------------------------------------
# 6. Choroba wg grupy wiekowej
# ---------------------------------------------------------------------------

df["age_group"] = pd.cut(df["age"], bins=[0, 45, 55, 65, 100],
                          labels=["<45", "45–55", "55–65", ">65"])
age_stats = df.groupby("age_group", observed=True)["target"].agg(["mean", "count"])

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(age_stats.index.astype(str), age_stats["mean"] * 100,
              color=["#4C9BE8", "#6BB5F0", "#E8604C", "#C04030"], edgecolor="white", width=0.5)
for bar, (_, row) in zip(bars, age_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{row['mean']*100:.0f}%\n(n={int(row['count'])})",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% chorych")
ax.set_xlabel("Grupa wiekowa")
ax.set_title("Częstość choroby wg grupy wiekowej – Cleveland", fontweight="bold")
ax.set_ylim(0, 80)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_age_group_disease.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Zapisano: eda/plots/06_age_group_disease.png")

print("\nEDA Cleveland zakończona.")
