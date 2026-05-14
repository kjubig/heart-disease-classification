"""
01_all_datasets.py
Porównawcza EDA wszystkich 4 zbiorów UCI Heart Disease.
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

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

FILES = {
    "Cleveland":      "datasets/cleveland.csv",
    "Hungarian":      "datasets/hungarian.csv",
    "Switzerland":    "datasets/switzerland.csv",
    "VA (Long Beach)":"datasets/va.csv",
}

# ---------------------------------------------------------------------------
# Wczytanie danych
# ---------------------------------------------------------------------------

datasets = {}
for name, path in FILES.items():
    df = pd.read_csv(path)
    df["source"] = name
    datasets[name] = df

all_df = pd.concat(datasets.values(), ignore_index=True)

# ---------------------------------------------------------------------------
# 1. Statystyki podstawowe
# ---------------------------------------------------------------------------

print("=" * 60)
print("STATYSTYKI PODSTAWOWE")
print("=" * 60)

for name, df in datasets.items():
    n = len(df)
    pos = df["target"].sum()
    neg = n - pos
    missing_rows = df.isnull().any(axis=1).sum()
    age_mean = df["age"].mean()
    age_std = df["age"].std()
    male = (df["sex"] == 1).sum()
    female = (df["sex"] == 0).sum()

    print(f"\n--- {name} ---")
    print(f"  Próbki:          {n}")
    print(f"  Chorzy (1):      {pos} ({pos/n*100:.1f}%)")
    print(f"  Zdrowi (0):      {neg} ({neg/n*100:.1f}%)")
    print(f"  Braki danych:    {missing_rows} wierszy ({missing_rows/n*100:.1f}%)")
    print(f"  Wiek:            {age_mean:.1f} ± {age_std:.1f} lat")
    print(f"  Płeć:            {male} M / {female} K")

    miss_feat = df.isnull().sum()
    miss_feat = miss_feat[miss_feat > 0]
    if len(miss_feat):
        print(f"  Braki wg cech:   {dict(miss_feat)}")

# ---------------------------------------------------------------------------
# 2. Rozkład klas – wykres słupkowy
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
fig.suptitle("Rozkład klas (0 = zdrowy, 1 = chory)", fontsize=13, fontweight="bold")

colors = ["#4C9BE8", "#E8604C"]
for ax, (name, df) in zip(axes, datasets.items()):
    counts = df["target"].value_counts().sort_index()
    bars = ax.bar(["Zdrowy", "Chory"], counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val}\n({val/len(df)*100:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(name, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.25)
    ax.set_ylabel("Liczba próbek" if ax == axes[0] else "")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Zapisano: eda/plots/01_class_distribution.png")

# ---------------------------------------------------------------------------
# 3. Braki danych – heatmapa dla każdego zbioru
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Braki danych wg cech", fontsize=13, fontweight="bold")

for ax, (name, df) in zip(axes, datasets.items()):
    miss_pct = (df.isnull().sum() / len(df) * 100).drop("target")
    colors_bar = ["#E8604C" if v > 0 else "#A8D8A8" for v in miss_pct.values]
    ax.barh(miss_pct.index, miss_pct.values, color=colors_bar)
    ax.set_title(name, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("% braków")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_missing_values.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Zapisano: eda/plots/02_missing_values.png")

# ---------------------------------------------------------------------------
# 4. Rozkład wieku wg ośrodka
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 4))
for name, df in datasets.items():
    ax.hist(df["age"], bins=15, alpha=0.55, label=name, edgecolor="white")
ax.set_xlabel("Wiek")
ax.set_ylabel("Liczba pacjentów")
ax.set_title("Rozkład wieku wg ośrodka", fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_age_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Zapisano: eda/plots/03_age_distribution.png")

# ---------------------------------------------------------------------------
# 5. Tabela podsumowująca
# ---------------------------------------------------------------------------

summary = []
for name, df in datasets.items():
    n = len(df)
    summary.append({
        "Ośrodek": name,
        "n": n,
        "Chorzy (%)": f"{df['target'].sum()/n*100:.1f}%",
        "Wiek śr.": f"{df['age'].mean():.1f} ± {df['age'].std():.1f}",
        "Mężczyźni": f"{(df['sex']==1).sum()} ({(df['sex']==1).sum()/n*100:.0f}%)",
        "Braki wierszy (%)": f"{df.isnull().any(axis=1).sum()/n*100:.1f}%",
    })

print("\n" + "=" * 60)
print("PODSUMOWANIE")
print("=" * 60)
print(pd.DataFrame(summary).to_string(index=False))
