"""
preprocessing.py
Imputacja, encoding, skalowanie, PCA oraz wizualizacje EDA.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # tryb bez GUI – zapis do pliku
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from pathlib import Path

from src.data_loader import CAT_FEATURES, NUM_FEATURES

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Imputacja
# ---------------------------------------------------------------------------

def impute(X: pd.DataFrame) -> pd.DataFrame:
    """
    Imputacja brakujących wartości:
      - cechy numeryczne  → mediana
      - cechy kategoryczne → moda (najczęstsza wartość)
    """
    X = X.copy()
    num_cols = [c for c in NUM_FEATURES if c in X.columns]
    cat_cols = [c for c in CAT_FEATURES if c in X.columns]

    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

    return X


def impute_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dopasowuje imputery na X_train, stosuje na obu zbiorach.
    Zapobiega data leakage.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = [c for c in NUM_FEATURES if c in X_train.columns]
    cat_cols = [c for c in CAT_FEATURES if c in X_train.columns]

    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        X_train[num_cols] = imp_num.fit_transform(X_train[num_cols])
        X_test[num_cols] = imp_num.transform(X_test[num_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = imp_cat.transform(X_test[cat_cols])

    return X_train, X_test


# ---------------------------------------------------------------------------
# Skalowanie
# ---------------------------------------------------------------------------

def scale_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """StandardScaler dopasowany na X_train."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_te = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index,
    )
    return X_tr, X_te, scaler


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: float | int = 0.95,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    print(f"[PCA] Składowe: {pca.n_components_} | "
          f"Wyjaśniona wariancja: {pca.explained_variance_ratio_.sum():.3f}")
    return X_tr, X_te, pca


# ---------------------------------------------------------------------------
# Wizualizacje EDA
# ---------------------------------------------------------------------------

def plot_class_distribution(y: pd.Series, title: str = "Cleveland", save: bool = True) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Brak choroby (0)", "Choroba (1)"]
    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, counts.values, color=["steelblue", "tomato"])
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha="center", va="bottom", fontsize=11)
    plt.title(f"Rozkład klas – {title}")
    plt.ylabel("Liczba pacjentów")
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f"class_dist_{title.lower()}.png", dpi=150)
    plt.close()


def plot_missing_values(X: pd.DataFrame, title: str = "", save: bool = True) -> None:
    missing = X.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print(f"[EDA] Brak brakujących wartości w {title}.")
        return
    plt.figure(figsize=(8, 4))
    missing.plot(kind="bar", color="coral")
    plt.axhline(0.05, color="red", linestyle="--", label="5%")
    plt.ylabel("Odsetek braków")
    plt.title(f"Brakujące wartości – {title}")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f"missing_{title.lower().replace(' ','_')}.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(X: pd.DataFrame, title: str = "", save: bool = True) -> None:
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", linewidths=0.4, annot_kws={"size": 8})
    plt.title(f"Macierz korelacji – {title}")
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f"corr_{title.lower().replace(' ','_')}.png", dpi=150)
    plt.close()


def plot_feature_boxplots(X: pd.DataFrame, y: pd.Series, save: bool = True) -> None:
    """Boxploty cech ciągłych względem targetu."""
    num_cols = [c for c in NUM_FEATURES if c in X.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data = [X[col][y == 0].dropna().values, X[col][y == 1].dropna().values]
        axes[i].boxplot(data, labels=["Brak choroby", "Choroba"], patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
        axes[i].set_title(col)
        axes[i].set_ylabel(col)
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Cechy ciągłe vs. diagnoza", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / "boxplots_features.png", dpi=150)
    plt.close()


def plot_pca_variance(pca: PCA, save: bool = True) -> None:
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(7, 4))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, alpha=0.6, label="Pojedyncza składowa")
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o",
             color="red", linewidth=1.5, label="Skumulowana")
    plt.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Składowa główna")
    plt.ylabel("Wyjaśniona wariancja")
    plt.title("PCA – wyjaśniona wariancja")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / "pca_variance.png", dpi=150)
    plt.close()


def plot_pca_scatter(X_pca: np.ndarray, y: pd.Series, title: str = "", save: bool = True) -> None:
    """Rzut na pierwsze 2 składowe PCA z podziałem na klasy."""
    plt.figure(figsize=(7, 5))
    for label, color, name in [(0, "steelblue", "Brak choroby"), (1, "tomato", "Choroba")]:
        mask = y.values == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=name, alpha=0.6, s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA – projekcja 2D {title}")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f"pca_scatter_{title.lower().replace(' ','_')}.png", dpi=150)
    plt.close()
