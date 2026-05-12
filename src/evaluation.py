"""
evaluation.py
Metryki, macierze pomyłek, krzywe ROC, SHAP, porównanie między ośrodkami.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.tree import export_graphviz, plot_tree

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Metryki
# ---------------------------------------------------------------------------

def evaluate(clf, X_test, y_test, name: str = "") -> dict:
    y_pred = clf.predict(X_test)
    y_prob = (clf.predict_proba(X_test)[:, 1]
              if hasattr(clf, "predict_proba") else None)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Klasyfikator": name,
        "Accuracy":    accuracy_score(y_test, y_pred),
        "Sensitivity": recall_score(y_test, y_pred),
        "Specificity": specificity,
        "Precision":   precision_score(y_test, y_pred, zero_division=0),
        "F1":          f1_score(y_test, y_pred),
        "AUC-ROC":     roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
    }


def evaluate_all(
    trained: dict,
    X_test,
    y_test,
    label: str = "test",
) -> pd.DataFrame:
    rows = [evaluate(clf, X_test, y_test, name=name)
            for name, clf in trained.items()]
    df = pd.DataFrame(rows).set_index("Klasyfikator")
    print(f"\n[Wyniki – {label}]")
    print(df.round(4).to_string())
    df.to_csv(RESULTS_DIR / f"metrics_{label}.csv")
    return df


# ---------------------------------------------------------------------------
# Macierz pomyłek
# ---------------------------------------------------------------------------

def plot_confusion_matrices(trained: dict, X_test, y_test,
                            label: str = "", save: bool = True) -> None:
    n = len(trained)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()

    for i, (name, clf) in enumerate(trained.items()):
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Brak", "Choroba"],
        )
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(name, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Macierze pomyłek – {label}", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f"confusion_matrices_{label}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Krzywe ROC
# ---------------------------------------------------------------------------

def plot_roc_curves(trained: dict, X_test, y_test,
                   label: str = "", save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, clf in trained.items():
        if hasattr(clf, "predict_proba"):
            RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Losowy")
    ax.set_title(f"Krzywe ROC – {label}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f"roc_curves_{label}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Porównanie klasyfikatorów – wykres słupkowy
# ---------------------------------------------------------------------------

def plot_metrics_comparison(df_cv: pd.DataFrame, save: bool = True) -> None:
    """Wykres słupkowy AUC-ROC i Accuracy z CV."""
    # Wyciągamy wartości mean z formatu "0.xxxx ± 0.xxxx"
    metrics = {}
    for col in df_cv.columns:
        metrics[col] = df_cv[col].apply(
            lambda v: float(v.split("±")[0].strip()) if isinstance(v, str) else v
        )
    df_plot = pd.DataFrame(metrics)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["Roc_auc", "Accuracy"]):
        if metric not in df_plot.columns:
            metric = metric.lower()
        col = [c for c in df_plot.columns if c.lower() == metric.lower()]
        if not col:
            continue
        vals = df_plot[col[0]].sort_values(ascending=True)
        vals.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlim(0.5, 1.0)
        ax.axvline(0.8, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(col[0])
        ax.set_xlabel("Wartość")

    plt.suptitle("Porównanie klasyfikatorów (10-fold CV – Cleveland)")
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / "cv_comparison.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# SHAP – interpretowalność
# ---------------------------------------------------------------------------

def plot_shap_summary(clf, X_train: pd.DataFrame, name: str = "",
                      save: bool = True) -> None:
    try:
        import shap
    except ImportError:
        print("[SHAP] Zainstaluj: pip install shap")
        return

    print(f"  [SHAP] Obliczanie wartości SHAP dla: {name} ...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(
        shap_values, X_train,
        feature_names=list(X_train.columns),
        show=False,
    )
    if save:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(RESULTS_DIR / f"shap_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Wizualizacja drzewa decyzyjnego
# ---------------------------------------------------------------------------

def plot_decision_tree(clf, feature_names: list, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Brak choroby", "Choroba"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    plt.title("Drzewo decyzyjne (max_depth=5)")
    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / "decision_tree.png", dpi=120)
    plt.close()
    print("  [DT] Zapisano wizualizację drzewa → results/decision_tree.png")


# ---------------------------------------------------------------------------
# Porównanie między ośrodkami
# ---------------------------------------------------------------------------

def evaluate_per_source(
    trained: dict,
    df_external: pd.DataFrame,
    X_external,
    label: str = "external",
) -> pd.DataFrame:
    """Ewaluacja osobno per ośrodek geograficzny."""
    sources = df_external["source"].unique()
    rows = []
    for src in sources:
        mask = df_external["source"] == src
        X_src = X_external[mask.values] if hasattr(X_external, "__getitem__") else X_external[mask]
        y_src = df_external.loc[mask, "target"]
        if len(y_src) == 0 or y_src.nunique() < 2:
            continue
        for name, clf in trained.items():
            m = evaluate(clf, X_src, y_src, name=name)
            m["Source"] = src
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"metrics_{label}_per_source.csv", index=False)
    print(f"\n[Wyniki per ośrodek – {label}]")
    print(df.set_index(["Source", "Klasyfikator"]).round(3).to_string())
    return df
