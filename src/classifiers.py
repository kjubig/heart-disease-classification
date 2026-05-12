"""
classifiers.py
Definicje klasyfikatorów i walidacja krzyżowa.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier


SCORING = ["accuracy", "recall", "precision", "f1", "roc_auc"]


def get_classifiers() -> dict:
    """
    Zwraca słownik {nazwa: klasyfikator}.
    Wszystkie modele z random_state=42 dla reprodukowalności.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=42, C=1.0
        ),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def cross_validate_all(
    classifiers: dict,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv_folds: int = 10,
) -> pd.DataFrame:
    """
    Stratified k-fold CV dla wszystkich klasyfikatorów.
    Zwraca DataFrame z metrykami (mean ± std).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rows = []

    for name, clf in classifiers.items():
        print(f"  → {name} ...", flush=True)
        scores = cross_validate(clf, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"Klasyfikator": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[metric.capitalize()] = f"{vals.mean():.4f} ± {vals.std():.4f}"
            row[f"{metric}_mean"] = vals.mean()
        rows.append(row)

    display_cols = ["Klasyfikator"] + [m.capitalize() for m in SCORING]
    df = pd.DataFrame(rows)
    return df.set_index("Klasyfikator")[display_cols[1:]]


def fit_all(classifiers: dict, X, y) -> dict:
    """Trenuje wszystkie klasyfikatory na pełnym zbiorze treningowym."""
    trained = {}
    for name, clf in classifiers.items():
        print(f"  → {name}")
        clf.fit(X, y)
        trained[name] = clf
    return trained
