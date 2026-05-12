"""
data_loader.py
Wczytywanie 4 baz danych Heart Disease (UCI #45) z lokalnych plików processed.*.data
i przygotowanie do klasyfikacji.

Kolumny (14 atrybutów wg literatury):
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal, num (target)

Binaryzacja targetu: 0 → brak choroby, 1-4 → choroba (→ 1)
Braki danych oznaczone jako '?' zastępowane przez NaN.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

# Cechy kategoryczne (do one-hot lub label encoding)
CAT_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Cechy ciągłe / porządkowe
NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va":         "processed.va.data",
}


def load_single(name: str) -> pd.DataFrame:
    """Wczytuje jeden plik i zwraca DataFrame z kolumną 'source'."""
    path = DATA_DIR / FILES[name]
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df["source"] = name
    return df


def load_all() -> pd.DataFrame:
    """Wczytuje i scala wszystkie 4 bazy danych."""
    frames = [load_single(name) for name in FILES]
    df = pd.concat(frames, ignore_index=True)

    # Binaryzacja: num 0 → 0 (brak), 1-4 → 1 (choroba)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    total = len(df)
    pos = df["target"].sum()
    print(f"[data_loader] Łącznie: {total} pacjentów | "
          f"Choroba: {pos} ({pos/total*100:.1f}%) | "
          f"Brak: {total-pos} ({(total-pos)/total*100:.1f}%)")

    for src, grp in df.groupby("source"):
        p = grp["target"].sum()
        print(f"  {src:12s}: {len(grp):4d} pac. | choroba: {p:3d} ({p/len(grp)*100:.0f}%)")

    return df


def get_cleveland_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca (cleveland_df, others_df).
    Cleveland → trening + CV; pozostałe → walidacja zewnętrzna.
    """
    cleve = df[df["source"] == "cleveland"].copy()
    others = df[df["source"] != "cleveland"].copy()
    return cleve, others


def get_features_target(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Zwraca (X, y) – usuwa 'source', 'target' i opcjonalne kolumny."""
    if drop_cols is None:
        drop_cols = []
    to_drop = list(set(["source", "target"] + drop_cols))
    X = df.drop(columns=[c for c in to_drop if c in df.columns])
    y = df["target"]
    return X, y
