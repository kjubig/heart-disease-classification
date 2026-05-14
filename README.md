# Klasyfikacja Choroby Serca – UCI Heart Disease

Projekt realizowany w ramach przedmiotu **Wprowadzenie do Sztucznej Inteligencji (WSI)**.

Źródło danych: [UCI Machine Learning Repository – Heart Disease (dataset #45)](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## Zbiory danych

| Ośrodek | Plik | Liczba próbek |
|---|---|---|
| Cleveland | `processed.cleveland.data` | 303 |
| Hungarian | `processed.hungarian.data` | 294 |
| Switzerland | `processed.switzerland.data` | 123 |
| VA (Long Beach) | `processed.va.data` | 200 |

Przetworzone pliki CSV z nagłówkami dostępne w folderze `datasets/`.

---

## Cechy

| Cecha | Opis | Typ |
|---|---|---|
| `age` | Wiek pacjenta (lata) | numeryczna |
| `sex` | Płeć (0 = kobieta, 1 = mężczyzna) | kategoryczna |
| `cp` | Typ bólu wieńcowego (1–4) | kategoryczna |
| `trestbps` | Ciśnienie spoczynkowe (mmHg) | numeryczna |
| `chol` | Cholesterol (mg/dl) | numeryczna |
| `fbs` | Cukier na czczo > 120 mg/dl (0/1) | kategoryczna |
| `restecg` | Wynik EKG spoczynkowego (0–2) | kategoryczna |
| `thalach` | Maksymalne tętno (udm/min) | numeryczna |
| `exang` | Dławica wysiłkowa (0/1) | kategoryczna |
| `oldpeak` | Obniżenie ST przy wysiłku | numeryczna |
| `slope` | Nachylenie odcinka ST (1–3) | kategoryczna |
| `ca` | Liczba zwężonych naczyń (0–3) | numeryczna |
| `thal` | Wynik scyntygrafii (3/6/7) | kategoryczna |
| `target` | Choroba wieńcowa: 0 = brak, 1 = obecna | zmienna docelowa |
