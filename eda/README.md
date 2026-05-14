# EDA – Exploratory Data Analysis

Skrypty i wykresy z analizy eksploracyjnej zbiorów UCI Heart Disease.

## Pliki

| Skrypt | Opis |
|---|---|
| `01_all_datasets.py` | Porównanie wszystkich 4 ośrodków |
| `02_cleveland_eda.py` | Szczegółowa analiza zbioru Cleveland |

Wykresy zapisywane do `eda/plots/`.

---

## Wnioski

### Wybór zbioru treningowego
Cleveland jako jedyny zbiór jest dobrze zbalansowany (54% zdrowi / 46% chorzy) i ma minimalne braki danych (2%). Pozostałe ośrodki mają masowe braki (99–100% wierszy) i silnie niezrównoważone klasy – nie nadają się do treningu.

### Cechy najsilniej powiązane z chorobą (Cleveland)
| Cecha | Korelacja z targetem | Obserwacja |
|---|---|---|
| `thal` | 0.53 | Odwracalny defekt scyntygrafii → 76% chorych |
| `ca` | 0.46 | Więcej zwężonych naczyń → wyższe ryzyko |
| `exang` | 0.43 | Dławica wysiłkowa → 77% chorych |
| `oldpeak` | 0.43 | Wyższe obniżenie ST → częstsza choroba |
| `thalach` | -0.42 | Niższe max tętno → częstsza choroba |
| `fbs`, `chol` | <0.10 | Praktycznie brak związku z targetem |

### Paradoks bólu (`cp`)
Pacjenci **bezobjawowi** (cp=4) mają najwyższy odsetek chorych – **73%**. Pacjenci z typowym bólem (cp=1–3): tylko 18–30% chorych.

### Płeć
Mężczyźni chorują ponad **2× częściej** niż kobiety (55% vs 26%).

### Wiek
Szczyt zachorowań w grupie **55–65 lat** (62% chorych). Poniżej 45 lat – tylko 25%.

### Wartości odstające
Nieliczne outliery w `chol` (max 564 mg/dl) i `trestbps` (max 200 mmHg) – klinicznie możliwe, nie usunięto.

### Inne wzorce

- **`slope` (nachylenie ST):** nachylenie opadające (typ 2) najsilniej związane z chorobą – odwrotność typowego oczekiwania
- **`restecg` (EKG spoczynkowe):** słaba korelacja (0.17), ale nieprawidłowości ST-T (wartość 2) częściej u chorych
- **Data shift między ośrodkami:** rozkład cech istotnie różni się między Cleveland a pozostałymi zbiorami – VA ma wyraźnie starszych pacjentów (+5 lat), Switzerland prawie wyłącznie chorych → modele trenowane na Cleveland mogą nie generalizować się dobrze
- **Cechy binarne (`fbs`, `exang`, `sex`):** `fbs` (cukier na czczo) praktycznie bez wartości predykcyjnej (0.025), mimo że klinicznie jest czynnikiem ryzyka – prawdopodobnie zbyt mało przypadków pozytywnych w Cleveland (tylko 45/303)

### Pozostałe zbiory (walidacja zewnętrzna)

- **Hungarian:** relatywnie zbalansowany (36/64%), ale masowe braki w `ca`, `thal`, `slope` – cechy o najwyższej sile predykcyjnej będą w całości imputowane → pogorszona jakość predykcji
- **Switzerland:** ekstremalny brak równowagi klas (93.5% chorych) – klasyfikator przewidujący zawsze „chory" miałby 93.5% accuracy, co czyni wyniki trudnymi do interpretacji
- **VA (Long Beach):** najstarsi pacjenci (59 lat), prawie wyłącznie mężczyźni (97%) – inna populacja niż Cleveland → spodziewany data shift
