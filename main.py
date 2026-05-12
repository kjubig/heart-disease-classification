"""
main.py
Pipeline klasyfikacji chorób serca – UCI Heart Disease Dataset (#45)

Dane:    data/processed.{cleveland,hungarian,switzerland,va}.data
Wyniki:  results/

Uruchomienie:
    python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from src.data_loader import load_all, get_cleveland_split, get_features_target
from src.preprocessing import (
    impute_train_test,
    scale_train_test,
    apply_pca,
    plot_class_distribution,
    plot_missing_values,
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_pca_variance,
    plot_pca_scatter,
)
from src.classifiers import get_classifiers, cross_validate_all, fit_all
from src.evaluation import (
    evaluate_all,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_shap_summary,
    plot_decision_tree,
    evaluate_per_source,
)

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

USE_PCA         = False   # True → PCA przed klasyfikacją (wyłącza SHAP na cechach)
PCA_VARIANCE    = 0.95
CV_FOLDS        = 10      # 10-fold CV (małe dane)
SHAP_MODELS     = ["Random Forest", "XGBoost"]   # modele do analizy SHAP


def main() -> None:
    print("=" * 65)
    print("  Heart Disease Detection – ML Pipeline (UCI #45)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Wczytanie i podział danych
    # ------------------------------------------------------------------
    print("\n[1] Wczytywanie danych...")
    df = load_all()

    cleveland_df, external_df = get_cleveland_split(df)
    print(f"\n  Trening/CV  : Cleveland  ({len(cleveland_df)} pac.)")
    print(f"  Zewnętrzne  : Hungary, Switzerland, VA Long Beach "
          f"({len(external_df)} pac.)")

    X_cleve, y_cleve = get_features_target(cleveland_df)
    X_ext_raw, y_ext = get_features_target(external_df)

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    print("\n[2] Wizualizacje EDA...")
    plot_class_distribution(y_cleve, title="Cleveland")
    plot_class_distribution(df["target"], title="Wszystkie ośrodki")
    plot_missing_values(X_cleve, title="Cleveland")
    plot_missing_values(X_ext_raw, title="External")
    plot_feature_boxplots(X_cleve, y_cleve)

    # ------------------------------------------------------------------
    # 3. Preprocessing – imputacja
    # ------------------------------------------------------------------
    print("\n[3] Imputacja brakujących wartości...")
    X_cleve_imp, X_ext_imp = impute_train_test(X_cleve, X_ext_raw)
    missing_after = X_cleve_imp.isnull().sum().sum()
    print(f"  Braki po imputacji (Cleveland): {missing_after}")

    # Macierz korelacji po imputacji
    plot_correlation_heatmap(X_cleve_imp, title="Cleveland")

    # ------------------------------------------------------------------
    # 4. Skalowanie
    # ------------------------------------------------------------------
    print("\n[4] Standaryzacja cech...")
    X_cleve_sc, X_ext_sc, scaler = scale_train_test(X_cleve_imp, X_ext_imp)

    # ------------------------------------------------------------------
    # 5. PCA (opcjonalnie)
    # ------------------------------------------------------------------
    if USE_PCA:
        print(f"\n[5] PCA (wariancja >= {PCA_VARIANCE})...")
        X_tr_fit, X_te_fit, pca = apply_pca(X_cleve_sc, X_ext_sc, PCA_VARIANCE)
        plot_pca_variance(pca)
        plot_pca_scatter(X_tr_fit, y_cleve, title="Cleveland")
        feature_names_fit = [f"PC{i+1}" for i in range(X_tr_fit.shape[1])]
    else:
        print("\n[5] PCA pominięte (USE_PCA=False).")
        X_tr_fit = X_cleve_sc
        X_te_fit = X_ext_sc
        feature_names_fit = list(X_cleve_sc.columns)

        # Wizualizacja PCA 2D mimo to (tylko do EDA)
        _, _, pca_vis = apply_pca(X_cleve_sc, X_ext_sc, n_components=2)
        X_pca2, _, _ = apply_pca(X_cleve_sc, X_ext_sc, n_components=2)
        plot_pca_scatter(X_pca2, y_cleve, title="Cleveland (EDA)")

    # ------------------------------------------------------------------
    # 6. Walidacja krzyżowa na Cleveland
    # ------------------------------------------------------------------
    print(f"\n[6] {CV_FOLDS}-fold stratified CV na Cleveland...")
    classifiers = get_classifiers()
    cv_results = cross_validate_all(classifiers, X_tr_fit, y_cleve, cv_folds=CV_FOLDS)
    print("\n[CV Results – Cleveland]")
    print(cv_results.to_string())
    cv_results.to_csv("results/cv_results_cleveland.csv")
    plot_metrics_comparison(cv_results)

    # ------------------------------------------------------------------
    # 7. Trening finalny na całym Cleveland
    # ------------------------------------------------------------------
    print("\n[7] Trening finalny na całym zbiorze Cleveland...")
    trained = fit_all(get_classifiers(), X_tr_fit, y_cleve)

    # ------------------------------------------------------------------
    # 8. Ewaluacja – Cleveland (train set, overfit check)
    # ------------------------------------------------------------------
    print("\n[8] Ewaluacja – Cleveland (train)...")
    evaluate_all(trained, X_tr_fit, y_cleve, label="cleveland_train")

    # ------------------------------------------------------------------
    # 9. Ewaluacja – zewnętrzna (Hungarian + Switzerland + VA)
    # ------------------------------------------------------------------
    print("\n[9] Ewaluacja – walidacja zewnętrzna (łącznie)...")
    evaluate_all(trained, X_te_fit, y_ext, label="external_all")

    # ------------------------------------------------------------------
    # 10. Ewaluacja per ośrodek
    # ------------------------------------------------------------------
    print("\n[10] Ewaluacja per ośrodek geograficzny...")
    # Indeksy zewnętrzne muszą odpowiadać X_te_fit
    external_df_reset = external_df.reset_index(drop=True)
    X_ext_df = pd.DataFrame(X_te_fit, columns=feature_names_fit)
    evaluate_per_source(trained, external_df_reset, X_ext_df, label="external")

    # ------------------------------------------------------------------
    # 11. Macierze pomyłek i krzywe ROC
    # ------------------------------------------------------------------
    print("\n[11] Wizualizacje wyników...")
    plot_confusion_matrices(trained, X_te_fit, y_ext, label="external")
    plot_roc_curves(trained, X_te_fit, y_ext, label="external")

    # ------------------------------------------------------------------
    # 12. Drzewo decyzyjne – wizualizacja
    # ------------------------------------------------------------------
    print("\n[12] Wizualizacja drzewa decyzyjnego...")
    if "Decision Tree" in trained:
        plot_decision_tree(trained["Decision Tree"], feature_names=feature_names_fit)

    # ------------------------------------------------------------------
    # 13. SHAP – interpretowalność (tylko bez PCA)
    # ------------------------------------------------------------------
    if not USE_PCA:
        print("\n[13] Analiza SHAP...")
        X_cleve_df = pd.DataFrame(X_tr_fit, columns=feature_names_fit)
        for model_name in SHAP_MODELS:
            if model_name in trained:
                plot_shap_summary(trained[model_name], X_cleve_df, name=model_name)
    else:
        print("\n[13] SHAP pominięty (USE_PCA=True).")

    print("\n" + "=" * 65)
    print("  [GOTOWE] Wyniki zapisane w folderze results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
