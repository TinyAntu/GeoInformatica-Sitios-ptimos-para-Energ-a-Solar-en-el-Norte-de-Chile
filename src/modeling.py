import json
import os
import joblib
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def _entrenar_y_evaluar(positivas_df, pool_neg_df, features, ratio, n_estimators, random_state):
    """Entrena un RF con el ratio dado y retorna (auc, importances, clf, dataset)."""
    n_sample = min(int(len(positivas_df) * ratio), len(pool_neg_df))
    if n_sample == 0:
        return None, None, None, None
    neg_sample = pool_neg_df.sample(n=n_sample, random_state=random_state)
    dataset = gpd.GeoDataFrame(
        pd.concat([positivas_df, neg_sample], ignore_index=True),
        geometry='geometry', crs='EPSG:32718',
    ).dropna(subset=features)
    if len(dataset) == 0:
        return None, None, None, None
    
    # Separar features y etiquetas para entrenamiento y evaluación
    X, y = dataset[features], dataset['clase']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=7,
                                 min_samples_leaf=4,
                                 n_jobs=-1,  
                                 random_state=random_state, 
                                 class_weight='balanced')
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Extraer importancias de características y ordenarlas de mayor a menor
    importances = (
        pd.DataFrame({'Criterio': features, 'Peso_ML': clf.feature_importances_})
        .sort_values('Peso_ML', ascending=False)
    )
    return auc, importances, clf, dataset


def entrenar_modelo_rf(
    positivas: gpd.GeoDataFrame,
    pool_negativos: gpd.GeoDataFrame,
    ratio: int,
    out_shp: str,
    random_state: int = 42,
    out_model_path: str | None = None,
    n_estimators: int = 500,
    ratio_alt: int | None = None,
) -> dict:
    """
    Entrena el Random Forest con el ratio principal y, opcionalmente,
    realiza un análisis de sensibilidad entrenando también con ratio_alt.
    El modelo guardado corresponde siempre al ratio principal.
    """
    # aspect: orientación de la ladera (0-360°). En el hemisferio sur, laderas
    # orientadas al norte (~0°/360°) maximizan la captación solar.
    features = ['ghi', 'slope', 'aspect', 'dist_transmision', 'elev']

    print(f"Entrenando modelo Random Forest con Ratio 1:{ratio}...")
    auc, importances, clf, dataset = _entrenar_y_evaluar(
        positivas, pool_negativos, features, ratio, n_estimators, random_state
    )
    if clf is None:
        raise ValueError("No hay suficientes datos negativos para entrenar.")

    print(f"  Resultado Ratio 1:{ratio} -> AUC: {auc:.4f}")
    print(importances.to_string(index=False))

    # Análisis de sensibilidad con ratio alternativo
    if ratio_alt is not None and ratio_alt != ratio:
        print(f"\nAnálisis de sensibilidad con Ratio 1:{ratio_alt}...")
        auc_alt, importances_alt, _, _ = _entrenar_y_evaluar(
            positivas, pool_negativos, features, ratio_alt, n_estimators, random_state
        )
        if auc_alt is not None:
            print(f"  Resultado Ratio 1:{ratio_alt} -> AUC: {auc_alt:.4f}")
            print(importances_alt.to_string(index=False))
            print(f"\n  Diferencia AUC (ratio {ratio} vs {ratio_alt}): {abs(auc - auc_alt):.4f}")
            if abs(auc - auc_alt) < 0.02:
                print("  → Modelo robusto al cambio de ratio.")
            else:
                print("  → Revisar balance del dataset, el AUC varía con el ratio.")
        else:
            print(f"  [AVISO] No hay suficientes negativos para el ratio alternativo 1:{ratio_alt}")
    else:
        auc_alt = None

    dataset.to_file(out_shp)
    print(f"\nDataset de entrenamiento guardado en: {out_shp}")

    if out_model_path:
        joblib.dump(clf, out_model_path)
        print(f"Modelo Random Forest guardado en: {out_model_path}")

        metrics_path = os.path.splitext(out_model_path)[0] + '_metrics.json'
        metrics = {
            'ratio_principal': ratio,
            'auc': float(auc),
            'ratio_alt': ratio_alt,
            'auc_alt': float(auc_alt) if auc_alt is not None else None,
            'n_positivas': int((dataset['clase'] == 1).sum()),
            'n_negativas': int((dataset['clase'] == 0).sum()),
            'feature_importances': importances.to_dict('records'),
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Métricas guardadas en: {metrics_path}")

    return {'auc': auc, 'importances': importances, 'modelo': clf}
