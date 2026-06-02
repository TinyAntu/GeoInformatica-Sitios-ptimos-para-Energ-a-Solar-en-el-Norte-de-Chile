import json
import os
import joblib
import pandas as pd
import optuna
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import brier_score_loss


def optimizar_hiperparametros_optuna(positivas_df, pool_neg_df, features, ratio, random_state, n_trials=30):
    """
    Ejecuta un estudio de Optuna para encontrar la combinación óptima de hiperparámetros
    que minimiza la pérdida Brier Score.
    """
    print(f"\n=== INICIANDO OPTIMIZACIÓN CON OPTUNA ({n_trials} INTENTOS) ===")
    
    # 1. Preparar el dataset idéntico al flujo principal
    n_sample = min(int(len(positivas_df) * ratio), len(pool_neg_df))
    neg_sample = pool_neg_df.sample(n=n_sample, random_state=random_state)
    dataset = gpd.GeoDataFrame(
        pd.concat([positivas_df, neg_sample], ignore_index=True),
        geometry='geometry', crs='EPSG:32718',
    ).dropna(subset=features)
    
    X, y = dataset[features], dataset['clase']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    # 2. Definir la función objetivo interna
    def objective(trial):
        # Definimos el espacio de búsqueda para cada hiperparámetro
        n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
        
        # Configuramos el modelo con las sugerencias de este 'trial'
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced'
        )
        
        # Entrenar
        clf.fit(X_train, y_train)
        
        # Calcular la pérdida (Brier Score) en el set de TEST/VALIDACIÓN
        y_probs = clf.predict_proba(X_test)[:, 1]
        brier_pérdida = brier_score_loss(y_test, y_probs)
        
        # Retornamos la pérdida objetiva (Optuna intentará que sea lo más cercana a 0)
        return brier_pérdida

    # 3. Crear el estudio de Optuna
    # Buscamos 'minimize' porque el Brier Score es una métrica de pérdida (menor es mejor)
    study = optuna.create_study(direction='minimize')
    
    # Ejecutar la optimización ocultando el texto invasivo por cada intento
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== OPTIMIZACIÓN FINALIZADA ===")
    print(f"Mejor pérdida lograda (Brier Score): {study.best_value:.4f}")
    print("Mejores Hiperparámetros encontrados:")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")
        
    return study.best_params


def _entrenar_y_evaluar(positivas_df, pool_neg_df, features, ratio, n_estimators, max_depth, min_samples_leaf, random_state):
    """Entrena un RF con el ratio dado y retorna (auc, brier, importances, clf, dataset, y_test, y_probs)."""
    n_sample = min(int(len(positivas_df) * ratio), len(pool_neg_df))
    if n_sample == 0:
        return None, None, None, None, None, None, None
    neg_sample = pool_neg_df.sample(n=n_sample, random_state=random_state)
    dataset = gpd.GeoDataFrame(
        pd.concat([positivas_df, neg_sample], ignore_index=True),
        geometry='geometry', crs='EPSG:32718',
    ).dropna(subset=features)
    if len(dataset) == 0:
        return None, None, None, None, None, None, None
    
    # Separar features y etiquetas para entrenamiento y evaluación
    X, y = dataset[features], dataset['clase']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 n_jobs=-1,  
                                 random_state=random_state, 
                                 class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predicción de probabilidades para la clase 1 (sitio óptimo)
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # CÁLCULO DE MÉTRICAS
    auc = roc_auc_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)  # AGREGADO: Brier Score

    # Extraer importancias de características y ordenarlas de mayor a menor
    importances = (
        pd.DataFrame({'Criterio': features, 'Peso_ML': clf.feature_importances_})
        .sort_values('Peso_ML', ascending=False)
    )
    return auc, brier, importances, clf, dataset, y_test, y_probs


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
    Entrena el Random Forest con el ratio principal incorporando Brier Score y análisis de sensibilidad.
    """
    # Lista de características unificada
    features = ['slope', 'ghi', 'elev', 'northness', 'dist_transmision']

    # Optimizacion con OPTUNA
    mejores_params = optimizar_hiperparametros_optuna(
        positivas, pool_negativos, features, ratio, random_state, n_trials=100
    )
    
    # Extraemos los mejores valores encontrados con la optimizacion bayesiana de Optuna. 
    # Si por alguna razón no se encuentran, se mantienen los valores por defecto.
    best_n_estimators = mejores_params.get('n_estimators', n_estimators)
    best_max_depth = mejores_params.get('max_depth', 7)
    best_min_samples_leaf = mejores_params.get('min_samples_leaf', 4)
    # =========================================================================

    print(f"\nEntrenando modelo final con los mejores parámetros descubiertos...")
    
    # Modifica la función interna `_entrenar_y_evaluar` para que reciba 
    # max_depth y min_samples_leaf en lugar de tenerlos estáticos adentro (hardcoded).
    auc, brier, importances, clf, dataset, y_test_p, y_probs_p = _entrenar_y_evaluar(
        positivas, pool_negativos, features, ratio, 
        n_estimators=best_n_estimators, 
        max_depth=best_max_depth, 
        min_samples_leaf=best_min_samples_leaf, 
        random_state=random_state
    )

    if clf is None:
        raise ValueError("No hay suficientes datos negativos para entrenar.")

    print(f"  Resultado Ratio 1:{ratio} -> AUC: {auc:.4f} | Brier Score: {brier:.4f}")
    print(importances.to_string(index=False))

    if ratio_alt is not None and ratio_alt != ratio:
        print(f"\nAnálisis de sensibilidad con Ratio 1:{ratio_alt}...")
        auc_alt, brier_alt, importances_alt, _, _, y_test_a, y_probs_a = _entrenar_y_evaluar(
            positivas, pool_negativos, features, ratio_alt, n_estimators, best_max_depth, best_min_samples_leaf, random_state
        )
        if auc_alt is not None:
            print(f"  Resultado Ratio 1:{ratio_alt} -> AUC: {auc_alt:.4f} | Brier Score: {brier_alt:.4f}")
            print(importances_alt.to_string(index=False))
            print(f"\n  Diferencia AUC (ratio {ratio} vs {ratio_alt}): {abs(auc - auc_alt):.4f}")
            if abs(auc - auc_alt) < 0.02:
                print("  → Modelo robusto al cambio de ratio.")
            else:
                print("  → Revisar balance del dataset, el AUC varía con el ratio.")
        else:
            print(f"  [AVISO] No hay suficientes negativos para el ratio alternativo 1:{ratio_alt}")
            auc_alt, brier_alt, y_test_a, y_probs_a = None, None, None, None
    else:
        auc_alt, brier_alt, y_test_a, y_probs_a = None, None, None, None

    # Guardar Shapefile con renombrado seguro para evitar truncado de ESRI
    dataset_shp = dataset.copy()
    if 'dist_transmision' in dataset_shp.columns:
        dataset_shp = dataset_shp.rename(columns={'dist_transmision': 'dist_trans'})
        
    dataset_shp.to_file(out_shp)
    print(f"\nDataset de entrenamiento guardado en: {out_shp}")

    if out_model_path:
        joblib.dump(clf, out_model_path)
        print(f"Modelo Random Forest guardado en: {out_model_path}")

        metrics_path = os.path.splitext(out_model_path)[0] + '_metrics.json'
        metrics = {
            'ratio_principal': ratio,
            'auc': float(auc),
            'brier_score': float(brier),  # AGREGADO al JSON
            'ratio_alt': ratio_alt,
            'auc_alt': float(auc_alt) if auc_alt is not None else None,
            'brier_score_alt': float(brier_alt) if brier_alt is not None else None,  # AGREGADO al JSON
            'n_positivas': int((dataset['clase'] == 1).sum()),
            'n_negativas': int((dataset['clase'] == 0).sum()),
            'feature_importances': importances.to_dict('records'),
            'best_params': {
                'n_estimators': best_n_estimators,
                'max_depth': best_max_depth,
                'min_samples_leaf': best_min_samples_leaf
            }
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Métricas guardadas en: {metrics_path}")

    # Retornamos los sets de prueba y probabilidades para poder graficar externamente si se desea
    return {
        'auc': auc, 
        'brier': brier,
        'importances': importances, 
        'modelo': clf,
        'eval_data': {
            'principal': (y_test_p, y_probs_p),
            'alternativo': (y_test_a, y_probs_a)
        }
    }