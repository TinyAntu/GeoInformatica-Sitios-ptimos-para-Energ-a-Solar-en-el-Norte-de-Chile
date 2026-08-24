import json
import os
import joblib
import pandas as pd
import optuna
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo: solo guardamos figuras, evita el crash de Tkinter en hilos
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, average_precision_score, roc_curve, auc
from src.spatial_validation import asignar_region_a_muestras, asignar_bloques_espaciales, make_objective_spatial, spatial_block_cv
from src.features import FEATURES


def optimizar_hiperparametros_optuna(positivas_df, pool_neg_df, features, ratio,
                                     random_state, optuna_config,
                                     regiones_gdf=None, tamano_bloque_km=15):
    n_trials = optuna_config.get('n_trials', 30)
    print(f"\n=== OPTIMIZACIÓN OPTUNA SOBRE SBCV ({n_trials} TRIALS) ===")

    # 1. Construir el dataset idéntico al flujo principal
    n_sample = min(int(len(positivas_df) * ratio), len(pool_neg_df))
    neg_sample = pool_neg_df.sample(n=n_sample, random_state=random_state)
    dataset = gpd.GeoDataFrame(
        pd.concat([positivas_df, neg_sample], ignore_index=True),
        geometry='geometry', crs='EPSG:32719',
    ).dropna(subset=features)

    # 2. Asignar REGION y bloques espaciales (la grilla CV consciente)
    if regiones_gdf is not None:
        dataset = asignar_region_a_muestras(dataset, regiones_gdf)
    dataset = asignar_bloques_espaciales(dataset, tamano_bloque_m=tamano_bloque_km * 1000)

    # 3. Objective basado en SBCV 
    objective = make_objective_spatial(dataset, features, optuna_config, random_state)

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nMejor Brier promedio (5-fold espacial): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  -> {k}: {v}")
    return study.best_params


def _entrenar_y_evaluar(positivas_df, pool_neg_df, features, ratio, n_estimators, max_depth, min_samples_leaf, random_state):
    """Entrena un RF con el ratio dado y retorna (auc, brier, importances, clf, dataset, y_test, y_probs)."""
    n_sample = min(int(len(positivas_df) * ratio), len(pool_neg_df))
    if n_sample == 0:
        return None, None, None, None, None, None, None
    neg_sample = pool_neg_df.sample(n=n_sample, random_state=random_state)
    dataset = gpd.GeoDataFrame(
        pd.concat([positivas_df, neg_sample], ignore_index=True),
        geometry='geometry', crs='EPSG:32719',
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

    # CÁLCULO DE MÉTRICAS (sobre el test retenido, no sobre datos vistos)
    auc = roc_auc_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)  # AGREGADO: Brier Score

    # Reentrenar el modelo final sobre el 100% de los datos para despliegue:
    # las métricas de arriba ya salieron del test retenido, así que el modelo que
    # genera el mapa debe aprovechar todas las muestras disponibles.
    clf.fit(X, y)

    # Extraer importancias de características y ordenarlas de mayor a menor
    importances = (
        pd.DataFrame({'Criterio': features, 'Peso_ML': clf.feature_importances_})
        .sort_values('Peso_ML', ascending=False)
    )
    return auc, brier, importances, clf, dataset, y_test, y_probs


def _figures_dir() -> str:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    figures_dir = os.path.join(root_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _guardar_importancias(importances: pd.DataFrame, filename: str) -> str:
    figures_dir = _figures_dir()
    output_path = os.path.join(figures_dir, filename)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances['Criterio'], importances['Peso_ML'], color='#2a7f62')
    ax.invert_yaxis()
    ax.set_xlabel('Importancia de Característica')
    ax.set_title('Importancias de Características del Random Forest')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _guardar_curva_precision_recall(y_test, y_probs, filename: str) -> str:
    figures_dir = _figures_dir()
    output_path = os.path.join(figures_dir, filename)

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ap_score = average_precision_score(y_test, y_probs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color='#d35400', lw=2)
    ax.set_xlabel('Recall (Sensibilidad)')
    ax.set_ylabel('Precision')
    ax.set_title(f'Curva Precision-Recall (AP = {ap_score:.4f})')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _guardar_curva_roc(y_test, y_probs, filename: str) -> str:
    figures_dir = _figures_dir()
    output_path = os.path.join(figures_dir, filename)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, color='#1f78b4', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def entrenar_modelo_rf(
    positivas: gpd.GeoDataFrame,
    pool_negativos: gpd.GeoDataFrame,
    ratio: int,
    out_shp: str,
    optuna_config: dict,
    random_state: int = 42,
    out_model_path: str | None = None,
    n_estimators: int = 500,
    ratio_alt: int | None = None,
    tamano_bloque_km: float = 15,
    regiones_gdf: gpd.GeoDataFrame | None = None,
) -> dict:
    """
    Entrena el Random Forest con el ratio principal incorporando Brier Score y análisis de sensibilidad.
    """
    # Lista de características canónica (definida una sola vez en src/features.py)
    features = list(FEATURES)

    # Mostrar matriz de correlación antes de la optimización para entender relaciones entre características
    muestras = pd.concat([positivas, pool_negativos.sample(n=min(int(len(positivas) * ratio), len(pool_negativos)), random_state=random_state)], ignore_index=True)
    corr = muestras[features].corr(method='spearman')
    print("\nMatriz de Correlación de Características (Spearman):")
    print(corr.to_string(float_format=lambda x: f"{x:.2f}"))

    # Optimizacion con OPTUNA
    mejores_params = optimizar_hiperparametros_optuna(
        positivas, pool_negativos, features, ratio, random_state, optuna_config,
        regiones_gdf=regiones_gdf, tamano_bloque_km=tamano_bloque_km,
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

    # --- MÉTRICA DE RECORD: Spatial Block CV (no el split aleatorio) ---
    # El AUC/Brier de arriba viene de un train_test_split ALEATORIO y por lo tanto
    # está inflado por autocorrelación espacial (Roberts et al. 2017; Ploton et al. 2020).
    # Reportamos SBCV como la métrica honesta y la guardamos como principal.
    #
    # Asignar REGION antes de armar los bloques: sin esto, `asignar_bloques_espaciales`
    # cae en el fallback sin columna REGION y los bloques NO quedan inter-regionalmente
    # disjuntos pese a que ese es justamente el propósito del bloque (ver docstring de
    # src/spatial_validation.py). Con REGION, el bloque incorpora el prefijo de región y
    # fuerza la disjunción que el diseño original pedía.
    dataset_cv_base = (
        asignar_region_a_muestras(dataset.copy(), regiones_gdf)
        if regiones_gdf is not None else dataset.copy()
    )
    dataset_cv = asignar_bloques_espaciales(dataset_cv_base, tamano_bloque_m=tamano_bloque_km * 1000)
    params_rf_sbcv = {
        'n_estimators':     best_n_estimators,
        'max_depth':        best_max_depth,
        'min_samples_leaf': best_min_samples_leaf,
        'class_weight':     'balanced',
    }
    sbcv = spatial_block_cv(
        dataset_cv, features, params_rf_sbcv,
        n_splits=5, random_state=random_state, tamano_bloque_km=tamano_bloque_km,
    )
    auc_sbcv, brier_sbcv = sbcv['auc_mean'], sbcv['brier_mean']
    print(f"  [MÉTRICA DE RECORD] SBCV AUC = {auc_sbcv:.4f} ± {sbcv['auc_std']:.4f} | "
          f"Brier = {brier_sbcv:.4f}  (el AUC del split aleatorio es referencia optimista)")

    feature_path = _guardar_importancias(importances, 'feature_importances_rf.png')
    pr_path = _guardar_curva_precision_recall(y_test_p, y_probs_p, f'precision_recall_ratio_{ratio}.png')
    roc_path = _guardar_curva_roc(y_test_p, y_probs_p, f'roc_curve_ratio_{ratio}.png')
    print(f"\nGráficos guardados en: {feature_path}, {pr_path}, {roc_path}")

    if ratio_alt is not None and ratio_alt != ratio:
        print(f"\nAnálisis de sensibilidad con Ratio 1:{ratio_alt}...")
        auc_alt, brier_alt, importances_alt, _, _, y_test_a, y_probs_a = _entrenar_y_evaluar(
            positivas, pool_negativos, features, ratio_alt, n_estimators, best_max_depth, best_min_samples_leaf, random_state
        )
        if auc_alt is not None:
            print(f"  Resultado Ratio 1:{ratio_alt} -> AUC: {auc_alt:.4f} | Brier Score: {brier_alt:.4f}")
            print(importances_alt.to_string(index=False))
            pr_alt_path = _guardar_curva_precision_recall(y_test_a, y_probs_a, f'precision_recall_ratio_{ratio_alt}.png')
            roc_alt_path = _guardar_curva_roc(y_test_a, y_probs_a, f'roc_curve_ratio_{ratio_alt}.png')
            print(f"  Gráficos alternativos guardados en: {pr_alt_path}, {roc_alt_path}")
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
    shp_rename = {}
    if 'dist_transmision' in dataset_shp.columns:
        shp_rename['dist_transmision'] = 'dist_trans'
    if 'dist_almacen' in dataset_shp.columns:
        shp_rename['dist_almacen'] = 'dist_almac'
    if 'dist_subestaciones' in dataset_shp.columns:
        shp_rename['dist_subestaciones'] = 'dist_subs'
    if shp_rename:
        dataset_shp = dataset_shp.rename(columns=shp_rename)
        
    dataset_shp.to_file(out_shp)
    print(f"\nDataset de entrenamiento guardado en: {out_shp}")

    if out_model_path:
        joblib.dump(clf, out_model_path)
        print(f"Modelo Random Forest guardado en: {out_model_path}")

        metrics_path = os.path.splitext(out_model_path)[0] + '_metrics.json'
        metrics = {
            'ratio_principal': ratio,
            # Métrica de record (espacialmente consciente):
            'auc_sbcv': float(auc_sbcv),
            'brier_sbcv': float(brier_sbcv),
            # Métricas del split aleatorio: OPTIMISTAS por autocorrelación espacial.
            'auc_split_aleatorio': float(auc),
            'auc': float(auc),  # se conserva por compatibilidad con scripts existentes
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