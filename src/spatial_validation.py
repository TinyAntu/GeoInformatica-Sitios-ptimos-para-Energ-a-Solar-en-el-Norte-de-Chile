"""
Validación cruzada espacialmente consciente para el modelo de aptitud solar.

Implementa las dos estrategias declaradas en PEP1:
  1) Spatial Block Cross-Validation (SBCV): k=5 folds sobre bloques de 15 km
     que controlan la autocorrelación espacial (Roberts et al., 2017).
  2) Leave-One-Region-Out CV (LOROCV): entrenamiento en una región y validación
     en la otra; cuantifica la transferibilidad inter-regional (Ploton et al., 2020).
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo: solo guardamos figuras, evita el crash de Tkinter en hilos
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss


# ===========================================================================
# 1. ASIGNACIÓN DE REGIÓN Y BLOQUES A LAS MUESTRAS
# ===========================================================================

def asignar_region_a_muestras(
    muestras: gpd.GeoDataFrame,
    regiones_gdf: gpd.GeoDataFrame,
    col_region: str = 'REGION',
) -> gpd.GeoDataFrame:
    """
    Spatial join: asigna a cada muestra (positiva o negativa) su atributo REGION
    a partir del polígono regional. Indispensable para LOROCV.
    """
    if muestras.crs != regiones_gdf.crs:
        regiones_gdf = regiones_gdf.to_crs(muestras.crs)

    join = gpd.sjoin(
        muestras,
        regiones_gdf[[col_region, 'geometry']],
        how='left',
        predicate='within',
    )
    join = join.drop(
        columns=[c for c in join.columns if c.startswith('index_')],
        errors='ignore',
    )
    # Si una muestra cae sobre frontera, sjoin puede duplicar
    join = join.loc[~join.index.duplicated(keep='first')]

    n_sin = join[col_region].isna().sum()
    if n_sin > 0:
        print(f"  [AVISO] {n_sin} muestras sin REGION (frontera o fuera); se descartan.")
        join = join.dropna(subset=[col_region])
    return join.reset_index(drop=True)


def asignar_bloques_espaciales(
    muestras: gpd.GeoDataFrame,
    tamano_bloque_m: float = 15000,
    col_region: str = 'REGION',
) -> gpd.GeoDataFrame:
    """
    Asigna a cada muestra el ID del bloque cuadrado al que pertenece.
    El bloque incorpora REGION para forzar disjunción inter-regional.
    
    Requiere que la geometría esté en CRS métrico (UTM 19S, EPSG:32719).
    """
    muestras = muestras.copy()
    bx = (muestras.geometry.x // tamano_bloque_m).astype(int)
    by = (muestras.geometry.y // tamano_bloque_m).astype(int)

    if col_region in muestras.columns:
        muestras['block_id'] = (
            muestras[col_region].astype(str) + '_'
            + bx.astype(str) + '_' + by.astype(str)
        )
    else:
        muestras['block_id'] = bx.astype(str) + '_' + by.astype(str)

    n_bloques = muestras['block_id'].nunique()
    print(f"  Bloques de {tamano_bloque_m/1000:.0f} km: {n_bloques} únicos para "
          f"{len(muestras)} muestras  (densidad media: "
          f"{len(muestras)/n_bloques:.1f} muestras/bloque)")
    return muestras


# ===========================================================================
# 2. SPATIAL BLOCK CROSS-VALIDATION (SBCV) — Roberts et al. 2017
# ===========================================================================

def spatial_block_cv(
    muestras: gpd.GeoDataFrame,
    features: list[str],
    params_rf: dict,
    n_splits: int = 5,
    random_state: int = 42,
    tamano_bloque_km: float = 15,
) -> dict:
    """
    k-fold CV agrupado por bloques espaciales con estratificación de clase.

    Usa StratifiedGroupKFold (sklearn ≥1.0): preserva simultáneamente
      - el balance de clases por fold (estratificación)
      - la disjunción de bloques entre train y test (sin fugas espaciales)

    Requiere las columnas: 'block_id', 'clase' y todas las de `features`.
    """
    df = muestras.dropna(subset=features + ['clase', 'block_id']).copy()
    X = df[features].values
    y = df['clase'].astype(int).values
    groups = df['block_id'].values

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    aucs, briers, imps = [], [], []
    fold_assign = np.full(len(df), -1, dtype=int)

    for k, (tr, te) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        clf = RandomForestClassifier(**params_rf, random_state=random_state, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]

        auc_k = roc_auc_score(y[te], probs)
        brier_k = brier_score_loss(y[te], probs)
        aucs.append(auc_k); briers.append(brier_k)
        imps.append(clf.feature_importances_)
        fold_assign[te] = k

        n_b_tr = pd.unique(groups[tr]).size
        n_b_te = pd.unique(groups[te]).size
        n_pos_te = int(y[te].sum())
        print(f"  Fold {k}: train={len(tr)} ({n_b_tr} bloques) | "
              f"test={len(te)} ({n_b_te} bloques, {n_pos_te} positivos) | "
              f"AUC={auc_k:.4f} | Brier={brier_k:.4f}")

    aucs = np.array(aucs); briers = np.array(briers); imps = np.array(imps)
    importancias = pd.DataFrame({
        'feature': features,
        'importance_mean': imps.mean(axis=0),
        'importance_std':  imps.std(axis=0),
    }).sort_values('importance_mean', ascending=False)

    print(f"\n  >>> SBCV  AUC = {aucs.mean():.4f} ± {aucs.std():.4f}  |  "
          f"Brier = {briers.mean():.4f} ± {briers.std():.4f}")

    return {
        'estrategia': 'spatial_block_cv',
        'n_splits': n_splits,
        'tamano_bloque_km': tamano_bloque_km,
        'auc_por_fold':   aucs.tolist(),
        'brier_por_fold': briers.tolist(),
        'auc_mean':  float(aucs.mean()),
        'auc_std':   float(aucs.std()),
        'brier_mean': float(briers.mean()),
        'brier_std':  float(briers.std()),
        'importancias': importancias.to_dict('records'),
        'fold_assignment': fold_assign.tolist(),
        'indices_evaluados': df.index.tolist(),
    }


# ===========================================================================
# 3. LEAVE-ONE-REGION-OUT CV (LOROCV) — Ploton et al. 2020
# ===========================================================================

def leave_one_region_out_cv(
    muestras: gpd.GeoDataFrame,
    features: list[str],
    params_rf: dict,
    regiones: tuple = ('Antofagasta', 'Atacama'),
    random_state: int = 42,
) -> dict:
    """
    Para cada región: entrena con la complementaria, valida con ella misma.
    Reporta AUC en cada dirección y el GAP entre ambas.

    Interpretación operativa del GAP (Ploton et al., 2020):
      - GAP < 0,05  → fenómeno homogéneo, modelo generalizable inter-regional.
      - GAP ≥ 0,05  → regímenes regionales distintos; la transferencia directa
                      entre regiones está comprometida.
    """
    df = muestras.dropna(subset=features + ['clase', 'REGION']).copy()
    folds = {}

    for region_test in regiones:
        test = df[df['REGION'] == region_test]
        train = df[df['REGION'] != region_test]

        if test['clase'].nunique() < 2 or train['clase'].nunique() < 2:
            print(f"  [AVISO] Fold {region_test}: partición sin ambas clases; se omite.")
            continue

        X_tr, y_tr = train[features].values, train['clase'].astype(int).values
        X_te, y_te = test[features].values,  test['clase'].astype(int).values

        clf = RandomForestClassifier(**params_rf, random_state=random_state, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        folds[region_test] = {
            'region_train': [r for r in regiones if r != region_test],
            'region_test':  region_test,
            'n_train':      len(train),
            'n_test':       len(test),
            'n_pos_train':  int((y_tr == 1).sum()),
            'n_pos_test':   int((y_te == 1).sum()),
            'auc':   float(roc_auc_score(y_te, probs)),
            'brier': float(brier_score_loss(y_te, probs)),
            'importancias': pd.DataFrame({
                'feature': features,
                'importance': clf.feature_importances_,
            }).sort_values('importance', ascending=False).to_dict('records'),
        }
        print(f"  Train={folds[region_test]['region_train']} -> Test={region_test} | "
              f"AUC={folds[region_test]['auc']:.4f} | "
              f"Brier={folds[region_test]['brier']:.4f}")

    aucs = [v['auc'] for v in folds.values()]
    gap = float(max(aucs) - min(aucs)) if len(aucs) >= 2 else None
    if gap is None:
        interp = "No evaluable (folds insuficientes)."
    elif gap < 0.05:
        interp = "Fenómeno geográficamente homogéneo; modelo generalizable inter-regional."
    elif gap < 0.10:
        interp = "Heterogeneidad regional moderada; generalización con reservas."
    else:
        interp = "Regímenes regionales distintos; generalización inter-regional comprometida."

    print(f"\n  >>> LOROCV  AUC medio = {np.mean(aucs):.4f}  |  "
          f"GAP = {gap:.4f}  |  {interp}")

    return {
        'estrategia': 'leave_one_region_out_cv',
        'folds': folds,
        'auc_mean': float(np.mean(aucs)) if aucs else None,
        'gap_auc':  gap,
        'interpretacion': interp,
    }


# ===========================================================================
# 4. INTEGRACIÓN CON OPTUNA — sustituye el train_test_split actual
# ===========================================================================

def make_objective_spatial(muestras: gpd.GeoDataFrame,
                            features: list[str],
                            optuna_config: dict,
                            random_state: int = 42):
    """
    Devuelve una función objective(trial) para Optuna que minimiza el
    Brier Score PROMEDIO sobre SBCV (k=5, bloques de 15 km).
    
    Esto elimina el test-set leakage de la versión original, que evaluaba
    Optuna y métricas finales sobre el mismo X_test.
    """
    df = muestras.dropna(subset=features + ['clase', 'block_id']).copy()
    X = df[features].values
    y = df['clase'].astype(int).values
    groups = df['block_id'].values

    def objective(trial):
        cfg_n = optuna_config.get('n_estimators', {'min': 100, 'max': 1000, 'step': 100})
        cfg_d = optuna_config.get('max_depth',    {'min': 3,   'max': 15})
        cfg_l = optuna_config.get('min_samples_leaf', {'min': 2, 'max': 10})

        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators', cfg_n['min'], cfg_n['max'], step=cfg_n.get('step', 1)),
            'max_depth': trial.suggest_int('max_depth', cfg_d['min'], cfg_d['max']),
            'min_samples_leaf': trial.suggest_int(
                'min_samples_leaf', cfg_l['min'], cfg_l['max']),
            'class_weight': 'balanced',
        }

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        briers = []
        for tr, te in sgkf.split(X, y, groups=groups):
            clf = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            probs = clf.predict_proba(X[te])[:, 1]
            briers.append(brier_score_loss(y[te], probs))
        return float(np.mean(briers))

    return objective


# ===========================================================================
# 5. VISUALIZACIÓN DE FOLDS Y REPORTE
# ===========================================================================

def graficar_folds_espaciales(
    muestras: gpd.GeoDataFrame,
    fold_assignment: list | np.ndarray,
    regiones_gdf: gpd.GeoDataFrame,
    output_path: str,
    tamano_bloque_km: float = 30,
) -> str:
    """Mapa de la distribución espacial de los folds SBCV."""
    df = muestras.copy().reset_index(drop=True)
    df['fold'] = fold_assignment

    if regiones_gdf.crs != df.crs:
        regiones_gdf = regiones_gdf.to_crs(df.crs)

    fig, ax = plt.subplots(figsize=(10, 12))
    regiones_gdf.boundary.plot(ax=ax, color='black', linewidth=0.6)

    cmap = plt.get_cmap('tab10')
    for k in sorted(df['fold'].unique()):
        if k < 0:
            continue
        sub = df[df['fold'] == k]
        ax.scatter(sub.geometry.x, sub.geometry.y,
                   s=14, color=cmap(int(k) % 10),
                   label=f'Fold {int(k)}', alpha=0.75,
                   edgecolor='white', linewidth=0.3)

    ax.set_title(f'Distribución espacial de muestras por fold '
                 f'(SBCV, bloques de {tamano_bloque_km:.0f} km)')
    ax.set_xlabel('Easting (m, UTM 19S)')
    ax.set_ylabel('Northing (m, UTM 19S)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_aspect('equal')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figura guardada en: {output_path}")
    return output_path


def guardar_reporte_validacion(sbcv: dict, lorocv: dict, output_path: str) -> str:
    """Reporte JSON unificado de ambas estrategias para incluir en el PEP1."""
    reporte = {
        'spatial_block_cv': {
            k: v for k, v in sbcv.items()
            if k not in ('fold_assignment', 'indices_evaluados')
        },
        'leave_one_region_out_cv': lorocv,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"  Reporte JSON guardado en: {output_path}")
    return output_path