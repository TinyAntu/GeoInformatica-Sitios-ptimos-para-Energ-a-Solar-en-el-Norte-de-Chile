"""
Ejecuta la validación espacial completa sobre el modelo entrenado y produce
los artefactos exigidos por PEP1: AUC por fold, GAP inter-regional,
importancias estables y figura de folds.
"""
import os
import sys
import json
import yaml
import joblib
import geopandas as gpd

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.spatial_validation import (
    asignar_region_a_muestras,
    asignar_bloques_espaciales,
    spatial_block_cv,
    leave_one_region_out_cv,
    graficar_folds_espaciales,
    guardar_reporte_validacion,
)
from src.features import FEATURES


def main():
    # --- Cargar configuración ---
    with open(os.path.join(directorio_raiz, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    paths_results = config['paths']['results']
    paths_raw     = config['paths']['raw']

    # --- Cargar dataset entrenado y modelo ---
    dataset_path = os.path.join(directorio_raiz, paths_results['dataset_ml'])
    print(f"Cargando dataset desde: {dataset_path}")
    muestras = gpd.read_file(dataset_path)

    # Restaurar nombre completo si ESRI truncó (Shapefile limita a 10 chars)
    rename_map = {
        'dist_trans': 'dist_transmision',
        'dist_almac': 'dist_almacen',
        'dist_subs':  'dist_subestaciones',
    }
    muestras = muestras.rename(columns={k: v for k, v in rename_map.items()
                                        if k in muestras.columns})

    metrics_path = os.path.splitext(
        os.path.join(directorio_raiz, paths_results['model_rf'])
    )[0] + '_metrics.json'
    with open(metrics_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)['best_params']

    params_rf = {
        'n_estimators':     best_params['n_estimators'],
        'max_depth':        best_params['max_depth'],
        'min_samples_leaf': best_params['min_samples_leaf'],
        'class_weight':     'balanced',
    }
    # Mismas features (y mismo orden) que el modelo entrenado (definidas en src/features.py)
    features = list(FEATURES)

    # Tamaño de bloque espacial centralizado en config.yaml (fallback 15 km)
    tam_km = config.get('validacion', {}).get('tamano_bloque_km', 15)

    # --- Asignar REGION + bloques ---
    print(f"\n[1/4] Asignando REGION y bloques espaciales de {tam_km:.0f} km...")
    regiones = gpd.read_file(paths_raw['vectores']['regiones']).to_crs(muestras.crs)
    regiones = regiones[regiones['REGION'].isin(['Antofagasta', 'Atacama'])]

    muestras = asignar_region_a_muestras(muestras, regiones)
    muestras = asignar_bloques_espaciales(muestras, tamano_bloque_m=tam_km * 1000)

    # --- SBCV ---
    print(f"\n[2/4] Ejecutando Spatial Block CV (k=5, bloques de {tam_km:.0f} km)...")
    sbcv = spatial_block_cv(muestras, features, params_rf, n_splits=5, tamano_bloque_km=tam_km)

    # --- LOROCV ---
    print("\n[3/4] Ejecutando Leave-One-Region-Out CV...")
    lorocv = leave_one_region_out_cv(muestras, features, params_rf)

    # --- Artefactos ---
    print("\n[4/4] Guardando artefactos...")
    figures_dir = os.path.join(directorio_raiz, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    graficar_folds_espaciales(
        muestras.loc[sbcv['indices_evaluados']],
        fold_assignment=sbcv['fold_assignment'],
        regiones_gdf=regiones,
        output_path=os.path.join(figures_dir, 'sbcv_distribucion_folds.png'),
        tamano_bloque_km=tam_km,
    )

    guardar_reporte_validacion(
        sbcv, lorocv,
        output_path=os.path.join(directorio_raiz, 'data/results/validacion_espacial.json'),
    )

    # --- Resumen ejecutivo ---
    print("\n" + "="*70)
    print("RESUMEN EJECUTIVO PARA PEP1")
    print("="*70)
    print(f"SBCV (k=5, bloques {tam_km:.0f} km):  AUC = {sbcv['auc_mean']:.4f} ± {sbcv['auc_std']:.4f}")
    print(f"LOROCV:                     AUC medio = {lorocv['auc_mean']:.4f}")
    print(f"                            GAP = {lorocv['gap_auc']:.4f}")
    print(f"                            {lorocv['interpretacion']}")
    print("="*70)


if __name__ == "__main__":
    main()