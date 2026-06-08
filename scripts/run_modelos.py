"""
Estudio de ablación: comparación de tres configuraciones de features para
diagnosticar la multicolinealidad dist_transmision ↔ dist_subestaciones (ρ=0,69).

Reporta AUC SBCV de cada configuración. Si la caída entre 'completo' y
'sin_subest' (o 'sin_trans') es < 0,01 AUC, las variables son redundantes
y debe eliminarse una. Si la caída es > 0,02, son complementarias.

Uso:
    python scripts/run_ablation_study.py
"""
import os
import sys
import json
import yaml
import geopandas as gpd
import pandas as pd

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.spatial_validation import (
    asignar_region_a_muestras,
    asignar_bloques_espaciales,
    spatial_block_cv,
)


# Configuraciones de features a comparar
MODELOS_ABLACION = {
    'completo': [
        'slope', 'ghi', 'elev', 'northness',
        'dist_transmision', 'dist_almacen', 'dist_subestaciones'
    ],
    'sin_subest': [
        'slope', 'ghi', 'elev', 'northness',
        'dist_transmision', 'dist_almacen'
    ],
    'sin_trans': [
        'slope', 'ghi', 'elev', 'northness',
        'dist_almacen', 'dist_subestaciones'
    ],
    'minimo': ['slope','ghi','elev','northness','dist_subestaciones']
}


def main():
    # --- Carga de configuración y dataset ya generado ---
    with open(os.path.join(directorio_raiz, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    paths_results = config['paths']['results']
    paths_raw     = config['paths']['raw']

    # Reutilizar el dataset entrenado del flujo principal
    dataset_path = os.path.join(directorio_raiz, paths_results['dataset_ml'])
    print(f"Cargando dataset desde: {dataset_path}")
    muestras = gpd.read_file(dataset_path)

    # Restaurar nombres truncados por ESRI Shapefile
    rename_map = {
        'dist_trans': 'dist_transmision',
        'dist_almac': 'dist_almacen',
        'dist_subs':  'dist_subestaciones',
    }
    muestras = muestras.rename(columns={k: v for k, v in rename_map.items()
                                        if k in muestras.columns})

    # Hiperparámetros ya optimizados del flujo principal
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
    print(f"Usando hiperparámetros: {params_rf}")

    # --- Preparar estructura espacial (REGION + bloques) ---
    tam_km = config.get('validacion', {}).get('tamano_bloque_km', 15)
    regiones = gpd.read_file(paths_raw['vectores']['regiones']).to_crs(muestras.crs)
    regiones = regiones[regiones['REGION'].isin(['Antofagasta', 'Atacama'])]
    muestras = asignar_region_a_muestras(muestras, regiones)
    muestras = asignar_bloques_espaciales(muestras, tamano_bloque_m=tam_km * 1000)

    # --- Ejecutar SBCV para cada configuración ---
    resultados = {}
    for nombre, features in MODELOS_ABLACION.items():
        print(f"\n{'='*70}")
        print(f"ABLACIÓN: configuración '{nombre}' ({len(features)} features)")
        print(f"  Features: {features}")
        print('='*70)

        sbcv = spatial_block_cv(
            muestras, features, params_rf,
            n_splits=5, random_state=42, tamano_bloque_km=tam_km,
        )
        resultados[nombre] = {
            'features':    features,
            'n_features':  len(features),
            'auc_mean':    sbcv['auc_mean'],
            'auc_std':     sbcv['auc_std'],
            'brier_mean':  sbcv['brier_mean'],
            'brier_std':   sbcv['brier_std'],
            'importancias': sbcv['importancias'],
        }

    # --- Análisis comparativo ---
    print('\n' + '='*70)
    print('RESUMEN COMPARATIVO DEL ESTUDIO DE ABLACIÓN')
    print('='*70)

    auc_completo = resultados['completo']['auc_mean']
    tabla = []
    for nombre, r in resultados.items():
        delta = r['auc_mean'] - auc_completo
        tabla.append({
            'Configuración':   nombre,
            'N features':      r['n_features'],
            'AUC SBCV':        f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
            'Δ vs completo':   f"{delta:+.4f}",
            'Brier':           f"{r['brier_mean']:.4f}",
        })
    df_tabla = pd.DataFrame(tabla)
    print(df_tabla.to_string(index=False))

    # --- Diagnóstico interpretativo ---
    print('\n' + '='*70)
    print('DIAGNÓSTICO ACADÉMICO')
    print('='*70)

    delta_subest = resultados['sin_subest']['auc_mean'] - auc_completo
    delta_trans  = resultados['sin_trans']['auc_mean']  - auc_completo

    print(f"\nCaída al eliminar dist_subestaciones: {delta_subest:+.4f}")
    print(f"Caída al eliminar dist_transmision:   {delta_trans:+.4f}")

    # --- Guardar reporte ---
    output_path = os.path.join(directorio_raiz, 'data/results/ablation_study.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'resultados': resultados,
            'delta_sin_subest': delta_subest,
            'delta_sin_trans':  delta_trans,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en: {output_path}")


if __name__ == "__main__":
    main()