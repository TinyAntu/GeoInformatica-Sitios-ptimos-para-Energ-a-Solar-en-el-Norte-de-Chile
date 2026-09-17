"""Evaluación comparativa del Área de Aplicabilidad

Evalúa el Índice de Disimilitud (DI) y la fracción de territorio dentro del AOA en tres
escenarios espaciales:
  1) In-sample: modelo completo evaluado en Antofagasta + Atacama (referencia).
  2) LORO (Leave-One-Region-Out): entrenamiento en una región y evaluación de AOA en la otra.
  3) Transferencia: modelo del norte evaluado en Coquimbo (fuera de dominio).

Uso:
    python scripts/run_aoa_lorocv.py --config config.yaml
"""

import os
import sys
import json
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.features import FEATURES
from src.aoa import extraer_pesos_modelo, ajustar_espacio_aoa, evaluar_aoa_zona
from src.explainability_spatial import _construir_features, zona_de_config
from src.metricas_topk import cargar_plantas, _plantas_a_filas_columnas
from src.transferibilidad import cargar_muestras_entrenamiento, RENAME_ESRI
from src.utils import _resolver_rutas, _ruta_abs


def evaluar_aoa_en_grilla(config, modelo, zona, X_train, out_dir=None):
    """Evalúa AOA sobre la grilla raster de una zona dada."""
    resolucion_m = zona.get('resolucion_m', 500)
    crs, transform, width, height, valido, X_pred = _construir_features(
        config, resolucion_m=resolucion_m, zona=zona
    )
    grid_shape = (height, width)
    base_meta = {
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height,
    }

    # Cargar plantas de la zona para saber cuántas caen en AOA
    try:
        plantas = cargar_plantas(config, directorio_raiz, crs, regiones_estudio=zona['regiones'])
        plantas_rc = _plantas_a_filas_columnas(plantas, transform, grid_shape)
    except Exception:
        plantas_rc = None

    pesos = extraer_pesos_modelo(modelo, FEATURES)
    ajuste = ajustar_espacio_aoa(X_train, pesos)

    res = evaluar_aoa_zona(
        ajuste_aoa=ajuste,
        X_zona=X_pred,
        valido=valido,
        base_meta=base_meta,
        dir_salida=out_dir,
        plantas_rc=plantas_rc,
    )
    res['resolucion_m'] = resolucion_m
    res['regiones'] = zona['regiones']
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out-json', default=None,
                        help='Ruta de salida (por defecto data/results/aoa_comparacion.json)')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])

    out_json = _ruta_abs(args.out_json) if args.out_json else os.path.join(
        directorio_raiz, 'data', 'results', 'aoa_comparacion.json'
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    print("=" * 75)
    print("EVALUACIÓN MULTI-ZONA DE ÁREA DE APLICABILIDAD")
    print("=" * 75)

    # 1. Cargar muestras y modelo entrenado
    muestras = cargar_muestras_entrenamiento(config, directorio_raiz)
    regiones_gdf = gpd.read_file(_ruta_abs(config['paths']['raw']['vectores']['regiones'])).to_crs(muestras.crs)
    regiones_norte = regiones_gdf[regiones_gdf['REGION'].isin(['Antofagasta', 'Atacama'])]
    from src.spatial_validation import asignar_region_a_muestras
    muestras = asignar_region_a_muestras(muestras, regiones_norte)

    modelo_completo = joblib.load(_ruta_abs(config['paths']['results']['model_rf']))
    X_train_full = muestras[FEATURES].dropna().values

    resultados = {}

    # Escenario 1: In-sample (Antofagasta + Atacama)
    print("\n[1/3] Evaluando AOA In-sample (Antofagasta + Atacama)...")
    zona_norte = zona_de_config(config, 'zona_estudio')
    zona_norte['resolucion_m'] = 500  # Homogéneo para comparación
    dir_norte = os.path.join(directorio_raiz, 'data', 'results')
    res_in_sample = evaluar_aoa_en_grilla(config, modelo_completo, zona_norte, X_train_full, out_dir=None)
    resultados['in_sample_norte'] = res_in_sample
    print(f"  -> Dentro de AOA: {res_in_sample['pct_dentro_aoa']:.1f}% | Fuera: {res_in_sample['pct_fuera_aoa']:.1f}% | DI mediano: {res_in_sample['estadisticas_di_zona']['mediana']:.3f}")

    # Escenario 2: LORO (Antofagasta vs Atacama)
    print("\n[2/3] Evaluando AOA en LORO (Leave-One-Region-Out)...")
    resultados_loro = {}
    ml_params = config.get('ml_params', {})
    params_rf = {
        'n_estimators': ml_params.get('n_estimators', 500),
        'max_depth': 8,
        'min_samples_leaf': 3,
        'class_weight': 'balanced',
        'random_state': 42,
    }

    for region_test in ['Antofagasta', 'Atacama']:
        region_train = 'Atacama' if region_test == 'Antofagasta' else 'Antofagasta'
        print(f"  Entrenando en {region_train} -> Evaluando AOA en {region_test}...")

        muestras_tr = muestras[muestras['REGION'] == region_train].dropna(subset=FEATURES)
        if len(muestras_tr) == 0 or muestras_tr['clase'].nunique() < 2:
            print(f"  [AVISO] No hay suficientes muestras para LORO en {region_test}.")
            continue

        clf_loro = RandomForestClassifier(**params_rf, n_jobs=-1)
        clf_loro.fit(muestras_tr[FEATURES], muestras_tr['clase'])
        X_tr_loro = muestras_tr[FEATURES].values

        codigos_antofagasta = [2, '2', '02', 'II', 'Antofagasta', 'ANTOFAGASTA']
        codigos_atacama = [3, '3', '03', 'III', 'Atacama', 'ATACAMA']
        zona_loro = {
            'regiones': [region_test],
            'dem': config['paths']['processed']['dem_32719'],
            'slope': config['paths']['processed']['slope'],
            'aspect': config['paths']['processed']['aspect'],
            'resolucion_m': 500,
            'modo_region': 'exacto',
            'codigos': codigos_antofagasta if region_test == 'Antofagasta' else codigos_atacama,
        }
        res_loro = evaluar_aoa_en_grilla(config, clf_loro, zona_loro, X_tr_loro, out_dir=None)
        resultados_loro[f'entrenado_{region_train.lower()}_eval_{region_test.lower()}'] = res_loro
        print(f"    -> En {region_test}: {res_loro['pct_dentro_aoa']:.1f}% dentro de AOA | DI mediano: {res_loro['estadisticas_di_zona']['mediana']:.3f}")

    resultados['loro_cv'] = resultados_loro

    # Escenario 3: Transferencia a Coquimbo (fuera de dominio)
    if 'transferibilidad' in config:
        print("\n[3/3] Evaluando AOA en Transferencia a Coquimbo...")
        bloque_transf = config['transferibilidad']
        dem_transf_path = _ruta_abs(bloque_transf['dem'])
        if not os.path.exists(dem_transf_path):
            from scripts.run_zona import preparar_dem
            print("  Procesando DEM/slope/aspect de Coquimbo...")
            preparar_dem(config, bloque_transf)

        zona_coquimbo = zona_de_config(config, 'transferibilidad')
        zona_coquimbo['resolucion_m'] = bloque_transf.get('resolucion_m', 500)
        dir_coquimbo = _ruta_abs(bloque_transf.get('dir_resultados', 'data/results/coquimbo'))
        res_coquimbo = evaluar_aoa_en_grilla(config, modelo_completo, zona_coquimbo, X_train_full, out_dir=dir_coquimbo)
        resultados['transferencia_coquimbo'] = res_coquimbo
        print(f"  -> Coquimbo dentro de AOA: {res_coquimbo['pct_dentro_aoa']:.1f}% | Fuera de AOA: {res_coquimbo['pct_fuera_aoa']:.1f}% | DI mediano: {res_coquimbo['estadisticas_di_zona']['mediana']:.3f}")
        if res_coquimbo.get('plantas_en_aoa'):
            pl = res_coquimbo['plantas_en_aoa']
            print(f"  -> Plantas de Coquimbo en AOA: {pl['pct_plantas_dentro_aoa']:.1f}% ({pl['n_plantas_dentro_aoa']}/{pl['n_plantas_evaluadas']})")

    # Guardar reporte JSON estructurado
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"[OK] Reporte comparativo de AOA guardado en: {out_json}")
    print("=" * 75)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
