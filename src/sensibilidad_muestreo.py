"""Análisis de sensibilidad al diseño de pseudo-ausencias en el modelo de aptitud solar.

Evalúa cómo condiciona la selección de negativos lo que el Random Forest aprende,
contrastando 3 diseños de muestreo:
  1) 'ahp_filtrado': buffer 5 km + exclusiones territoriales + filtros técnicos AHP.
  2) 'fondo_aleatorio': buffer 5 km + exclusiones territoriales (sin filtros técnicos).
  3) 'fondo_objetivo': buffer 5 km + condicionado a cercanía de red (sin filtros geofísicos).

Calcula la estabilidad de métricas (SBCV 5-fold sobre bloques espaciales), la jerarquía
SHAP de variables, la correlación de rangos (Kendall's tau y Spearman rho) y la dominancia
de macro-familias ('acceso a red' vs 'recurso / topografía').
"""

import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import kendalltau, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap

from src.features import FEATURES
from src.sampling import generar_dataset_muestras
from src.spatial_validation import (
    asignar_region_a_muestras,
    asignar_bloques_espaciales,
    spatial_block_cv,
)

# Macro-familias de variables para análisis agrupado
FAMILIA_RED = ['dist_transmision', 'dist_subestaciones', 'dist_almacen']
FAMILIA_RECURSO = ['ghi', 'slope', 'northness', 'elev']

ESTRATEGIAS_NOMBRES = {
    'ahp_filtrado': 'AHP Filtrado (Línea Base)',
    'fondo_aleatorio': 'Fondo Aleatorio (Sin Filtros)',
    'fondo_objetivo': 'Fondo Objetivo (Red Relajada)',
}


def _calcular_estadisticas_distribucion(df: pd.DataFrame, features: list[str]) -> dict:
    """Calcula estadísticos descriptivos (cuartiles, media, std) de las variables."""
    stats = {}
    for feat in features:
        serie = df[feat].dropna().astype(float)
        if len(serie) == 0:
            continue
        stats[feat] = {
            'min': round(float(serie.min()), 2),
            'p25': round(float(serie.quantile(0.25)), 2),
            'mediana': round(float(serie.median()), 2),
            'p75': round(float(serie.quantile(0.75)), 2),
            'max': round(float(serie.max()), 2),
            'media': round(float(serie.mean()), 2),
            'std': round(float(serie.std()), 2),
        }
    return stats


def evaluar_estrategia_individual(
    vectores: dict,
    rutas_rasters: dict,
    criterios: dict,
    estrategia: str,
    params_rf: dict,
    ratio: int = 3,
    random_state: int = 42,
    tamano_bloque_km: float = 30.0,
    regiones_gdf: gpd.GeoDataFrame = None,
) -> dict:
    """Genera dataset, entrena y evalúa una estrategia específica de pseudo-ausencias.

    Retorna un diccionario con métricas SBCV, importancias SHAP y distribución de negativos.
    """
    print(f"\n" + "=" * 70)
    print(f"Evaluando estrategia de pseudo-ausencias: '{estrategia}' "
          f"({ESTRATEGIAS_NOMBRES.get(estrategia, estrategia)})")
    print("=" * 70)

    # 1. Generar muestras bajo la estrategia indicada
    positivas, pool_negativos = generar_dataset_muestras(
        vectores=vectores,
        rutas_rasters=rutas_rasters,
        criterios=criterios,
        ratio=ratio,
        random_state=random_state,
        estrategia_negativos=estrategia,
    )

    n_sample_neg = min(int(len(positivas) * ratio), len(pool_negativos))
    neg_sample = pool_negativos.sample(n=n_sample_neg, random_state=random_state).copy()

    # 2. Construir dataset conjunto
    dataset = gpd.GeoDataFrame(
        pd.concat([positivas, neg_sample], ignore_index=True),
        geometry='geometry',
        crs='EPSG:32719',
    ).dropna(subset=FEATURES)

    # 3. Validación espacial SBCV
    if regiones_gdf is not None:
        dataset = asignar_region_a_muestras(dataset, regiones_gdf)
    dataset = asignar_bloques_espaciales(dataset, tamano_bloque_m=tamano_bloque_km * 1000)

    res_sbcv = spatial_block_cv(
        muestras=dataset,
        features=FEATURES,
        params_rf=params_rf,
        n_splits=5,
        random_state=random_state,
        tamano_bloque_km=tamano_bloque_km,
    )

    # 4. Ajuste del modelo completo para explicabilidad SHAP
    X = dataset[FEATURES].values
    y = dataset['clase'].astype(int).values

    clf_full = RandomForestClassifier(**params_rf, random_state=random_state, n_jobs=-1)
    clf_full.fit(X, y)

    # 5. Explicabilidad SHAP (TreeExplainer sobre la clase 1: sitio apto)
    print(f"  Calculando valores SHAP con TreeExplainer ({len(X)} muestras)...")
    explainer = shap.TreeExplainer(clf_full)
    sv = explainer.shap_values(X)
    sv1 = sv[:, :, 1] if sv.ndim == 3 else sv

    # Importancia global (|SHAP| medio y porcentaje)
    mean_abs_shap = np.abs(sv1).mean(axis=0)
    total_shap = mean_abs_shap.sum()
    pct_shap = 100.0 * (mean_abs_shap / total_shap) if total_shap > 0 else np.zeros_like(mean_abs_shap)
    shap_dir = sv1.mean(axis=0)

    importancias_shap = []
    for feat, raw, pct, direc in zip(FEATURES, mean_abs_shap, pct_shap, shap_dir):
        importancias_shap.append({
            'feature': feat,
            'shap_mean_abs': round(float(raw), 5),
            'importancia_pct': round(float(pct), 2),
            'direccion_media': round(float(direc), 5),
        })
    importancias_shap = sorted(importancias_shap, key=lambda x: x['importancia_pct'], reverse=True)

    # Importancia agrupada por macro-familias
    feat_to_pct = {item['feature']: item['importancia_pct'] for item in importancias_shap}
    pct_red = sum(feat_to_pct.get(f, 0.0) for f in FAMILIA_RED)
    pct_recurso = sum(feat_to_pct.get(f, 0.0) for f in FAMILIA_RECURSO)

    # Estadísticas de distribución del pool de negativos
    stats_negativos = _calcular_estadisticas_distribucion(pool_negativos, FEATURES)

    return {
        'estrategia': estrategia,
        'nombre_descriptivo': ESTRATEGIAS_NOMBRES.get(estrategia, estrategia),
        'n_positivas': int(len(positivas)),
        'n_negativos_pool': int(len(pool_negativos)),
        'n_negativos_muestreados': int(len(neg_sample)),
        'metricas_sbcv': {
            'auc_mean': res_sbcv['auc_mean'],
            'auc_std': res_sbcv['auc_std'],
            'brier_mean': res_sbcv['brier_mean'],
            'brier_std': res_sbcv['brier_std'],
        },
        'importancias_shap': importancias_shap,
        'importancia_macro_familias': {
            'acceso_red_pct': round(float(pct_red), 2),
            'recurso_topografia_pct': round(float(pct_recurso), 2),
            'ratio_red_vs_recurso': round(float(pct_red / pct_recurso), 2) if pct_recurso > 0 else None,
        },
        'estadisticas_negativos': stats_negativos,
        'pool_negativos_df': pool_negativos,
    }


def _generar_figuras_sensibilidad(resultados: dict, figures_dir: str):
    """Genera gráficos comparativos de SHAP y de distribución de covariables."""
    os.makedirs(figures_dir, exist_ok=True)
    estrategias = ['ahp_filtrado', 'fondo_aleatorio', 'fondo_objetivo']

    # 1. Gráfico de barras agrupadas: Importancia SHAP (%) por variable
    fig, ax = plt.subplots(figsize=(10, 6))
    ancho = 0.25
    x = np.arange(len(FEATURES))

    colores = {
        'ahp_filtrado': '#9A4A1E',     # Cobre Atacama
        'fondo_aleatorio': '#2E6B4A',  # Verde
        'fondo_objetivo': '#1B2430',   # Azul oscuro
    }

    for i, est in enumerate(estrategias):
        res = resultados['estrategias'][est]
        pct_map = {item['feature']: item['importancia_pct'] for item in res['importancias_shap']}
        valores = [pct_map.get(f, 0.0) for f in FEATURES]
        ax.bar(x + i * ancho, valores, width=ancho, label=ESTRATEGIAS_NOMBRES[est],
               color=colores[est], alpha=0.9)

    ax.set_ylabel('Importancia SHAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensibilidad al Diseño de Pseudo-Ausencias: Estabilidad de Importancias SHAP',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x + ancho)
    ax.set_xticklabels(FEATURES, rotation=25, ha='right', fontsize=10)
    ax.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#CDD3DA')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    bar_path = os.path.join(figures_dir, 'sensibilidad_shap_comparacion.png')
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Distribución de variables clave en los pools de negativos
    variables_clave = ['slope', 'ghi', 'elev', 'dist_transmision']
    nombres_vars = {'slope': 'Pendiente (°)', 'ghi': 'GHI (kWh/m²)',
                    'elev': 'Elevación (m.s.n.m.)', 'dist_transmision': 'Dist. Transmisión (m)'}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for idx, var in enumerate(variables_clave):
        ax_v = axes[idx]
        datos_box = []
        labels_box = []
        for est in estrategias:
            df_pool = resultados['estrategias'][est].get('pool_negativos_df')
            if df_pool is not None and var in df_pool.columns:
                datos_box.append(df_pool[var].dropna().values)
                labels_box.append(est.replace('_', '\n'))
        if datos_box:
            bp = ax_v.boxplot(datos_box, tick_labels=labels_box, patch_artist=True,
                              showfliers=False)
            for patch, est in zip(bp['boxes'], estrategias):
                patch.set_facecolor(colores[est])
                patch.set_alpha(0.7)
        ax_v.set_title(nombres_vars.get(var, var), fontsize=11, fontweight='bold')
        ax_v.grid(axis='y', linestyle='--', alpha=0.3)

    plt.suptitle('Distribución de Covariables en los Pools de Pseudo-Ausencias',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    dist_path = os.path.join(figures_dir, 'sensibilidad_distribucion_negativos.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close()

    return {'bar_shap': bar_path, 'distribucion': dist_path}


def analizar_sensibilidad_muestreo(
    config: dict,
    directorio_raiz: str,
    out_json: str = None,
    figures_dir: str = None,
) -> dict:
    """Ejecuta el análisis de sensibilidad completo para los 3 diseños de pseudo-ausencias."""
    from src.preprocessing import cargar_capas_vectoriales

    paths_raw = config['paths']['raw']
    processed = config['paths']['processed']
    vectores = cargar_capas_vectoriales(paths_raw['vectores'])
    rutas_rasters = {
        'ghi_32719': processed['ghi_32719'],
        'slope':     processed['slope'],
        'aspect':    processed['aspect'],
        'dem_32719': processed['dem_32719'],
    }

    ml = config['ml_params']
    criterios = config['criterios']
    tamano_bloque = config.get('validacion', {}).get('tamano_bloque_km', 30)

    # Hiperparámetros base consistentes
    params_rf = {
        'n_estimators': ml.get('n_estimators', 500),
        'max_depth': 8,
        'min_samples_leaf': 3,
        'class_weight': 'balanced',
    }

    regiones_gdf = vectores['regiones'][
        vectores['regiones']['REGION'].isin(['Antofagasta', 'Atacama'])
    ]

    estrategias = ['ahp_filtrado', 'fondo_aleatorio', 'fondo_objetivo']
    resultados_estrategias = {}

    for est in estrategias:
        res = evaluar_estrategia_individual(
            vectores=vectores,
            rutas_rasters=rutas_rasters,
            criterios=criterios,
            estrategia=est,
            params_rf=params_rf,
            ratio=ml.get('ratio_negativos', 3),
            random_state=ml.get('random_state', 42),
            tamano_bloque_km=tamano_bloque,
            regiones_gdf=regiones_gdf,
        )
        resultados_estrategias[est] = res

    # Comparación de rangos de importancia SHAP
    # Extraer vectores de importancia (%) alineados con el orden canónico FEATURES
    vec_ahp = [next(item['importancia_pct'] for item in resultados_estrategias['ahp_filtrado']['importancias_shap'] if item['feature'] == f) for f in FEATURES]
    vec_aleat = [next(item['importancia_pct'] for item in resultados_estrategias['fondo_aleatorio']['importancias_shap'] if item['feature'] == f) for f in FEATURES]
    vec_obj = [next(item['importancia_pct'] for item in resultados_estrategias['fondo_objetivo']['importancias_shap'] if item['feature'] == f) for f in FEATURES]

    tau_ahp_aleat, p_tau_1 = kendalltau(vec_ahp, vec_aleat)
    tau_ahp_obj, p_tau_2 = kendalltau(vec_ahp, vec_obj)
    tau_aleat_obj, p_tau_3 = kendalltau(vec_aleat, vec_obj)

    rho_ahp_aleat, p_rho_1 = spearmanr(vec_ahp, vec_aleat)
    rho_ahp_obj, p_rho_2 = spearmanr(vec_ahp, vec_obj)
    rho_aleat_obj, p_rho_3 = spearmanr(vec_aleat, vec_obj)

    # Identificar variable número 1 en cada diseño
    top1_ahp = resultados_estrategias['ahp_filtrado']['importancias_shap'][0]['feature']
    top1_aleat = resultados_estrategias['fondo_aleatorio']['importancias_shap'][0]['feature']
    top1_obj = resultados_estrategias['fondo_objetivo']['importancias_shap'][0]['feature']

    resumen_comparativo = {
        'variable_dominante': {
            'ahp_filtrado': top1_ahp,
            'fondo_aleatorio': top1_aleat,
            'fondo_objetivo': top1_obj,
            'es_invariante': bool(top1_ahp == top1_aleat == top1_obj),
        },
        'correlacion_kendall_tau': {
            'ahp_vs_aleatorio': round(float(tau_ahp_aleat), 4),
            'ahp_vs_objetivo': round(float(tau_ahp_obj), 4),
            'aleatorio_vs_objetivo': round(float(tau_aleat_obj), 4),
        },
        'correlacion_spearman_rho': {
            'ahp_vs_aleatorio': round(float(rho_ahp_aleat), 4),
            'ahp_vs_objetivo': round(float(rho_ahp_obj), 4),
            'aleatorio_vs_objetivo': round(float(rho_aleat_obj), 4),
        },
        'comparacion_macro_familias': {
            est: resultados_estrategias[est]['importancia_macro_familias']
            for est in estrategias
        },
    }

    resultado_final = {
        'resumen_comparativo': resumen_comparativo,
        'estrategias': resultados_estrategias,
    }

    # Generar figuras si se especificó directorio
    if figures_dir:
        figs = _generar_figuras_sensibilidad(resultado_final, figures_dir)
        resultado_final['figuras'] = figs

    # Limpiar DataFrames internos antes de serializar a JSON
    for est in estrategias:
        if 'pool_negativos_df' in resultado_final['estrategias'][est]:
            del resultado_final['estrategias'][est]['pool_negativos_df']

    if out_json:
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(resultado_final, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Resultados de sensibilidad guardados en: {out_json}")

    return resultado_final
