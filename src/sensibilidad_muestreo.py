"""Análisis de sensibilidad al diseño de pseudo-ausencias en el modelo de aptitud solar.

Evalúa cómo condiciona la selección de negativos lo que el Random Forest aprende,
contrastando 4 diseños de muestreo:
  1) 'ahp_filtrado': línea base — buffer 5 km + exclusiones + filtros técnicos AHP.
  2) 'fondo_aleatorio': buffer 5 km + exclusiones territoriales (sin filtros técnicos).
  3) 'sin_filtros_geofisicos': ablación de la línea base, conserva solo la distancia a red.
  4) 'grupo_objetivo': target-group background (Phillips et al. 2009) — candidatos extraídos
     del entorno de subestaciones y almacenamiento, donde el sector ya prospectó.

Calcula la estabilidad de métricas (SBCV 5-fold sobre bloques espaciales), la jerarquía
SHAP de variables, la correlación de rangos (Kendall's tau y Spearman rho, cada uno con su
p-valor) y la dominancia de macro-familias ('acceso a red' vs 'recurso / topografía').

Los hiperparámetros del RF NO se fijan aquí: se leen de los que Optuna eligió para el modelo
del proyecto, o la comparación no diría nada sobre el modelo que cita el informe.
"""

import os
import json
from itertools import combinations
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

# Variables cuya distribución se compara entre pools de negativos (panel de boxplots).
VARIABLES_BOXPLOT = ('slope', 'ghi', 'elev', 'dist_transmision')

ESTRATEGIAS_NOMBRES = {
    'ahp_filtrado': 'AHP Filtrado (Línea Base)',
    'fondo_aleatorio': 'Fondo Aleatorio (Sin Filtros)',
    'sin_filtros_geofisicos': 'Sin Filtros Geofísicos (Ablación)',
    'grupo_objetivo': 'Grupo Objetivo (TGB)',
}

# Orden canónico de comparación. La línea base va primera: el resto se contrasta contra ella.
ESTRATEGIAS = ('ahp_filtrado', 'fondo_aleatorio', 'sin_filtros_geofisicos', 'grupo_objetivo')

COLORES_ESTRATEGIA = {
    'ahp_filtrado': '#9A4A1E',            # Cobre Atacama
    'fondo_aleatorio': '#2E6B4A',         # Verde
    'sin_filtros_geofisicos': '#1B2430',  # Azul oscuro
    'grupo_objetivo': '#F2B134',          # Sol
}


def cargar_hiperparametros_del_modelo(ruta_metricas: str) -> Tuple[dict, str]:
    """Recupera los hiperparámetros que Optuna eligió para el modelo del proyecto.

    La comparación entre diseños de pseudo-ausencias solo es concluyente si se hace sobre EL
    bosque del proyecto: con otra profundidad o otro mínimo de hoja, la jerarquía SHAP puede
    ordenarse distinto y la conclusión no se traslada al modelo que cita el informe.
    `entrenar_modelo_rf` ya persiste esos valores en `best_params` de model_rf_metrics.json.

    Devuelve (params_rf, origen) donde `origen` documenta de dónde salieron, para dejarlo
    registrado en el JSON de salida.
    """
    # class_weight='balanced' no lo elige Optuna: es fijo en src/modeling.py:77 y debe
    # replicarse aquí o los modelos no serían comparables.
    if os.path.exists(ruta_metricas):
        with open(ruta_metricas, 'r', encoding='utf-8') as f:
            best = (json.load(f) or {}).get('best_params') or {}
        if best:
            params = {
                'n_estimators': int(best['n_estimators']),
                'max_depth': int(best['max_depth']),
                'min_samples_leaf': int(best['min_samples_leaf']),
                'class_weight': 'balanced',
            }
            return params, f"best_params de {os.path.basename(ruta_metricas)} (Optuna)"

    raise FileNotFoundError(
        f"No se encontraron los hiperparámetros del modelo en '{ruta_metricas}'. "
        "Corre antes el entrenamiento (scripts/run_entrenamiento.py): comparar los diseños "
        "de pseudo-ausencias con hiperparámetros distintos a los del modelo del proyecto "
        "haría que la conclusión sobre la jerarquía SHAP no aplique al modelo publicado."
    )


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
        # Solo las columnas que los boxplots necesitan, como arrays sueltos. Antes se
        # devolvía el GeoDataFrame completo y los tres (el de 'fondo_aleatorio' es varias
        # veces mayor) quedaban vivos a la vez junto a las matrices SHAP.
        '_muestras_para_boxplot': {
            var: pool_negativos[var].dropna().to_numpy(dtype=float, copy=True)
            for var in VARIABLES_BOXPLOT if var in pool_negativos.columns
        },
    }


def _generar_figuras_sensibilidad(resultados: dict, figures_dir: str):
    """Genera gráficos comparativos de SHAP y de distribución de covariables."""
    os.makedirs(figures_dir, exist_ok=True)
    estrategias = list(ESTRATEGIAS)

    # 1. Gráfico de barras agrupadas: Importancia SHAP (%) por variable
    fig, ax = plt.subplots(figsize=(11, 6))
    # El ancho se deriva del número de diseños: con 4 barras fijas de 0.25 se solapaban.
    ancho = 0.8 / len(estrategias)
    x = np.arange(len(FEATURES))
    colores = COLORES_ESTRATEGIA

    for i, est in enumerate(estrategias):
        res = resultados['estrategias'][est]
        pct_map = {item['feature']: item['importancia_pct'] for item in res['importancias_shap']}
        valores = [pct_map.get(f, 0.0) for f in FEATURES]
        ax.bar(x + i * ancho, valores, width=ancho, label=ESTRATEGIAS_NOMBRES[est],
               color=colores[est], alpha=0.9)

    ax.set_ylabel('Importancia SHAP (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensibilidad al Diseño de Pseudo-Ausencias: Estabilidad de Importancias SHAP',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x + ancho * (len(estrategias) - 1) / 2)
    ax.set_xticklabels(FEATURES, rotation=25, ha='right', fontsize=10)
    ax.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#CDD3DA')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    bar_path = os.path.join(figures_dir, 'sensibilidad_shap_comparacion.png')
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Distribución de variables clave en los pools de negativos
    nombres_vars = {'slope': 'Pendiente (°)', 'ghi': 'GHI (kWh/m²)',
                    'elev': 'Elevación (m.s.n.m.)', 'dist_transmision': 'Dist. Transmisión (m)'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for idx, var in enumerate(VARIABLES_BOXPLOT):
        ax_v = axes[idx]
        datos_box, labels_box, ests_dibujadas = [], [], []
        for est in estrategias:
            muestras = resultados['estrategias'][est].get('_muestras_para_boxplot', {})
            serie = muestras.get(var)
            if serie is not None and len(serie) > 0:
                datos_box.append(serie)
                labels_box.append(est.replace('_', '\n'))
                # Se registra qué estrategia produjo cada caja: colorear con zip sobre la
                # lista completa desalineaba los colores en cuanto una quedaba fuera.
                ests_dibujadas.append(est)
        if datos_box:
            bp = ax_v.boxplot(datos_box, tick_labels=labels_box, patch_artist=True,
                              showfliers=False)
            for patch, est in zip(bp['boxes'], ests_dibujadas):
                patch.set_facecolor(colores[est])
                patch.set_alpha(0.7)
        ax_v.set_title(nombres_vars.get(var, var), fontsize=11, fontweight='bold')
        ax_v.tick_params(axis='x', labelsize=7)
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
    out_json: str = None,
    figures_dir: str = None,
) -> dict:
    """Ejecuta el análisis de sensibilidad completo para los diseños de `ESTRATEGIAS`.

    `config['paths']` debe venir ya resuelto a rutas absolutas (`src.utils._resolver_rutas`).
    """
    from src.preprocessing import cargar_capas_vectoriales

    # Los hiperparámetros se resuelven ANTES de cargar nada pesado: si falta el modelo
    # entrenado, conviene abortar en el primer segundo y no tras leer las 8 capas vectoriales.
    ruta_metricas = config['paths']['results'].get('model_rf', '')
    ruta_metricas = os.path.join(os.path.dirname(ruta_metricas), 'model_rf_metrics.json')
    params_rf, origen_params = cargar_hiperparametros_del_modelo(ruta_metricas)
    print(f"Hiperparámetros del RF: {params_rf}  [origen: {origen_params}]")

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

    regiones_gdf = vectores['regiones'][
        vectores['regiones']['REGION'].isin(['Antofagasta', 'Atacama'])
    ]

    estrategias = list(ESTRATEGIAS)
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

    # Comparación de rangos de importancia SHAP, generalizada a todos los pares: así agregar
    # o quitar un diseño no obliga a reescribir la comparación (antes estaba cableada a 3).
    def _vector_importancias(est: str) -> list:
        """Importancias (%) de un diseño, alineadas al orden canónico de FEATURES."""
        pct = {item['feature']: item['importancia_pct']
               for item in resultados_estrategias[est]['importancias_shap']}
        return [pct[f] for f in FEATURES]

    vectores_imp = {est: _vector_importancias(est) for est in estrategias}

    # Cada estadístico va con su p-valor: son correlaciones de rango sobre solo 7 variables,
    # donde un tau alto puede no ser distinguible del azar. Reportar el coeficiente solo sería
    # el error que advierte la clase 15 (láms. 15 y 27).
    kendall, spearman = {}, {}
    for a, b in combinations(estrategias, 2):
        clave = f"{a}_vs_{b}"
        tau, p_tau = kendalltau(vectores_imp[a], vectores_imp[b])
        rho, p_rho = spearmanr(vectores_imp[a], vectores_imp[b])
        kendall[clave] = {'tau': round(float(tau), 4), 'p_valor': round(float(p_tau), 4)}
        spearman[clave] = {'rho': round(float(rho), 4), 'p_valor': round(float(p_rho), 4)}

    dominantes = {est: resultados_estrategias[est]['importancias_shap'][0]['feature']
                  for est in estrategias}

    resumen_comparativo = {
        'variable_dominante': {
            **dominantes,
            'es_invariante': bool(len(set(dominantes.values())) == 1),
        },
        'correlacion_kendall_tau': kendall,
        'correlacion_spearman_rho': spearman,
        'n_variables_correlacionadas': len(FEATURES),
        'comparacion_macro_familias': {
            est: resultados_estrategias[est]['importancia_macro_familias']
            for est in estrategias
        },
    }

    resultado_final = {
        # Queda registrado con qué bosque se comparó: sin esto no se puede afirmar que la
        # conclusión aplique al modelo del informe.
        'hiperparametros_rf': dict(params_rf),
        'origen_hiperparametros': origen_params,
        'resumen_comparativo': resumen_comparativo,
        'estrategias': resultados_estrategias,
    }

    # Generar figuras si se especificó directorio
    if figures_dir:
        figs = _generar_figuras_sensibilidad(resultado_final, figures_dir)
        resultado_final['figuras'] = figs

    # Limpiar DataFrames internos antes de serializar a JSON
    for est in estrategias:
        resultado_final['estrategias'][est].pop('_muestras_para_boxplot', None)

    if out_json:
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(resultado_final, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Resultados de sensibilidad guardados en: {out_json}")

    return resultado_final
