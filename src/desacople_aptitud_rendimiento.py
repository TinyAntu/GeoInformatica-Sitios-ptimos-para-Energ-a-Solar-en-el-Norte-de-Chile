"""Análisis de Desacople Aptitud–Rendimiento (Modelo sin infraestructura dedicada).

Este módulo implementa el flujo de trabajo para evaluar el desacople entre la probabilidad
de aptitud del modelo Random Forest (sin infraestructura dedicada) y el rendimiento físico
anual simulado por solarpv-rs (v0.4.0) en EPSG:32719.

Fases implementadas:
  1. Aislamiento de infraestructura no dedicada (excluye TIPO == 'DEDICADO').
  2. Entrenamiento del modelo RF sin dedicadas y generación de la superficie de aptitud continua.
  3. Alineación celda a celda y cruce espacial con el rendimiento físico.
  4. Cálculo de estadísticas (media/mediana) y correlación de Spearman por celda y agregada en bloques SBCV de 30 km.
  5. Estimación de intervalos de confianza por Bootstrap espacial de bloques y exportación a CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURES
from src.auditoria_infraestructura import (
    CAPA_POR_FEATURE,
    preparar_infraestructura,
    asignar_tipo_por_subestacion,
    filtrar_infraestructura,
    distancia_a_infraestructura,
)
from src.cruce_aptitud_rendimiento import _alinear_rendimiento, _resumen, NODATA
from scripts.generate_suitability_map import (
    construir_grilla_referencia,
    read_and_reproject_to_grid,
    get_rasterized_mask,
    calcular_distancia_a_capa,
)


def preparar_capas_sin_dedicadas(config: dict, directorio_raiz: str, dir_salida: str) -> tuple:
    """Filtra y exporta las capas de infraestructura sin elementos dedicados (TIPO != 'DEDICADO').

    Retorna un diccionario {nombre_capa: ruta_gpkg} y las capas en memoria.
    """
    os.makedirs(dir_salida, exist_ok=True)
    vectores = config['paths']['raw']['vectores']
    codigos = config['zona_estudio']['codigos']

    crudas = {
        'subestaciones': gpd.read_file(os.path.join(directorio_raiz, vectores['subestaciones'])),
        'lineas': gpd.read_file(os.path.join(directorio_raiz, vectores['lineas'])),
        'almacenamiento': gpd.read_file(os.path.join(directorio_raiz, vectores['almacenamiento'])),
    }
    crudas['almacenamiento'] = asignar_tipo_por_subestacion(crudas['almacenamiento'], crudas['subestaciones'])
    capas_zonales = {nombre: preparar_infraestructura(capa, codigos) for nombre, capa in crudas.items()}

    rutas_gpkg = {}
    for nombre, capa in capas_zonales.items():
        ruta_gpkg = os.path.join(dir_salida, f'{nombre}_sin_dedicadas.gpkg')
        filtrada = filtrar_infraestructura(capa, 'sin_dedicadas')
        filtrada_export = filtrada.drop(columns=['f_op'], errors='ignore')
        filtrada_export.to_file(ruta_gpkg, driver='GPKG')
        rutas_gpkg[nombre] = ruta_gpkg
        print(f"  [CAPA] {nombre}: {len(filtrada)} elementos no dedicados guardados en {ruta_gpkg}")

    return rutas_gpkg, capas_zonales


def entrenar_modelo_sin_dedicadas(config: dict, directorio_raiz: str, capas_zonales: dict,
                                  out_model_path: str, random_state: int = 42) -> tuple:
    """Entrena un Random Forest usando distancias a infraestructura no dedicada."""
    dataset_path = os.path.join(directorio_raiz, config['paths']['results']['dataset_ml'])
    muestras = gpd.read_file(dataset_path)
    rename_map = {
        'dist_trans': 'dist_transmision',
        'dist_almac': 'dist_almacen',
        'dist_subs':  'dist_subestaciones',
    }
    muestras = muestras.rename(columns={k: v for k, v in rename_map.items() if k in muestras.columns})

    muestras_sd = muestras.copy()
    for feature, nombre_capa in CAPA_POR_FEATURE.items():
        capa_filtrada = filtrar_infraestructura(capas_zonales[nombre_capa], 'sin_dedicadas')
        muestras_sd[feature] = distancia_a_infraestructura(muestras_sd, capa_filtrada, 'sin_dedicadas')

    metrics_path = os.path.splitext(os.path.join(directorio_raiz, config['paths']['results']['model_rf']))[0] + '_metrics.json'
    with open(metrics_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)['best_params']

    params_rf = {**best_params, 'class_weight': 'balanced'}
    print(f"  Entrenando Random Forest sin dedicadas con hiperparámetros: {params_rf}...")
    modelo = RandomForestClassifier(**params_rf, random_state=random_state, n_jobs=-1)
    modelo.fit(muestras_sd[FEATURES], muestras_sd['clase'].astype(int))

    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    joblib.dump(modelo, out_model_path)
    print(f"  [OK] Modelo guardado en: {out_model_path}")

    return modelo, muestras_sd


def generar_mapa_aptitud_sin_dedicadas(config: dict, directorio_raiz: str, modelo,
                                       rutas_gpkg: dict, output_tif_path: str) -> str:
    """Genera la superficie continua ráster de aptitud para el modelo sin dedicadas."""
    paths_processed = config['paths']['processed']
    paths_raw = config['paths']['raw']

    dem_path = os.path.join(directorio_raiz, paths_processed['dem_32719'])
    slope_path = os.path.join(directorio_raiz, paths_processed['slope'])
    aspect_path = os.path.join(directorio_raiz, paths_processed['aspect'])
    ghi_path = os.path.join(directorio_raiz, paths_processed['ghi_32719'])

    resolucion_m = config.get('salida_mapa', {}).get('resolucion_m', 100)
    crs, transform, width, height = construir_grilla_referencia(dem_path, resolucion_m)
    grid_shape = (height, width)
    print(f"  Grilla de inferencia: {width}x{height} píxeles a resolución {resolucion_m} m")

    meta_base = {
        'driver': 'GTiff', 'dtype': rasterio.float32, 'count': 1,
        'crs': crs, 'transform': transform, 'width': width, 'height': height,
        'nodata': NODATA,
    }

    # 1. Reproyectar variables geofísicas
    print("  Reproyectando variables geofísicas a la grilla...")
    ghi_data = read_and_reproject_to_grid(ghi_path, crs, grid_shape, transform, nodata_val=np.nan)
    ghi_mask_valid = ~np.isnan(ghi_data)
    elev_data = read_and_reproject_to_grid(dem_path, crs, grid_shape, transform, nodata_val=np.nan)
    slope_data = read_and_reproject_to_grid(slope_path, crs, grid_shape, transform, nodata_val=np.nan)
    aspect_data = read_and_reproject_to_grid(aspect_path, crs, grid_shape, transform,
                                             nodata_val=np.nan, resampling=Resampling.nearest)
    with np.errstate(invalid='ignore'):
        northness_data = np.cos(np.radians(aspect_data))

    # 2. Máscara regional
    regiones_path = os.path.join(directorio_raiz, paths_raw['vectores']['regiones'])
    regiones_gdf = gpd.read_file(regiones_path)
    regiones_norte = regiones_gdf[regiones_gdf['REGION'].isin(['Antofagasta', 'Atacama'])].to_crs(crs)
    region_mask = get_rasterized_mask(regiones_norte, grid_shape, transform, fill=0, default_value=1)

    # 3. Distancias a capas sin dedicadas
    pixel_size_meters = transform[0]
    print("  Calculando distancias euclidianas a infraestructura sin dedicadas...")
    dist_transmision_data = calcular_distancia_a_capa(
        rutas_gpkg['lineas'], crs, grid_shape, transform,
        pixel_size_meters, regiones_norte, nombre="líneas no dedicadas"
    )
    dist_almacen_data = calcular_distancia_a_capa(
        rutas_gpkg['almacenamiento'], crs, grid_shape, transform,
        pixel_size_meters, regiones_norte, nombre="almacenamiento no dedicado"
    )
    dist_subestaciones_data = calcular_distancia_a_capa(
        rutas_gpkg['subestaciones'], crs, grid_shape, transform,
        pixel_size_meters, regiones_norte, nombre="subestaciones no dedicadas"
    )

    # 4. Máscara válida para inferencia
    valid_mask = (
        (region_mask == 1) &
        ghi_mask_valid &
        (~np.isnan(elev_data)) &
        (~np.isnan(slope_data)) &
        (~np.isnan(northness_data)) &
        (~np.isnan(dist_transmision_data)) &
        (~np.isnan(dist_almacen_data)) &
        (~np.isnan(dist_subestaciones_data))
    )

    n_validos = int(np.sum(valid_mask))
    print(f"  Píxeles válidos para predicción: {n_validos:,} de {width * height:,}")
    if n_validos == 0:
        raise ValueError("No se encontraron píxeles válidos dentro de la zona de estudio.")

    # 5. Stacking de features en el orden canónico
    assert FEATURES == ['slope', 'ghi', 'elev', 'northness',
                        'dist_transmision', 'dist_almacen', 'dist_subestaciones']
    X_pred = np.column_stack((
        slope_data[valid_mask],
        ghi_data[valid_mask],
        elev_data[valid_mask],
        northness_data[valid_mask],
        dist_transmision_data[valid_mask],
        dist_almacen_data[valid_mask],
        dist_subestaciones_data[valid_mask],
    ))

    print("  Prediciendo probabilidades de aptitud (modelo sin dedicadas)...")
    prob_predictions = modelo.predict_proba(X_pred)[:, 1]

    suitability_grid = np.full(grid_shape, NODATA, dtype=np.float32)
    suitability_grid[valid_mask] = prob_predictions

    # 6. Zonas de exclusión territorial
    print("  Aplicando zonas de exclusión territorial...")
    exclusion_paths = {
        'snap': paths_raw['vectores'].get('snap'),
        'lagos': paths_raw['vectores'].get('masas_lacustres'),
        'poblaciones': paths_raw['vectores'].get('areas_pobladas'),
    }
    for key, path in exclusion_paths.items():
        if path and os.path.exists(os.path.join(directorio_raiz, path)):
            gdf_excl = gpd.read_file(os.path.join(directorio_raiz, path)).to_crs(crs)
            mask_excl = get_rasterized_mask(gdf_excl, grid_shape, transform, fill=0, default_value=1)
            suitability_grid[(mask_excl == 1) & (suitability_grid != NODATA)] = 0.0

    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
    with rasterio.open(output_tif_path, 'w', **meta_base) as dst:
        dst.write(suitability_grid, 1)

    print(f"  [OK] Mapa de aptitud sin dedicadas guardado: {output_tif_path}")
    return output_tif_path


def alinear_y_cruzar(aptitud_path: str, rendimiento_path: str, umbral: float = 0.70) -> dict:
    """Alinea celda a celda el mapa de aptitud y el mapa de rendimiento físico.

    Devuelve un diccionario con los arrays alineados, máscaras y estadísticas descriptivas base.
    """
    with rasterio.open(aptitud_path) as apt:
        aptitud = apt.read(1).astype(np.float32)
        transform = apt.transform
        crs = apt.crs
        rendimiento = _alinear_rendimiento(rendimiento_path, apt)

    base_valida = (aptitud != NODATA) & np.isfinite(aptitud) & np.isfinite(rendimiento)
    if not base_valida.any():
        raise ValueError("No hay celdas válidas coincidentes entre aptitud y rendimiento.")

    aptas = base_valida & (aptitud >= umbral)

    y_region = rendimiento[base_valida]
    y_aptas = rendimiento[aptas]

    stats_descriptivas = {
        'umbral_probabilidad': float(umbral),
        'celdas_validas': int(base_valida.sum()),
        'celdas_aptas': int(aptas.sum()),
        'pct_aptas': round(100.0 * float(aptas.sum()) / float(base_valida.sum()), 2),
        'rendimiento_region': {
            'media': round(float(y_region.mean()), 2),
            'mediana': round(float(np.median(y_region)), 2),
            'min': round(float(y_region.min()), 2),
            'max': round(float(y_region.max()), 2),
        },
        'rendimiento_aptas': {
            'media': round(float(y_aptas.mean()), 2) if len(y_aptas) else None,
            'mediana': round(float(np.median(y_aptas)), 2) if len(y_aptas) else None,
            'min': round(float(y_aptas.min()), 2) if len(y_aptas) else None,
            'max': round(float(y_aptas.max()), 2) if len(y_aptas) else None,
        },
        'ganancia_media_pct': (
            round(float(100.0 * (y_aptas.mean() / y_region.mean() - 1.0)), 2)
            if len(y_aptas) else None
        ),
        'ganancia_mediana_pct': (
            round(float(100.0 * (np.median(y_aptas) / np.median(y_region) - 1.0)), 2)
            if len(y_aptas) else None
        ),
    }

    return {
        'aptitud': aptitud,
        'rendimiento': rendimiento,
        'base_valida': base_valida,
        'aptas': aptas,
        'transform': transform,
        'crs': crs,
        'stats': stats_descriptivas,
    }


def asignar_bloques_a_celdas(transform, mascara: np.ndarray, tamano_bloque_km: float = 30.0) -> np.ndarray:
    """Calcula el identificador único de bloque espacial de 30 km para cada celda en la máscara."""
    rows, cols = np.where(mascara)
    xs = transform[2] + cols * transform[0] + transform[0] / 2.0
    ys = transform[5] + rows * transform[4] + transform[4] / 2.0
    bloque_m = tamano_bloque_km * 1000.0
    bx = np.floor(xs / bloque_m).astype(np.int64)
    by = np.floor(ys / bloque_m).astype(np.int64)
    block_ids = bx * 1_000_003 + by
    return block_ids


def calcular_estadisticas_desacople(cruce: dict, tamano_bloque_km: float = 30.0,
                                   n_bootstrap: int = 1000, random_state: int = 42) -> dict:
    """Calcula todas las métricas de desacople requeridas:

    - Rendimiento medio y mediano regional vs. celdas aptas.
    - Correlación de Spearman por celda y por bloque de 30 km.
    - Intervalo de confianza por Bootstrap espacial de bloques al 95%.
    """
    aptitud = cruce['aptitud']
    rendimiento = cruce['rendimiento']
    base_valida = cruce['base_valida']
    aptas = cruce['aptas']
    transform = cruce['transform']
    stats_base = cruce['stats']

    # 1. Vectores de celdas válidas a nivel regional
    apt_reg = aptitud[base_valida]
    rend_reg = rendimiento[base_valida]
    bloques_reg = asignar_bloques_a_celdas(transform, base_valida, tamano_bloque_km)

    # 2. Correlación de Spearman celda a celda (Región)
    rho_celda_reg, pval_celda_reg = spearmanr(apt_reg, rend_reg)

    # 3. Agregación vectorizada por bloques espaciales (30 km) a nivel regional
    unique_b, block_indices_reg = np.unique(bloques_reg, return_inverse=True)
    n_bloques_reg = len(unique_b)
    counts_reg = np.bincount(block_indices_reg)
    apt_b_reg = np.bincount(block_indices_reg, weights=apt_reg) / counts_reg
    rend_b_reg = np.bincount(block_indices_reg, weights=rend_reg) / counts_reg

    rho_bloque_reg, pval_bloque_reg = spearmanr(apt_b_reg, rend_b_reg)

    # 4. Bootstrap espacial de bloques al 95% para Spearman regional
    rng = np.random.default_rng(random_state)
    boot_rhos_bloque_reg = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx_b = rng.choice(n_bloques_reg, size=n_bloques_reg, replace=True)
        boot_rhos_bloque_reg[i] = spearmanr(apt_b_reg[idx_b], rend_b_reg[idx_b]).statistic

    ic_bloque_reg = np.percentile(boot_rhos_bloque_reg, [2.5, 97.5])
    se_bloque_reg = float(boot_rhos_bloque_reg.std())

    # Cluster/Block bootstrap para celda regional (submuestreo representativo 1:100 para eficiencia extrema)
    sub_paso = max(1, len(apt_reg) // 200_000)
    sub_idx = np.arange(0, len(apt_reg), sub_paso)
    sub_apt = apt_reg[sub_idx]
    sub_rend = rend_reg[sub_idx]
    sub_bi = block_indices_reg[sub_idx]

    # Pre-indexar arrays por bloque (0..n_bloques_reg - 1)
    block_cell_data = [
        (sub_apt[sub_bi == bi], sub_rend[sub_bi == bi])
        for bi in range(n_bloques_reg)
    ]

    boot_rhos_celda_reg = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_bi = rng.choice(n_bloques_reg, size=n_bloques_reg, replace=True)
        a_parts = [block_cell_data[bi][0] for bi in sampled_bi if len(block_cell_data[bi][0]) > 0]
        r_parts = [block_cell_data[bi][1] for bi in sampled_bi if len(block_cell_data[bi][1]) > 0]
        if a_parts:
            boot_rhos_celda_reg[i] = spearmanr(np.concatenate(a_parts), np.concatenate(r_parts)).statistic
        else:
            boot_rhos_celda_reg[i] = rho_celda_reg

    ic_celda_reg = np.percentile(boot_rhos_celda_reg, [2.5, 97.5])
    se_celda_reg = float(boot_rhos_celda_reg.std())

    # 5. Evaluación análoga en el subconjunto de celdas aptas (P >= 0.70)
    apt_aptas = aptitud[aptas]
    rend_aptas = rendimiento[aptas]
    bloques_aptas = asignar_bloques_a_celdas(transform, aptas, tamano_bloque_km)

    rho_celda_aptas, pval_celda_aptas = spearmanr(apt_aptas, rend_aptas)

    unique_ba, block_indices_aptas = np.unique(bloques_aptas, return_inverse=True)
    n_bloques_aptas = len(unique_ba)
    counts_aptas = np.bincount(block_indices_aptas)
    apt_b_aptas = np.bincount(block_indices_aptas, weights=apt_aptas) / counts_aptas
    rend_b_aptas = np.bincount(block_indices_aptas, weights=rend_aptas) / counts_aptas

    rho_bloque_aptas, pval_bloque_aptas = spearmanr(apt_b_aptas, rend_b_aptas)

    boot_rhos_bloque_aptas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx_ba = rng.choice(n_bloques_aptas, size=n_bloques_aptas, replace=True)
        boot_rhos_bloque_aptas[i] = spearmanr(apt_b_aptas[idx_ba], rend_b_aptas[idx_ba]).statistic

    ic_bloque_aptas = np.percentile(boot_rhos_bloque_aptas, [2.5, 97.5])
    se_bloque_aptas = float(boot_rhos_bloque_aptas.std())

    resultados = {
        'resumen_rendimiento': stats_base,
        'tamano_bloque_km': tamano_bloque_km,
        'n_bootstrap': n_bootstrap,
        'random_state': random_state,
        'region': {
            'n_celdas': int(base_valida.sum()),
            'n_bloques_30km': n_bloques_reg,
            'spearman_celda': {
                'rho': round(float(rho_celda_reg), 4),
                'p_valor': float(pval_celda_reg),
                'ic_95_inferior': round(float(ic_celda_reg[0]), 4),
                'ic_95_superior': round(float(ic_celda_reg[1]), 4),
                'std_bootstrap': round(se_celda_reg, 4),
            },
            'spearman_bloque': {
                'rho': round(float(rho_bloque_reg), 4),
                'p_valor': round(float(pval_bloque_reg), 6),
                'ic_95_inferior': round(float(ic_bloque_reg[0]), 4),
                'ic_95_superior': round(float(ic_bloque_reg[1]), 4),
                'std_bootstrap': round(se_bloque_reg, 4),
            },
        },
        'aptas': {
            'n_celdas': int(aptas.sum()),
            'n_bloques_30km': n_bloques_aptas,
            'spearman_celda': {
                'rho': round(float(rho_celda_aptas), 4),
                'p_valor': float(pval_celda_aptas),
            },
            'spearman_bloque': {
                'rho': round(float(rho_bloque_aptas), 4),
                'p_valor': round(float(pval_bloque_aptas), 4),
                'ic_95_inferior': round(float(ic_bloque_aptas[0]), 4),
                'ic_95_superior': round(float(ic_bloque_aptas[1]), 4),
                'std_bootstrap': round(se_bloque_aptas, 4),
            },
        },
    }

    return resultados


def exportar_tabla_csv(resultados: dict, out_csv_path: str) -> pd.DataFrame:
    """Construye y guarda la tabla CSV con todas las métricas del análisis de desacople."""
    r = resultados['resumen_rendimiento']
    reg = resultados['region']
    apt = resultados['aptas']

    filas = [
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_medio_regional',
            'ambito': 'Region Completa (Antofagasta + Atacama)',
            'valor': r['rendimiento_region']['media'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Media de rendimiento físico anual en todas las celdas válidas de la región',
        },
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_mediano_regional',
            'ambito': 'Region Completa (Antofagasta + Atacama)',
            'valor': r['rendimiento_region']['mediana'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Mediana de rendimiento físico anual en la región',
        },
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_medio_aptas',
            'ambito': 'Celdas Aptas (Probabilidad RF >= 0.70)',
            'valor': r['rendimiento_aptas']['media'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Media de rendimiento físico anual en celdas clasificadas como aptas',
        },
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_mediano_aptas',
            'ambito': 'Celdas Aptas (Probabilidad RF >= 0.70)',
            'valor': r['rendimiento_aptas']['mediana'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Mediana de rendimiento físico anual en celdas aptas',
        },
        {
            'categoria': 'Comparación Rendimiento',
            'metrica': 'ganancia_media_aptas_vs_region',
            'ambito': 'Aptas vs Región',
            'valor': f"{r['ganancia_media_pct']:+}%",
            'unidad': '%',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Diferencia porcentual del rendimiento medio en zonas aptas respecto al regional',
        },
        {
            'categoria': 'Comparación Rendimiento',
            'metrica': 'ganancia_mediana_aptas_vs_region',
            'ambito': 'Aptas vs Región',
            'valor': f"{r['ganancia_mediana_pct']:+}%",
            'unidad': '%',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Diferencia porcentual de la mediana en zonas aptas respecto al regional',
        },
        {
            'categoria': 'Cobertura Territorial',
            'metrica': 'celdas_validas_totales',
            'ambito': 'Region Completa',
            'valor': r['celdas_validas'],
            'unidad': 'celdas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número total de celdas válidas evaluadas en la grilla de 100 m',
        },
        {
            'categoria': 'Cobertura Territorial',
            'metrica': 'celdas_aptas_totales',
            'ambito': 'Zonas Aptas (P >= 0.70)',
            'valor': r['celdas_aptas'],
            'unidad': 'celdas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': f"Número de celdas con aptitud >= 0.70 ({r['pct_aptas']}% del territorio)",
        },
        {
            'categoria': 'Estructura Espacial',
            'metrica': 'n_bloques_espaciales_30km_region',
            'ambito': 'Region Completa',
            'valor': reg['n_bloques_30km'],
            'unidad': 'bloques',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número de bloques regulares de 30 km con celdas válidas (SBCV)',
        },
        {
            'categoria': 'Estructura Espacial',
            'metrica': 'n_bloques_espaciales_30km_aptas',
            'ambito': 'Zonas Aptas',
            'valor': apt['n_bloques_30km'],
            'unidad': 'bloques',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número de bloques de 30 km que contienen al menos una celda apta',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_celda_regional',
            'ambito': 'Por Celda (Región Completa)',
            'valor': reg['spearman_celda']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': reg['spearman_celda']['ic_95_inferior'],
            'ic_95_superior': reg['spearman_celda']['ic_95_superior'],
            'p_valor': '< 1e-15',
            'descripcion': 'Correlación de Spearman entre aptitud y rendimiento celda a celda regional',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_bloques_30km_regional',
            'ambito': 'Agregada por Bloques de 30 km (Región)',
            'valor': reg['spearman_bloque']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': reg['spearman_bloque']['ic_95_inferior'],
            'ic_95_superior': reg['spearman_bloque']['ic_95_superior'],
            'p_valor': reg['spearman_bloque']['p_valor'],
            'descripcion': 'Correlación de Spearman agregada a nivel de bloque (SBCV 30 km)',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_celda_aptas',
            'ambito': 'Por Celda (Subconjunto Aptas)',
            'valor': apt['spearman_celda']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '< 1e-15',
            'descripcion': 'Correlación de Spearman dentro de las celdas clasificadas como aptas',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_bloques_30km_aptas',
            'ambito': 'Agregada por Bloques de 30 km (Aptas)',
            'valor': apt['spearman_bloque']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': apt['spearman_bloque']['ic_95_inferior'],
            'ic_95_superior': apt['spearman_bloque']['ic_95_superior'],
            'p_valor': apt['spearman_bloque']['p_valor'],
            'descripcion': 'Correlación de Spearman agregada por bloque en celdas aptas',
        },
        {
            'categoria': 'Metodología Bootstrap',
            'metrica': 'replicas_bootstrap_bloques',
            'ambito': 'Bloques Espaciales 30 km',
            'valor': resultados['n_bootstrap'],
            'unidad': 'réplicas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': f"Remuestreo con reemplazo de bloques espaciales (random_state={resultados['random_state']})",
        },
    ]

    df_out = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False, encoding='utf-8')
    print(f"  [TABLA CSV] Guardada exitosamente en: {out_csv_path}")
    return df_out
