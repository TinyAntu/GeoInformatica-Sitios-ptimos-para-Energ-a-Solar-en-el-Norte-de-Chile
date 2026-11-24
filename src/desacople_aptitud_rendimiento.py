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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr, t as student_t
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


def _asignar_grilla_bloques(transform, mascara: np.ndarray, tamano_bloque_km: float = 30.0) -> dict:
    """Asigna coordenadas UTM, bloques regulares y superbloques 2x2 a las celdas válidas."""
    rows, cols = np.where(mascara)
    xs = transform[2] + cols * transform[0] + transform[0] / 2.0
    ys = transform[5] + rows * transform[4] + transform[4] / 2.0
    bloque_m = tamano_bloque_km * 1000.0
    bx = np.floor(xs / bloque_m).astype(np.int64)
    by = np.floor(ys / bloque_m).astype(np.int64)
    block_ids = bx * 1_000_003 + by
    # Agrupación jerárquica 2x2: cada superbloque cubre (2 * tamano_bloque_km)^2
    sbx = bx // 2
    sby = by // 2
    superblock_ids = sbx * 1_000_003 + sby
    return {
        'xs': xs,
        'ys': ys,
        'block_ids': block_ids,
        'superblock_ids': superblock_ids,
    }


def asignar_bloques_a_celdas(transform, mascara: np.ndarray, tamano_bloque_km: float = 30.0) -> np.ndarray:
    """Calcula el identificador único de bloque espacial de 30 km para cada celda en la máscara."""
    return _asignar_grilla_bloques(transform, mascara, tamano_bloque_km)['block_ids']


def _agregar_por_bloques(apt_vals: np.ndarray, rend_vals: np.ndarray, transform,
                        mascara: np.ndarray, tamano_bloque_km: float = 30.0) -> dict:
    """Agrega celdas válidas por bloque de 30 km calculando medias, centroides y superbloques 2x2."""
    grilla = _asignar_grilla_bloques(transform, mascara, tamano_bloque_km)
    unique_b, first_idx, block_indices = np.unique(
        grilla['block_ids'], return_index=True, return_inverse=True
    )
    counts = np.bincount(block_indices)
    apt_b = np.bincount(block_indices, weights=apt_vals) / counts
    rend_b = np.bincount(block_indices, weights=rend_vals) / counts
    x_centro = np.bincount(block_indices, weights=grilla['xs']) / counts
    y_centro = np.bincount(block_indices, weights=grilla['ys']) / counts
    coords_b = np.column_stack((x_centro, y_centro))
    superblock_b = grilla['superblock_ids'][first_idx]

    return {
        'apt_b': apt_b,
        'rend_b': rend_b,
        'coords_b': coords_b,
        'counts': counts,
        'block_indices': block_indices,
        'superblock_b': superblock_b,
        'n_bloques': int(len(unique_b)),
        'n_superbloques': int(len(np.unique(superblock_b))),
        'mediana_celdas_por_bloque': int(np.median(counts)),
    }


def prueba_t_dutilleul(x: np.ndarray, y: np.ndarray, coords: np.ndarray,
                       usar_rangos: bool = True, n_clases: int | None = None) -> dict:
    """Prueba t modificada de Dutilleul (1993) para correlación espacial con autocorrelación.

    Estima los grados de libertad efectivos (gl_efectivos) corrigiendo la varianza del
    coeficiente de correlación mediante las matrices de autocovarianza espacial por clases
    de distancia (Dutilleul, 1993, Biometrics 49: 305-314).

    Parámetros:
        x, y: Vectores 1D de valores por bloque/sitio.
        coords: Array (n, 2) con coordenadas espaciales en metros (UTM 19S).
        usar_rangos: Si True, transforma x e y a rangos (coherente con Spearman);
                     si False, evalúa sobre las medias crudas (Pearson).
        n_clases: Número de clases de distancia; por defecto usa la regla de Sturges.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    coords_arr = np.asarray(coords, dtype=np.float64)

    valido = np.isfinite(x_arr) & np.isfinite(y_arr) & np.all(np.isfinite(coords_arr), axis=1)
    x_arr = x_arr[valido]
    y_arr = y_arr[valido]
    coords_arr = coords_arr[valido]

    n = len(x_arr)
    gl_clasicos = max(0, n - 2)
    clave_r = 'rho' if usar_rangos else 'r'

    if n < 4:
        return {
            clave_r: 0.0,
            'gl_clasicos': gl_clasicos,
            'gl_efectivos': float(gl_clasicos),
            'factor_reduccion_gl': 1.0,
            't_estadistico': 0.0,
            'p_valor': 1.0,
            'n_clases_distancia': 0,
        }

    vx = rankdata(x_arr, method='average') if usar_rangos else x_arr
    vy = rankdata(y_arr, method='average') if usar_rangos else y_arr

    dx = vx - np.mean(vx)
    dy = vy - np.mean(vy)
    ss_x = float(np.dot(dx, dx))
    ss_y = float(np.dot(dy, dy))
    if ss_x <= 0.0 or ss_y <= 0.0:
        return {
            clave_r: 0.0,
            'gl_clasicos': gl_clasicos,
            'gl_efectivos': float(gl_clasicos),
            'factor_reduccion_gl': 1.0,
            't_estadistico': 0.0,
            'p_valor': 1.0,
            'n_clases_distancia': 0,
        }

    r = float(np.clip(np.dot(dx, dy) / np.sqrt(ss_x * ss_y), -1.0, 1.0))

    dists_pares = pdist(coords_arr, metric='euclidean')
    n_pares = len(dists_pares)
    if n_clases is None:
        n_clases = max(3, int(np.ceil(1.0 + 3.322 * np.log10(max(n_pares, 2)))))

    d_min = float(dists_pares.min())
    d_max = float(dists_pares.max())

    # Sin variación espacial en coordenadas, no es posible estimar lags de distancia
    if d_max <= d_min:
        t_est = r * np.sqrt(gl_clasicos / max(1.0 - r * r, 1e-12))
        p_val = float(2.0 * student_t.sf(abs(t_est), df=max(gl_clasicos, 1)))
        return {
            clave_r: round(r, 4),
            'gl_clasicos': gl_clasicos,
            'gl_efectivos': float(gl_clasicos),
            'factor_reduccion_gl': 1.0,
            't_estadistico': round(float(t_est), 4),
            'p_valor': round(p_val, 6),
            'n_clases_distancia': 1,
        }

    iu, ju = np.triu_indices(n, k=1)
    prod_x = dx[iu] * dx[ju]
    prod_y = dy[iu] * dy[ju]

    # Construir matrices de autocovarianza muestral por clase de distancia
    bordes = np.linspace(d_min, d_max, n_clases + 1)
    sigma_x = np.eye(n, dtype=np.float64) * (ss_x / n)
    sigma_y = np.eye(n, dtype=np.float64) * (ss_y / n)

    clases_activas = 0
    for k in range(n_clases):
        if k < n_clases - 1:
            mask_k = (dists_pares >= bordes[k]) & (dists_pares < bordes[k + 1])
        else:
            mask_k = (dists_pares >= bordes[k]) & (dists_pares <= bordes[k + 1])

        if not np.any(mask_k):
            continue

        clases_activas += 1
        cov_x_k = float(np.mean(prod_x[mask_k]))
        cov_y_k = float(np.mean(prod_y[mask_k]))
        ik = iu[mask_k]
        jk = ju[mask_k]
        sigma_x[ik, jk] = cov_x_k
        sigma_x[jk, ik] = cov_x_k
        sigma_y[ik, jk] = cov_y_k
        sigma_y[jk, ik] = cov_y_k

    # Matriz de centrado B = I - (1/n) 1 1^T aplicada directamente sobre filas/columnas
    sigma_xc = (
        sigma_x
        - sigma_x.mean(axis=1, keepdims=True)
        - sigma_x.mean(axis=0, keepdims=True)
        + sigma_x.mean()
    )
    sigma_yc = (
        sigma_y
        - sigma_y.mean(axis=1, keepdims=True)
        - sigma_y.mean(axis=0, keepdims=True)
        + sigma_y.mean()
    )

    tr_x = float(np.trace(sigma_xc))
    tr_y = float(np.trace(sigma_yc))
    tr_xy = float(np.sum(sigma_xc * sigma_yc))

    if tr_xy <= 1e-15 or tr_x <= 0.0 or tr_y <= 0.0:
        v_hat = float(gl_clasicos)
    else:
        v_hat = (tr_x * tr_y) / tr_xy - 1.0
        # Resguardo numérico en muestras pequeñas o con lags ruidosos
        v_hat = max(v_hat, 1.0)

    if abs(r) >= 1.0 - 1e-12:
        t_est = np.sign(r) * np.inf
        p_val = 0.0
    else:
        t_est = r * np.sqrt(v_hat / (1.0 - r * r))
        p_val = float(2.0 * student_t.sf(abs(t_est), df=v_hat))

    return {
        clave_r: round(r, 4),
        'gl_clasicos': gl_clasicos,
        'gl_efectivos': round(float(v_hat), 2),
        'factor_reduccion_gl': round(float(v_hat / gl_clasicos), 4) if gl_clasicos > 0 else 1.0,
        't_estadistico': round(float(t_est), 4) if np.isfinite(t_est) else float(t_est),
        'p_valor': round(p_val, 6),
        'n_clases_distancia': clases_activas,
    }


def _spearman_rapido(x: np.ndarray, y: np.ndarray, fallback_rho: float) -> float:
    """Calcula rho de Spearman sin el overhead de p-valor ni validaciones repetidas."""
    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom <= 0.0:
        return float(fallback_rho)
    return float(np.clip(np.dot(dx, dy) / denom, -1.0, 1.0))


def _evaluar_lote_grupos(grupos_data: list, lotes_idx: np.ndarray, fallback_rho: float) -> np.ndarray:
    """Evalúa un lote de réplicas bootstrap sobre grupos espaciales pre-indexados."""
    out = np.empty(len(lotes_idx), dtype=np.float64)
    for k, sampled_gi in enumerate(lotes_idx):
        a_parts = [grupos_data[gi][0] for gi in sampled_gi if len(grupos_data[gi][0]) > 0]
        r_parts = [grupos_data[gi][1] for gi in sampled_gi if len(grupos_data[gi][1]) > 0]
        if a_parts:
            out[k] = _spearman_rapido(np.concatenate(a_parts), np.concatenate(r_parts), fallback_rho)
        else:
            out[k] = fallback_rho
    return out


def _bootstrap_spearman_por_grupos(apt_vals: np.ndarray, rend_vals: np.ndarray,
                                   group_ids: np.ndarray, n_bootstrap: int,
                                   rng: np.random.Generator, fallback_rho: float) -> dict:
    """Ejecuta bootstrap espacial remuestreando con reemplazo grupos espaciales completos."""
    unique_g, inv_g = np.unique(group_ids, return_inverse=True)
    n_grupos = len(unique_g)

    if n_grupos < 2:
        return {
            'n_grupos': int(n_grupos),
            'ic_95_inferior': round(float(fallback_rho), 4),
            'ic_95_superior': round(float(fallback_rho), 4),
            'std_bootstrap': 0.0,
        }

    # Si cada grupo contiene exactamente 1 observación (p. ej. bloques de 30 km), se indexa directo
    if n_grupos == len(apt_vals):
        boot_rhos = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.choice(n_grupos, size=n_grupos, replace=True)
            boot_rhos[i] = _spearman_rapido(apt_vals[idx], rend_vals[idx], fallback_rho)
    else:
        grupos_data = [
            (apt_vals[inv_g == gi], rend_vals[inv_g == gi])
            for gi in range(n_grupos)
        ]
        matriz_muestreos = rng.integers(0, n_grupos, size=(n_bootstrap, n_grupos))
        # Paralelizar solo cuando el volumen por réplica justifica el arranque de workers
        if len(apt_vals) > 5000 and n_bootstrap >= 500:
            n_workers = max(1, min(os.cpu_count() or 1, 12))
            lotes = np.array_split(matriz_muestreos, n_workers * 2)
            partes = joblib.Parallel(n_jobs=n_workers)(
                joblib.delayed(_evaluar_lote_grupos)(grupos_data, lote, fallback_rho)
                for lote in lotes if len(lote) > 0
            )
            boot_rhos = np.concatenate(partes)
        else:
            boot_rhos = _evaluar_lote_grupos(grupos_data, matriz_muestreos, fallback_rho)

    ic = np.nanpercentile(boot_rhos, [2.5, 97.5])
    return {
        'n_grupos': int(n_grupos),
        'ic_95_inferior': round(float(ic[0]), 4),
        'ic_95_superior': round(float(ic[1]), 4),
        'std_bootstrap': round(float(np.nanstd(boot_rhos)), 4),
    }


def _evaluar_ambito_desacople(apt_vals: np.ndarray, rend_vals: np.ndarray, transform,
                              mascara: np.ndarray, tamano_bloque_km: float,
                              n_bootstrap: int, submuestra_max: int,
                              rng_clasico: np.random.Generator,
                              rng_super: np.random.Generator) -> dict:
    """Calcula correlaciones por celda y por bloque (incluyendo Dutilleul y superbloques 2x2)."""
    rho_celda, pval_celda = spearmanr(apt_vals, rend_vals)
    agregado = _agregar_por_bloques(apt_vals, rend_vals, transform, mascara, tamano_bloque_km)

    apt_b = agregado['apt_b']
    rend_b = agregado['rend_b']
    rho_bloque, pval_bloque = spearmanr(apt_b, rend_b)

    # Bootstrap de bloques de 30 km (unidad = bloque individual)
    boot_bloque = _bootstrap_spearman_por_grupos(
        apt_b, rend_b, np.arange(agregado['n_bloques']), n_bootstrap, rng_clasico, rho_bloque
    )

    # Cluster/Block bootstrap para correlación por celda (submuestreo uniforme para eficiencia)
    sub_paso = max(1, len(apt_vals) // submuestra_max)
    sub_idx = np.arange(0, len(apt_vals), sub_paso)
    boot_celda = _bootstrap_spearman_por_grupos(
        apt_vals[sub_idx],
        rend_vals[sub_idx],
        agregado['block_indices'][sub_idx],
        n_bootstrap,
        rng_clasico,
        rho_celda,
    )

    # Control espacial 1: Prueba t modificada de Dutilleul (1993) sobre rangos y medias crudas
    dutilleul_spearman = prueba_t_dutilleul(
        apt_b, rend_b, agregado['coords_b'], usar_rangos=True
    )
    dutilleul_pearson = prueba_t_dutilleul(
        apt_b, rend_b, agregado['coords_b'], usar_rangos=False
    )

    # Control espacial 2: Bootstrap espacial con superbloques 2x2 (60 km)
    boot_super = _bootstrap_spearman_por_grupos(
        apt_b, rend_b, agregado['superblock_b'], n_bootstrap, rng_super, rho_bloque
    )

    return {
        'n_celdas': int(len(apt_vals)),
        'n_bloques_30km': agregado['n_bloques'],
        'n_superbloques_60km': agregado['n_superbloques'],
        'mediana_celdas_por_bloque': agregado['mediana_celdas_por_bloque'],
        'spearman_celda': {
            'rho': round(float(rho_celda), 4),
            'p_valor': float(pval_celda),
            'ic_95_inferior': boot_celda['ic_95_inferior'],
            'ic_95_superior': boot_celda['ic_95_superior'],
            'std_bootstrap': boot_celda['std_bootstrap'],
        },
        'spearman_bloque': {
            'rho': round(float(rho_bloque), 4),
            'p_valor': round(float(pval_bloque), 6),
            'ic_95_inferior': boot_bloque['ic_95_inferior'],
            'ic_95_superior': boot_bloque['ic_95_superior'],
            'std_bootstrap': boot_bloque['std_bootstrap'],
            'dutilleul_spearman': dutilleul_spearman,
            'dutilleul_pearson': dutilleul_pearson,
            'bootstrap_superbloques_60km': {
                'n_superbloques': boot_super['n_grupos'],
                'tamano_superbloque_km': float(tamano_bloque_km * 2.0),
                'ic_95_inferior': boot_super['ic_95_inferior'],
                'ic_95_superior': boot_super['ic_95_superior'],
                'std_bootstrap': boot_super['std_bootstrap'],
            },
        },
    }


def calcular_estadisticas_desacople(cruce: dict, tamano_bloque_km: float = 30.0,
                                   n_bootstrap: int = 10000, random_state: int = 42) -> dict:
    """Calcula todas las métricas de desacople requeridas:

    - Rendimiento medio y mediano regional vs. celdas aptas.
    - Correlación de Spearman por celda y por bloque de 30 km.
    - Intervalo de confianza por Bootstrap espacial de bloques al 95%.
    - Control espacial 1: Prueba t modificada de Dutilleul (rangos/Spearman y medias crudas/Pearson).
    - Control espacial 2: Bootstrap espacial con superbloques de 2x2 bloques (60 km).
    """
    aptitud = cruce['aptitud']
    rendimiento = cruce['rendimiento']
    base_valida = cruce['base_valida']
    aptas = cruce['aptas']
    transform = cruce['transform']
    stats_base = cruce['stats']

    rng_clasico = np.random.default_rng(random_state)
    # Flujo independiente para superbloques 2x2 preservando reproducibilidad del flujo base
    rng_super = np.random.default_rng(np.random.SeedSequence([random_state, 60]))

    region = _evaluar_ambito_desacople(
        aptitud[base_valida],
        rendimiento[base_valida],
        transform,
        base_valida,
        tamano_bloque_km,
        n_bootstrap,
        submuestra_max=200_000,
        rng_clasico=rng_clasico,
        rng_super=rng_super,
    )

    aptas_res = _evaluar_ambito_desacople(
        aptitud[aptas],
        rendimiento[aptas],
        transform,
        aptas,
        tamano_bloque_km,
        n_bootstrap,
        submuestra_max=100_000,
        rng_clasico=rng_clasico,
        rng_super=rng_super,
    )

    resultados = {
        'resumen_rendimiento': stats_base,
        'tamano_bloque_km': tamano_bloque_km,
        'tamano_superbloque_km': float(tamano_bloque_km * 2.0),
        'n_bootstrap': n_bootstrap,
        'random_state': random_state,
        'region': region,
        'aptas': aptas_res,
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
            'metrica': 'mediana_celdas_por_bloque_30km_region',
            'ambito': 'Region Completa',
            'valor': reg['mediana_celdas_por_bloque'],
            'unidad': 'celdas/bloque',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Mediana de celdas válidas por bloque de 30 km en la región',
        },
        {
            'categoria': 'Estructura Espacial',
            'metrica': 'n_superbloques_espaciales_60km_region',
            'ambito': 'Region Completa',
            'valor': reg['n_superbloques_60km'],
            'unidad': 'superbloques',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número de superbloques 2x2 (60 km) con al menos un bloque válido',
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
            'categoria': 'Estructura Espacial',
            'metrica': 'n_superbloques_espaciales_60km_aptas',
            'ambito': 'Zonas Aptas',
            'valor': apt['n_superbloques_60km'],
            'unidad': 'superbloques',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número de superbloques 2x2 (60 km) con al menos un bloque apto',
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
            'descripcion': 'Correlación de Spearman agregada a nivel de bloque (SBCV 30 km, bootstrap por bloques de 30 km)',
        },
        {
            'categoria': 'Control Espacial (Dutilleul)',
            'metrica': 'spearman_dutilleul_bloques_30km_regional',
            'ambito': 'Agregada por Bloques de 30 km (Región)',
            'valor': reg['spearman_bloque']['dutilleul_spearman']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': reg['spearman_bloque']['dutilleul_spearman']['p_valor'],
            'descripcion': (
                f"Prueba t modificada de Dutilleul sobre rangos (Spearman): "
                f"gl_clasicos={reg['spearman_bloque']['dutilleul_spearman']['gl_clasicos']} -> "
                f"gl_efectivos={reg['spearman_bloque']['dutilleul_spearman']['gl_efectivos']}"
            ),
        },
        {
            'categoria': 'Control Espacial (Dutilleul)',
            'metrica': 'pearson_dutilleul_bloques_30km_regional',
            'ambito': 'Agregada por Bloques de 30 km (Región)',
            'valor': reg['spearman_bloque']['dutilleul_pearson']['r'],
            'unidad': 'r',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': reg['spearman_bloque']['dutilleul_pearson']['p_valor'],
            'descripcion': (
                f"Prueba t modificada de Dutilleul sobre medias crudas (Pearson): "
                f"gl_clasicos={reg['spearman_bloque']['dutilleul_pearson']['gl_clasicos']} -> "
                f"gl_efectivos={reg['spearman_bloque']['dutilleul_pearson']['gl_efectivos']}"
            ),
        },
        {
            'categoria': 'Control Espacial (Superbloques 60 km)',
            'metrica': 'spearman_bootstrap_superbloques_60km_regional',
            'ambito': 'Agregada por Bloques 30 km, IC en Superbloques 2x2 (Región)',
            'valor': reg['spearman_bloque']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': reg['spearman_bloque']['bootstrap_superbloques_60km']['ic_95_inferior'],
            'ic_95_superior': reg['spearman_bloque']['bootstrap_superbloques_60km']['ic_95_superior'],
            'p_valor': '',
            'descripcion': (
                f"Bootstrap espacial por superbloques 2x2 de 60 km "
                f"(G={reg['spearman_bloque']['bootstrap_superbloques_60km']['n_superbloques']})"
            ),
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_celda_aptas',
            'ambito': 'Por Celda (Subconjunto Aptas)',
            'valor': apt['spearman_celda']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': apt['spearman_celda']['ic_95_inferior'],
            'ic_95_superior': apt['spearman_celda']['ic_95_superior'],
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
            'categoria': 'Control Espacial (Dutilleul)',
            'metrica': 'spearman_dutilleul_bloques_30km_aptas',
            'ambito': 'Agregada por Bloques de 30 km (Aptas)',
            'valor': apt['spearman_bloque']['dutilleul_spearman']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': apt['spearman_bloque']['dutilleul_spearman']['p_valor'],
            'descripcion': (
                f"Prueba t modificada de Dutilleul sobre rangos (Spearman, aptas): "
                f"gl_clasicos={apt['spearman_bloque']['dutilleul_spearman']['gl_clasicos']} -> "
                f"gl_efectivos={apt['spearman_bloque']['dutilleul_spearman']['gl_efectivos']}"
            ),
        },
        {
            'categoria': 'Control Espacial (Dutilleul)',
            'metrica': 'pearson_dutilleul_bloques_30km_aptas',
            'ambito': 'Agregada por Bloques de 30 km (Aptas)',
            'valor': apt['spearman_bloque']['dutilleul_pearson']['r'],
            'unidad': 'r',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': apt['spearman_bloque']['dutilleul_pearson']['p_valor'],
            'descripcion': (
                f"Prueba t modificada de Dutilleul sobre medias crudas (Pearson, aptas): "
                f"gl_clasicos={apt['spearman_bloque']['dutilleul_pearson']['gl_clasicos']} -> "
                f"gl_efectivos={apt['spearman_bloque']['dutilleul_pearson']['gl_efectivos']}"
            ),
        },
        {
            'categoria': 'Control Espacial (Superbloques 60 km)',
            'metrica': 'spearman_bootstrap_superbloques_60km_aptas',
            'ambito': 'Agregada por Bloques 30 km, IC en Superbloques 2x2 (Aptas)',
            'valor': apt['spearman_bloque']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': apt['spearman_bloque']['bootstrap_superbloques_60km']['ic_95_inferior'],
            'ic_95_superior': apt['spearman_bloque']['bootstrap_superbloques_60km']['ic_95_superior'],
            'p_valor': '',
            'descripcion': (
                f"Bootstrap espacial por superbloques 2x2 de 60 km en aptas "
                f"(G={apt['spearman_bloque']['bootstrap_superbloques_60km']['n_superbloques']})"
            ),
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
