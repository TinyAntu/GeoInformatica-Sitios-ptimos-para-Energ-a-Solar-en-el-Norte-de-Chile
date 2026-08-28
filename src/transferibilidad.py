"""Generalización del modelo: aplicarlo sobre una región que no participó del entrenamiento.

Este módulo NO reentrena nada. Carga `model_rf.pkl` tal cual, arma la grilla de features de
otra región con la misma función canónica que usa el resto del pipeline
(`explainability_spatial._construir_features`) y mide allí Recall@K / Precisión@K contra las
plantas reales de esa región.

Junto con el LORO ya calculado en src/metricas_topk.py quedan tres puntos sobre la misma
métrica, y la caída entre ellos es el resultado:

    A  entrenado en Antofagasta+Atacama, evaluado en Atacama   -> in-sample (cota optimista)
    B  entrenado en Antofagasta,          evaluado en Atacama   -> transferencia intra-norte
    C  entrenado en Antofagasta+Atacama, evaluado en Coquimbo  -> fuera del dominio

Advertencia imprescindible para leer C: un Random Forest **no extrapola**. Fuera del rango de
valores visto en entrenamiento devuelve el valor de la hoja más cercana, así que una variable
desplazada no produce un error visible sino una predicción silenciosamente saturada. Por eso
`diagnostico_covariate_shift` se calcula siempre y se reporta junto a las métricas: sin él, un
mal resultado no se puede atribuir a geografía frente a desplazamiento de covariables.
"""

import os

import numpy as np
import geopandas as gpd

from src.features import FEATURES
from src.metricas_topk import (
    KS_DEFECTO, _ruta, cargar_plantas, rasterizar_plantas,
    _plantas_a_filas_columnas, calcular_metricas_topk,
)

# ESRI trunca los nombres de columna del shapefile a 10 caracteres (ver AGENTS.md).
RENAME_ESRI = {
    "dist_trans": "dist_transmision",
    "dist_almac": "dist_almacen",
    "dist_subs": "dist_subestaciones",
}


def cargar_muestras_entrenamiento(config, directorio_raiz):
    """Carga el dataset de entrenamiento con los nombres de columna restaurados."""
    ruta = _ruta(directorio_raiz, config['paths']['results']['dataset_ml'])
    g = gpd.read_file(ruta)
    return g.rename(columns={k: v for k, v in RENAME_ESRI.items() if k in g.columns})


def diagnostico_covariate_shift(muestras, X_zona):
    """Compara la distribución de cada feature entre entrenamiento y la zona evaluada.

    Devuelve, por variable, el rango y los cuartiles de ambas poblaciones y —lo importante—
    qué fracción de los píxeles de la zona cae FUERA del rango visto en entrenamiento. Ese
    porcentaje es el que acota cuánto de la predicción es interpolación legítima y cuánto es
    el árbol devolviendo el valor de su hoja extrema.
    """
    salida = {}
    for i, f in enumerate(FEATURES):
        tr = muestras[f].astype(float).dropna().values
        zo = X_zona[:, i].astype(float)
        zo = zo[np.isfinite(zo)]
        if tr.size == 0 or zo.size == 0:
            continue
        lo, hi = float(tr.min()), float(tr.max())
        debajo = float((zo < lo).mean())
        encima = float((zo > hi).mean())
        salida[f] = {
            'entrenamiento': {
                'min': round(lo, 4), 'p25': round(float(np.percentile(tr, 25)), 4),
                'mediana': round(float(np.median(tr)), 4),
                'p75': round(float(np.percentile(tr, 75)), 4), 'max': round(hi, 4),
                'media': round(float(tr.mean()), 4),
            },
            'zona': {
                'min': round(float(zo.min()), 4), 'p25': round(float(np.percentile(zo, 25)), 4),
                'mediana': round(float(np.median(zo)), 4),
                'p75': round(float(np.percentile(zo, 75)), 4), 'max': round(float(zo.max()), 4),
                'media': round(float(zo.mean()), 4),
            },
            'pct_fuera_de_rango': round(100.0 * (debajo + encima), 2),
            'pct_por_debajo_del_min': round(100.0 * debajo, 2),
            'pct_por_encima_del_max': round(100.0 * encima, 2),
            # Diferencia de medias en desviaciones estándar del entrenamiento: mide el
            # desplazamiento en unidades comparables entre variables de escalas distintas.
            'desplazamiento_medias_sd': (
                round(float((zo.mean() - tr.mean()) / tr.std()), 3) if tr.std() > 0 else None
            ),
        }
    return salida


def evaluar_zona(config, directorio_raiz, modelo, zona, resolucion_m,
                 ks=KS_DEFECTO, ha_por_mw=None):
    """Puntúa `modelo` sobre la grilla de `zona` y calcula las métricas top-K allí.

    Devuelve (resultado, X, extras) donde `X` es la matriz de features de la zona —necesaria
    para el diagnóstico de covariate shift— y `extras` trae la grilla para escribir rásters.
    """
    from src.explainability_spatial import _construir_features

    crs, transform, width, height, valido, X = _construir_features(
        config, resolucion_m, zona=zona)
    grid_shape = (height, width)
    if X.shape[0] == 0:
        raise ValueError(f"No hay píxeles válidos en la zona {zona['regiones']}.")

    prob = np.full(grid_shape, np.nan, dtype=np.float32)
    prob[valido] = modelo.predict_proba(X)[:, 1]
    area_px_m2 = abs(transform[0] * transform[4])

    plantas = cargar_plantas(config, directorio_raiz, crs,
                             regiones_estudio=zona['regiones'])
    mask_plantas = rasterizar_plantas(plantas, grid_shape, transform)
    plantas_rc = _plantas_a_filas_columnas(plantas, transform, grid_shape)

    mask_buffer = None
    if ha_por_mw:
        mask_buffer = rasterizar_plantas(
            cargar_plantas(config, directorio_raiz, crs, ha_por_mw,
                           regiones_estudio=zona['regiones']),
            grid_shape, transform)

    res = calcular_metricas_topk(prob, valido, mask_plantas, plantas_rc, area_px_m2, ks,
                                 mask_buffer=mask_buffer)
    res['resolucion_m'] = round(float(np.sqrt(area_px_m2)), 1)
    res['regiones'] = list(zona['regiones'])
    res['modo_filtro_region'] = zona.get('modo_region', 'exacto')
    res['potencia_total_mw'] = round(float(plantas['POTENCIAMW'].astype(float).sum()), 1)
    res['area_zona_km2'] = round(float(valido.sum()) * area_px_m2 / 1e6, 1)

    extras = {'crs': crs, 'transform': transform, 'shape': grid_shape,
              'valido': valido, 'prob': prob}
    return res, X, extras


def escribir_mapa_aptitud(extras, ruta, nodata=-9999.0):
    """Guarda el mapa de probabilidad de la zona como GeoTIFF."""
    import rasterio

    os.makedirs(os.path.dirname(ruta) or '.', exist_ok=True)
    alto, ancho = extras['shape']
    datos = np.where(np.isfinite(extras['prob']), extras['prob'], nodata).astype('float32')
    meta = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'nodata': nodata,
        'crs': extras['crs'], 'transform': extras['transform'],
        'width': ancho, 'height': alto,
    }
    with rasterio.open(ruta, 'w', **meta) as dst:
        dst.write(datos, 1)
    return ruta
