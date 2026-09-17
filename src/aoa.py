"""Área de Aplicabilidad (AOA) e Índice de Disimilitud (DI) — Meyer & Pebesma (2021).

Implementa la metodología formal de detección de extrapolación geográfica para modelos
espaciales (Methods in Ecology and Evolution, 12(9), 1620-1633):
  1) Pondera las distancias euclidianas multidimensionales según la importancia relativa de
     las covariables en el modelo entrenado.
  2) Estandariza por la varianza de cada variable en los datos de entrenamiento.
  3) Calcula el Índice de Disimilitud (DI) de cada píxel respecto al conjunto de entrenamiento.
  4) Deriva el umbral de aplicabilidad (DI_thresh = Q3 + 1.5 * IQR) a partir de los vecinos más
     cercanos en validación cruzada.
  5) Clasifica el territorio en una máscara binaria (AOA = 1 si DI <= DI_thresh; 0 si extrapola).
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import rasterio
from sklearn.neighbors import NearestNeighbors

from src.features import FEATURES

NODATA_DI = -9999.0
NODATA_AOA = 255


@dataclass
class AjusteAOA:
    """Parámetros de referencia del espacio de entrenamiento para AOA."""
    media: np.ndarray             # (p,)
    desv_std: np.ndarray          # (p,)
    pesos: np.ndarray             # (p,)
    X_train_ponderado: np.ndarray # (n, p)
    dist_media_train: float       # d_barra (distancia media entre muestras de entrenamiento)
    umbral_di: float              # DI_thresh (Q3 + 1.5 * IQR de DI en entrenamiento)
    nn_model: NearestNeighbors    # Modelo 1-NN ajustado sobre X_train_ponderado
    di_train_stats: dict          # Estadísticas de DI en entrenamiento


def extraer_pesos_modelo(modelo, features: list[str] = FEATURES) -> np.ndarray:
    """Extrae y normaliza los pesos relativos de importancia de las covariables.

    Si el modelo tiene `feature_importances_` (Random Forest), se normalizan a suma 1.
    Si no están disponibles, se asignan pesos uniformes (1/p).
    """
    if hasattr(modelo, 'feature_importances_'):
        raw_imp = np.array(modelo.feature_importances_, dtype=np.float64)
    else:
        raw_imp = np.ones(len(features), dtype=np.float64)

    total = raw_imp.sum()
    if total > 0:
        pesos = raw_imp / total
    else:
        pesos = np.full(len(features), 1.0 / len(features), dtype=np.float64)
    return pesos


def ajustar_espacio_aoa(
    X_train: np.ndarray,
    pesos: Optional[np.ndarray] = None,
    grupos_cv: Optional[np.ndarray] = None,
) -> AjusteAOA:
    """Ajusta el espacio estandarizado y ponderado a partir de los datos de entrenamiento.

    Calcula la distancia 1-NN para cada muestra de entrenamiento y deriva el umbral AOA
    según la regla de Tukey (Q3 + 1.5 * IQR) de Meyer & Pebesma (2021).
    """
    X_tr = np.asarray(X_train, dtype=np.float64)
    n_samples, n_features = X_tr.shape

    if pesos is None:
        pesos = np.full(n_features, 1.0 / n_features, dtype=np.float64)
    else:
        pesos = np.asarray(pesos, dtype=np.float64)
        if pesos.sum() > 0:
            pesos = pesos / pesos.sum()

    # 1. Estandarización por desviación estándar del entrenamiento
    media = np.nanmean(X_tr, axis=0)
    desv_std = np.nanstd(X_tr, axis=0)
    desv_std = np.where(desv_std == 0, 1.0, desv_std)  # Evitar división por cero

    # 2. Ponderación por raíz cuadrada de pesos: dist = sqrt(sum w_j * ((x_j - mu_j)/sigma_j)^2)
    X_tr_norm = (X_tr - media) / desv_std
    X_tr_pond = X_tr_norm * np.sqrt(pesos)

    # 3. Distancias 1-NN en el conjunto de entrenamiento
    if grupos_cv is not None and len(np.unique(grupos_cv)) > 1:
        # Si hay grupos de CV (p. ej. bloques espaciales), la distancia de cada punto
        # se mide contra el vecino más cercano de OTRO bloque para reflejar CV espacial.
        dists_train = []
        grupos_unicos = np.unique(grupos_cv)
        for g in grupos_unicos:
            idx_test = np.where(grupos_cv == g)[0]
            idx_train = np.where(grupos_cv != g)[0]
            if len(idx_train) == 0:
                continue
            nn_cv = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_tr_pond[idx_train])
            d, _ = nn_cv.kneighbors(X_tr_pond[idx_test])
            dists_train.extend(d.ravel().tolist())
        dists_train = np.array(dists_train, dtype=np.float64)
    else:
        # Si no hay grupos, se toma el vecino más cercano distinto del propio punto (k=2)
        nn_tr = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_tr_pond)
        dists, _ = nn_tr.kneighbors(X_tr_pond)
        dists_train = dists[:, 1]  # índice 1 = vecino más cercano excluyéndose a sí mismo

    dist_media_train = float(np.mean(dists_train)) if len(dists_train) > 0 else 1.0
    if dist_media_train == 0:
        dist_media_train = 1.0

    # 4. Índice de Disimilitud (DI) en entrenamiento
    di_train = dists_train / dist_media_train

    # 5. Umbral de Tukey: Q3 + 1.5 * IQR
    q25 = float(np.percentile(di_train, 25))
    q75 = float(np.percentile(di_train, 75))
    iqr = q75 - q25
    umbral_di = float(q75 + 1.5 * iqr)

    # 6. Modelo 1-NN global para predicción de nuevos píxeles
    nn_global = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_tr_pond)

    di_train_stats = {
        'n_muestras': int(n_samples),
        'dist_media_train': round(dist_media_train, 5),
        'di_min': round(float(di_train.min()), 4),
        'di_q25': round(q25, 4),
        'di_mediana': round(float(np.median(di_train)), 4),
        'di_q75': round(q75, 4),
        'di_max': round(float(di_train.max()), 4),
        'di_iqr': round(float(iqr), 4),
        'umbral_aoa': round(umbral_di, 4),
    }

    return AjusteAOA(
        media=media,
        desv_std=desv_std,
        pesos=pesos,
        X_train_ponderado=X_tr_pond,
        dist_media_train=dist_media_train,
        umbral_di=umbral_di,
        nn_model=nn_global,
        di_train_stats=di_train_stats,
    )


def calcular_di_y_aoa(
    ajuste: AjusteAOA,
    X_pred: np.ndarray,
    tamano_lote: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula el Índice de Disimilitud (DI) y la máscara de AOA para una matriz de píxeles.

    Procesa por lotes (chunking) para controlar el uso de memoria en grillas grandes.
    Retorna (di_array, aoa_array) donde `di_array` es Float64 continuo y `aoa_array` es Bool.
    """
    X_p = np.asarray(X_pred, dtype=np.float64)
    n_pred = len(X_p)
    if n_pred == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

    di_resultado = np.empty(n_pred, dtype=np.float64)

    for i in range(0, n_pred, tamano_lote):
        fin = min(i + tamano_lote, n_pred)
        lote = X_p[i:fin]

        # Normalizar y ponderar
        lote_norm = (lote - ajuste.media) / ajuste.desv_std
        lote_pond = lote_norm * np.sqrt(ajuste.pesos)

        # Distancia 1-NN al conjunto de entrenamiento
        dists, _ = ajuste.nn_model.kneighbors(lote_pond)
        di_resultado[i:fin] = dists.ravel() / ajuste.dist_media_train

    aoa_resultado = di_resultado <= ajuste.umbral_di
    return di_resultado, aoa_resultado


def escribir_rasters_aoa(
    di_valores: np.ndarray,
    aoa_valores: np.ndarray,
    valido: np.ndarray,
    base_meta: dict,
    dir_salida: str,
) -> Dict[str, str]:
    """Reconstruye y guarda los rásters 2D de DI y AOA en formato GeoTIFF (EPSG:32719)."""
    os.makedirs(dir_salida, exist_ok=True)
    alto, ancho = base_meta['height'], base_meta['width']

    # 1. Ráster DI continuo (Float32)
    grid_di = np.full((alto, ancho), NODATA_DI, dtype=np.float32)
    grid_di[valido] = di_valores.astype(np.float32)

    meta_di = dict(base_meta, driver='GTiff', dtype='float32', count=1, nodata=NODATA_DI)
    ruta_di = os.path.join(dir_salida, 'di.tif')
    with rasterio.open(ruta_di, 'w', **meta_di) as dst:
        dst.write(grid_di, 1)

    # 2. Ráster AOA binario (UInt8: 1=dentro AOA, 0=fuera AOA, 255=nodata)
    grid_aoa = np.full((alto, ancho), NODATA_AOA, dtype=np.uint8)
    grid_aoa[valido] = aoa_valores.astype(np.uint8)

    meta_aoa = dict(base_meta, driver='GTiff', dtype='uint8', count=1, nodata=NODATA_AOA)
    ruta_aoa = os.path.join(dir_salida, 'aoa.tif')
    with rasterio.open(ruta_aoa, 'w', **meta_aoa) as dst:
        dst.write(grid_aoa, 1)

    return {'di': ruta_di, 'aoa': ruta_aoa}


def evaluar_aoa_zona(
    ajuste_aoa: AjusteAOA,
    X_zona: np.ndarray,
    valido: np.ndarray,
    base_meta: dict,
    dir_salida: Optional[str] = None,
    plantas_rc: Optional[list] = None,
) -> dict:
    """Evalúa formalmente el Área de Aplicabilidad sobre la grilla de una zona geográfica.

    Devuelve un resumen con estadísticas de DI, porcentaje dentro/fuera de AOA y,
    si se proveen los centroides de plantas solares, la tasa de plantas dentro de AOA.
    """
    if len(X_zona) == 0 or not valido.any():
        raise ValueError("No hay píxeles válidos para evaluar AOA en la zona.")

    di_vals, aoa_vals = calcular_di_y_aoa(ajuste_aoa, X_zona)

    n_valido = int(len(X_zona))
    n_dentro = int(aoa_vals.sum())
    n_fuera = n_valido - n_dentro
    pct_dentro = round(100.0 * n_dentro / n_valido, 2) if n_valido > 0 else 0.0
    pct_fuera = round(100.0 * n_fuera / n_valido, 2) if n_valido > 0 else 0.0

    # Estadísticas de DI en la zona evaluada
    di_stats_zona = {
        'min': round(float(di_vals.min()), 4),
        'p25': round(float(np.percentile(di_vals, 25)), 4),
        'mediana': round(float(np.median(di_vals)), 4),
        'p75': round(float(np.percentile(di_vals, 75)), 4),
        'p95': round(float(np.percentile(di_vals, 95)), 4),
        'max': round(float(di_vals.max()), 4),
        'media': round(float(di_vals.mean()), 4),
        'std': round(float(di_vals.std()), 4),
    }

    # Evaluación de plantas reales dentro del AOA (si se proporcionan centroides)
    plantas_info = None
    if plantas_rc is not None and len(plantas_rc) > 0:
        # Mapear valores AOA a grilla 2D
        alto, ancho = base_meta['height'], base_meta['width']
        aoa_grid = np.zeros((alto, ancho), dtype=bool)
        aoa_grid[valido] = aoa_vals

        plantas_en_valido = [(f, c) for f, c in plantas_rc if valido[f, c]]
        n_plantas_total = len(plantas_en_valido)
        plantas_en_aoa = sum(1 for f, c in plantas_en_valido if aoa_grid[f, c])
        pct_plantas_aoa = round(100.0 * plantas_en_aoa / n_plantas_total, 2) if n_plantas_total > 0 else 0.0

        plantas_info = {
            'n_plantas_evaluadas': n_plantas_total,
            'n_plantas_dentro_aoa': plantas_en_aoa,
            'n_plantas_fuera_aoa': n_plantas_total - plantas_en_aoa,
            'pct_plantas_dentro_aoa': pct_plantas_aoa,
        }

    rutas_rasters = None
    if dir_salida:
        rutas_rasters = escribir_rasters_aoa(di_vals, aoa_vals, valido, base_meta, dir_salida)

    return {
        'umbral_aoa_entrenamiento': ajuste_aoa.umbral_di,
        'n_pixeles_evaluados': n_valido,
        'n_pixeles_dentro_aoa': n_dentro,
        'n_pixeles_fuera_aoa': n_fuera,
        'pct_dentro_aoa': pct_dentro,
        'pct_fuera_aoa': pct_fuera,
        'estadisticas_di_zona': di_stats_zona,
        'estadisticas_di_entrenamiento': ajuste_aoa.di_train_stats,
        'plantas_en_aoa': plantas_info,
        'rasters': rutas_rasters,
    }
