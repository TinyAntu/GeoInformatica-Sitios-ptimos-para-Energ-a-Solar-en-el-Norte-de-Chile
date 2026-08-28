"""Cruce del mapa de aptitud (RF) con el mapa de rendimiento físico (solarpv-rs).

Convierte *dónde es apto* (probabilidad 0–1 del Random Forest) en *dónde conviene y
cuánto produce* (kWh/kWp/año). Produce dos salidas complementarias, ambas en EPSG:32719
y sobre la grilla del mapa de aptitud:

  (a) `rendimiento_en_aptas.tif`  — rendimiento físico enmascarado a las zonas aptas
      (probabilidad ≥ umbral). Responde: "de los sitios que el modelo aprueba, ¿cuánto
      produce cada uno?".
  (b) `aptitud_x_rendimiento.tif` — ranking combinado 0–1 que exige *ambas* cosas:
      ser apto Y producir mucho. Responde: "¿cuáles son los mejores sitios considerando
      logística/restricciones y producción real a la vez?".

Decisión de la fórmula del ranking (b):
    yield_norm = clip( (rendimiento − p1) / (p99 − p1), 0, 1 )    # normalización robusta
    score      = probabilidad_RF × yield_norm

Se usa el producto (no un promedio ponderado) porque queremos que un sitio puntúe alto
solo si es apto *y* de alto rendimiento: si cualquiera de los dos es bajo, el score cae.
Las zonas de exclusión ya vienen con probabilidad 0.0 en el mapa de aptitud, así que su
score queda 0 automáticamente. La normalización usa percentiles 1/99 para que un único
píxel extremo no comprima la escala. Ambas salidas preservan nodata = -9999.0.
"""

import os
import json

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr

NODATA = -9999.0


def _alinear_rendimiento(rendimiento_path, base):
    """Reproyecta el rendimiento a la grilla del mapa de aptitud (bilinear).

    `base` es el dataset de aptitud abierto (define CRS, transform, shape de salida).
    Devuelve un array float32 con NaN donde no hay dato.
    """
    destino = np.full((base.height, base.width), np.nan, dtype=np.float32)
    with rasterio.open(rendimiento_path) as ren:
        reproject(
            source=rasterio.band(ren, 1),
            destination=destino,
            src_transform=ren.transform, src_crs=ren.crs, src_nodata=ren.nodata,
            dst_transform=base.transform, dst_crs=base.crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    return destino


def cruzar(aptitud_path, rendimiento_path, umbral,
           out_en_aptas, out_ranking, out_json):
    """Genera las dos salidas del cruce y un JSON de estadísticas. Devuelve el dict.

    - aptitud_path: mapa_probabilidad_aptitud.tif (nodata -9999.0; 0.0 = exclusión).
    - rendimiento_path: *_specific_yield.tif del motor (kWh/kWp/año).
    - umbral: probabilidad mínima para considerar una celda "apta" (p. ej. 0.70).
    """
    with rasterio.open(aptitud_path) as apt:
        aptitud = apt.read(1).astype(np.float32)
        meta = apt.meta.copy()
        rendimiento = _alinear_rendimiento(rendimiento_path, apt)

    # Celdas con dato real en ambas capas (excluye nodata de aptitud y NaN de rendimiento).
    base_valida = (aptitud != NODATA) & np.isfinite(aptitud) & np.isfinite(rendimiento)
    if not base_valida.any():
        raise ValueError("El cruce no tiene celdas válidas comunes entre aptitud y rendimiento. "
                         "Revisa que ambos rasters cubran la misma zona (EPSG:32719).")

    # --- Salida (a): rendimiento en zonas aptas ---
    aptas = base_valida & (aptitud >= umbral)
    en_aptas = np.full(aptitud.shape, NODATA, dtype=np.float32)
    en_aptas[aptas] = rendimiento[aptas]

    # --- Salida (b): ranking combinado prob × rendimiento_normalizado ---
    y = rendimiento[base_valida]
    p1, p99 = np.percentile(y, [1, 99])
    rango = max(p99 - p1, 1e-6)  # evita división por cero si el rendimiento es constante
    yield_norm = np.clip((rendimiento - p1) / rango, 0.0, 1.0)
    ranking = np.full(aptitud.shape, NODATA, dtype=np.float32)
    ranking[base_valida] = (aptitud[base_valida] * yield_norm[base_valida]).astype(np.float32)

    # --- Análisis de sensibilidad de la fórmula de combinación ---
    # El producto es una decisión de diseño justificada solo narrativamente (ver docstring),
    # a diferencia de los pesos AHP del proyecto (derivados formalmente con Ratio de
    # Consistencia de Saaty, ver src/ahp.py). Se contrasta contra una alternativa razonable
    # (media aritmética ponderada 50/50) para reportar cuán sensible es el ranking a esta
    # elección — no para reemplazar la fórmula, sino para que quede auditable en el informe.
    ranking_prod = ranking[base_valida]
    ranking_media = 0.5 * aptitud[base_valida] + 0.5 * yield_norm[base_valida]
    rho_formula, _ = spearmanr(ranking_prod, ranking_media)
    k_top = max(1, int(round(0.01 * base_valida.sum())))  # top 1% de las celdas válidas
    top_prod = set(np.argsort(ranking_prod)[-k_top:].tolist())
    top_media = set(np.argsort(ranking_media)[-k_top:].tolist())
    overlap_top1pct_pct = round(100.0 * len(top_prod & top_media) / k_top, 1)

    # --- Escribir GeoTIFFs (mismo perfil que la aptitud) ---
    meta.update(dtype=rasterio.float32, count=1, nodata=NODATA)
    for path, data in ((out_en_aptas, en_aptas), (out_ranking, ranking)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(data, 1)

    # --- Estadísticas ---
    y_region = rendimiento[base_valida]
    y_aptas = rendimiento[aptas]
    stats = {
        "umbral_probabilidad": float(umbral),
        "celdas_validas": int(base_valida.sum()),
        "celdas_aptas": int(aptas.sum()),
        "pct_aptas": round(100.0 * aptas.sum() / base_valida.sum(), 2),
        "rendimiento_region": _resumen(y_region),
        "rendimiento_aptas": _resumen(y_aptas) if aptas.any() else None,
        "ganancia_aptas_pct": (round(float(100.0 * (y_aptas.mean() / y_region.mean() - 1)), 2)
                               if aptas.any() else None),
        "sensibilidad_formula_combinacion": {
            "formula_usada": "producto: probabilidad_RF * yield_norm",
            "alternativa": "media aritmética ponderada 50/50",
            "spearman_ranking_vs_alternativa": round(float(rho_formula), 4),
            "overlap_top_1pct_pct": overlap_top1pct_pct,
            "nota": (
                "Si el spearman y el overlap son altos, el ranking (y los top-N sitios que se "
                "reporten) son robustos a esta elección de fórmula; si son bajos, la elección "
                "de 'producto' pesa más de lo que parece y debería justificarse con más "
                "alternativas antes de presentarse como definitiva."
            ),
        },
        "salidas": {"rendimiento_en_aptas": out_en_aptas, "aptitud_x_rendimiento": out_ranking},
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def _resumen(a):
    """Resumen estadístico de un array 1D (kWh/kWp/año), redondeado."""
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 1),
        "media": round(float(a.mean()), 1),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p90": round(float(np.percentile(a, 90)), 1),
        "max": round(float(a.max()), 1),
    }
