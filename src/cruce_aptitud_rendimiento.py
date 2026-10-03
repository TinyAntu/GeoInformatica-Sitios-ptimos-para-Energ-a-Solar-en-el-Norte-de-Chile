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
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except (ImportError, Exception):
    rasterio = None
    reproject = None
    Resampling = None

try:
    import tifffile
except (ImportError, Exception):
    tifffile = None

import scipy.ndimage
from scipy.stats import spearmanr

NODATA = -9999.0


def _obtener_geotags(path):
    """Extrae ModelPixelScale y ModelTiepoint de un GeoTIFF con tifffile."""
    if tifffile is None:
        return None, None, []
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = []
            for code in (33550, 33922, 34735, 34736, 34737):
                t = page.tags.get(code)
                if t is not None:
                    tags.append((t.code, t.dtype, len(t.value) if isinstance(t.value, (tuple, list)) else 1, t.value, True))
            scale = page.tags.get(33550).value if 33550 in page.tags else (100.0, 100.0, 0.0)
            tiepoint = page.tags.get(33922).value if 33922 in page.tags else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return scale, tiepoint, tags
    except Exception:
        return None, None, []


def _alinear_rendimiento(rendimiento_path, aptitud_path_o_dataset):
    """Reproyecta el rendimiento a la grilla del mapa de aptitud (bilinear)."""
    if rasterio is not None and hasattr(aptitud_path_o_dataset, 'read'):
        base = aptitud_path_o_dataset
        destino = np.full((base.height, base.width), np.nan, dtype=np.float32)
        with rasterio.open(rendimiento_path) as ren:
            src_crs = ren.crs
            if not src_crs or not getattr(src_crs, 'is_projected', False):
                src_crs = base.crs
            reproject(
                source=rasterio.band(ren, 1),
                destination=destino,
                src_transform=ren.transform, src_crs=src_crs, src_nodata=ren.nodata,
                dst_transform=base.transform, dst_crs=base.crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
        return destino

    # Fallback con tifffile y scipy.ndimage si rasterio no está disponible
    apt_path = aptitud_path_o_dataset if isinstance(aptitud_path_o_dataset, str) else getattr(aptitud_path_o_dataset, 'name', '')
    with tifffile.TiffFile(apt_path) as ta:
        apt_shape = ta.pages[0].shape
    with tifffile.TiffFile(rendimiento_path) as tr:
        ren = tr.asarray().astype(np.float32)

    scale_apt, tie_apt, _ = _obtener_geotags(apt_path)
    scale_ren, tie_ren, _ = _obtener_geotags(rendimiento_path)

    res_apt_x, res_apt_y = scale_apt[0], scale_apt[1]
    res_ren_x, res_ren_y = scale_ren[0], scale_ren[1]
    x0_apt, y0_apt = tie_apt[3], tie_apt[4]
    x0_ren, y0_ren = tie_ren[3], tie_ren[4]

    H, W = apt_shape
    r_coords = (np.arange(H, dtype=np.float32) * res_apt_y + (y0_ren - y0_apt)) / res_ren_y
    c_coords = (np.arange(W, dtype=np.float32) * res_apt_x + (x0_apt - x0_ren)) / res_ren_x

    grid_r, grid_c = np.meshgrid(r_coords, c_coords, indexing='ij')
    coords = np.array([grid_r, grid_c])
    destino = scipy.ndimage.map_coordinates(ren, coords, order=1, mode='constant', cval=np.nan)
    return destino.astype(np.float32)


def cruzar(aptitud_path, rendimiento_path, umbral,
           out_en_aptas, out_ranking, out_json):
    """Genera las dos salidas del cruce y un JSON de estadísticas. Devuelve el dict."""
    if rasterio is not None:
        try:
            with rasterio.open(aptitud_path) as apt:
                aptitud = apt.read(1).astype(np.float32)
                meta = apt.meta.copy()
                rendimiento = _alinear_rendimiento(rendimiento_path, apt)
        except Exception:
            aptitud = tifffile.imread(aptitud_path).astype(np.float32)
            meta = None
            rendimiento = _alinear_rendimiento(rendimiento_path, aptitud_path)
    else:
        aptitud = tifffile.imread(aptitud_path).astype(np.float32)
        meta = None
        rendimiento = _alinear_rendimiento(rendimiento_path, aptitud_path)

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
    ranking_prod = ranking[base_valida]
    ranking_media = 0.5 * aptitud[base_valida] + 0.5 * yield_norm[base_valida]
    rho_formula, _ = spearmanr(ranking_prod, ranking_media)
    k_top = max(1, int(round(0.01 * base_valida.sum())))  # top 1% de las celdas válidas
    top_prod = set(np.argsort(ranking_prod)[-k_top:].tolist())
    top_media = set(np.argsort(ranking_media)[-k_top:].tolist())
    overlap_top1pct_pct = round(100.0 * len(top_prod & top_media) / k_top, 1)

    # --- Escribir GeoTIFFs (mismo perfil que la aptitud) ---
    _, _, tags = _obtener_geotags(aptitud_path)
    for path, data in ((out_en_aptas, en_aptas), (out_ranking, ranking)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if meta is not None and rasterio is not None:
            try:
                meta.update(dtype=rasterio.float32, count=1, nodata=NODATA)
                with rasterio.open(path, "w", **meta) as dst:
                    dst.write(data, 1)
                continue
            except Exception:
                pass
        if tifffile is not None:
            tifffile.imwrite(path, data.astype(np.float32), extratags=tags)

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
