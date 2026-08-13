"""Consenso vs. divergencia entre los perfiles de aptitud (Brecha 6, punto 4).

Los 3 perfiles —conservador (AHP, prioriza infraestructura), balanceado (RF/ML) y
agresivo (AHP, prioriza radiación)— responden la misma pregunta con distintas ponderaciones
de criterios. Este módulo identifica:

  - **Zonas de consenso**: aptas en los 3 perfiles → robustas ante la preferencia del
    stakeholder (apuestas seguras).
  - **Zonas de divergencia**: aptas en 1 o 2 perfiles → dependen del criterio priorizado
    (dónde la decisión de inversión importa).

Los perfiles están en escalas distintas (el RF es probabilidad 0–1; los AHP son scores WLC
centrados ~0.78), por lo que "apto en un perfil" se define de forma **relativa**: pertenecer
al top (100−percentil)% de ese perfil. Así la comparación es justa entre escalas. Todos los
mapas se alinean a la grilla del mapa RF (la más fina). Salida: un raster con el número de
perfiles que consideran apta cada celda (0–3) y un JSON de estadísticas.
"""

import os
import json

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

NODATA = -9999.0


def _alinear(path, base):
    """Reproyecta un perfil a la grilla del mapa base (bilinear). NaN donde no hay dato."""
    dst = np.full((base.height, base.width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=base.transform, dst_crs=base.crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dst


def analizar_consenso(base_path, conservador_path, agresivo_path, percentil,
                      out_raster, out_json):
    """Genera el raster de consenso (0–3 perfiles) y un JSON. Devuelve el dict de stats.

    - base_path: mapa_probabilidad_aptitud.tif (RF = perfil balanceado y grilla de referencia).
    - percentil: define el umbral relativo por perfil (p. ej. 90 → top 10% de cada perfil).
    """
    with rasterio.open(base_path) as base:
        balanceado = base.read(1).astype(np.float32)
        base_nodata = base.nodata
        meta = base.meta.copy()
        conservador = _alinear(conservador_path, base)
        agresivo = _alinear(agresivo_path, base)

    balanceado = np.where(balanceado == base_nodata, np.nan, balanceado)
    perfiles = {"conservador": conservador, "balanceado": balanceado, "agresivo": agresivo}

    base_valida = (np.isfinite(conservador) & np.isfinite(agresivo) & np.isfinite(balanceado))
    if not base_valida.any():
        raise ValueError("Los 3 perfiles no tienen celdas válidas comunes (revisa CRS/extensión).")

    # "Apto" relativo por perfil: top (100-percentil)% de ese perfil.
    apt, umbrales = {}, {}
    for nombre, arr in perfiles.items():
        thr = float(np.percentile(arr[base_valida], percentil))
        umbrales[nombre] = round(thr, 4)
        apt[nombre] = base_valida & (arr >= thr)

    # Nº de perfiles que consideran apta cada celda (0–3).
    n_apto = np.zeros(balanceado.shape, dtype=np.float32)
    for m in apt.values():
        n_apto += m
    n_apto = np.where(base_valida, n_apto, NODATA).astype(np.float32)

    meta.update(dtype=rasterio.float32, count=1, nodata=NODATA)
    os.makedirs(os.path.dirname(out_raster) or ".", exist_ok=True)
    with rasterio.open(out_raster, "w", **meta) as dst:
        dst.write(n_apto, 1)

    total = int(base_valida.sum())
    consenso = int((n_apto == 3).sum())
    div2 = int((n_apto == 2).sum())
    div1 = int((n_apto == 1).sum())
    no_apto = int((n_apto == 0).sum())

    def _pct(x):
        return round(100.0 * x / total, 2)

    stats = {
        "percentil_apto": percentil,
        "umbrales_por_perfil": umbrales,
        "celdas_validas": total,
        "apto_por_perfil": {n: int(m.sum()) for n, m in apt.items()},
        "consenso_3_perfiles": {"celdas": consenso, "pct": _pct(consenso)},
        "divergencia_2_perfiles": {"celdas": div2, "pct": _pct(div2)},
        "divergencia_1_perfil": {"celdas": div1, "pct": _pct(div1)},
        "no_apto": {"celdas": no_apto, "pct": _pct(no_apto)},
        "interpretacion": (
            f"El {_pct(consenso)}% de la zona válida es apta en los 3 perfiles (consenso: "
            f"robusto ante la preferencia del stakeholder); {_pct(div1 + div2)}% es apta solo "
            f"en 1 o 2 perfiles (divergencia: la decisión depende del criterio priorizado)."
        ),
        "salida_raster": out_raster,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats
