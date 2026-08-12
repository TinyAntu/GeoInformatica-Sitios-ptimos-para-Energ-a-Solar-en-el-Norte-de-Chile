"""Comparación de montaje fijo vs. seguidor de un eje (perfiles de inversión).

El motor `solarpv-rs` estima el rendimiento (kWh/kWp/año) para distintos tipos de montaje.
Este módulo contrasta el montaje **fijo** (`--mount tilt`) contra el **seguidor de un eje**
(`--mount tracker`) dentro de las zonas aptas del modelo RF, cuantificando la ganancia del
seguidor. Ese número alimenta la lectura de los perfiles de inversión del proyecto
(conservador vs. agresivo) sin tocar las matrices AHP.

Referencia del autor del motor: un seguidor de un eje rinde ~+36 % sobre módulos fijos
horizontales (validado contra pvlib). Aquí medimos la ganancia real sobre nuestro DEM.
"""

import os
import json

import numpy as np
import rasterio

# Reutilizamos utilidades del cruce (T2): alineado a la grilla de aptitud y resumen.
from src.cruce_aptitud_rendimiento import _alinear_rendimiento, _resumen, NODATA


def comparar(fijo_path, seguidor_path, aptitud_path, umbral, out_json):
    """Compara rendimiento fijo vs. seguidor en las zonas aptas y guarda un JSON.

    Devuelve el dict de estadísticas. Ambos rasters de rendimiento se alinean a la grilla
    del mapa de aptitud (EPSG:32719) antes de comparar, celda a celda.
    """
    with rasterio.open(aptitud_path) as apt:
        aptitud = apt.read(1).astype(np.float32)
        fijo = _alinear_rendimiento(fijo_path, apt)
        seguidor = _alinear_rendimiento(seguidor_path, apt)

    base_valida = (
        (aptitud != NODATA) & np.isfinite(aptitud)
        & np.isfinite(fijo) & np.isfinite(seguidor)
    )
    aptas = base_valida & (aptitud >= umbral)
    if not aptas.any():
        raise ValueError("No hay celdas aptas comunes a fijo, seguidor y aptitud.")

    fijo_aptas = fijo[aptas]
    seg_aptas = seguidor[aptas]

    # Ganancia agregada (medias) y ganancia por celda (más honesta ante distribuciones sesgadas).
    ganancia_media_pct = float(100.0 * (seg_aptas.mean() / fijo_aptas.mean() - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_celda = np.where(fijo_aptas > 0, seg_aptas / fijo_aptas - 1.0, np.nan)
    gain_celda = gain_celda[np.isfinite(gain_celda)]
    ganancia_mediana_celda_pct = float(100.0 * np.median(gain_celda))

    stats = {
        "umbral_probabilidad": float(umbral),
        "celdas_aptas": int(aptas.sum()),
        "rendimiento_fijo_aptas": _resumen(fijo_aptas),
        "rendimiento_seguidor_aptas": _resumen(seg_aptas),
        "ganancia_seguidor_media_pct": round(ganancia_media_pct, 2),
        "ganancia_seguidor_mediana_celda_pct": round(ganancia_mediana_celda_pct, 2),
        "referencia_autor_pct": 36.0,
        "interpretacion": _interpretar(ganancia_media_pct),
        "entradas": {"fijo": fijo_path, "seguidor": seguidor_path},
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def _interpretar(ganancia_pct):
    """Lectura para los perfiles de inversión a partir de la ganancia del seguidor."""
    base = (
        f"El seguidor de un eje rinde {ganancia_pct:+.1f}% sobre el montaje fijo en las "
        "zonas aptas. El perfil 'agresivo' (prioriza radiación/rendimiento) se beneficia "
        "más de esta ganancia; el 'conservador' (prioriza cercanía a infraestructura) la "
        "pondera contra el mayor CAPEX y mantenimiento del seguidor."
    )
    if ganancia_pct < 25:
        return base + " La ganancia medida queda por debajo del ~36% de referencia del " \
                      "autor: conviene discutir por qué (terreno, latitud, sombreado)."
    return base
