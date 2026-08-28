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
import pandas as pd
import rasterio

# Reutilizamos utilidades del cruce (T2): alineado a la grilla de aptitud y resumen.
from src.cruce_aptitud_rendimiento import _alinear_rendimiento, _resumen, NODATA


def comparar(fijo_path, seguidor_path, aptitud_path, umbral, out_json, tamano_bloque_km=15, fijo_tilt0_path=None):
    """Compara rendimiento fijo vs. seguidor (y opcionalmente fijo tilt=0) en zonas aptas y guarda un JSON.

    Devuelve el dict de estadísticas. Todos los rasters de rendimiento se alinean a la grilla
    del mapa de aptitud (EPSG:32719) antes de comparar, celda a celda.

    `tamano_bloque_km` (mismo valor que `validacion.tamano_bloque_km`, usado por el SBCV del
    RF) agrega la ganancia por bloque espacial antes de resumir entre bloques: los píxeles de
    "zonas aptas" no son observaciones independientes (pixeles vecinos comparten casi la misma
    pendiente/orientación/sombreado), así que tratar cada píxel como una muestra independiente
    sobreestima el tamaño de muestra efectivo (el mismo problema de autocorrelación espacial
    que motiva el Spatial Block CV, ver src/spatial_validation.py — Roberts et al. 2017;
    Ploton et al. 2020). El resumen por celda se conserva por continuidad, pero el por bloque
    es el que debería citarse si se reporta un rango/dispersión de la ganancia.
    """
    with rasterio.open(aptitud_path) as apt:
        aptitud = apt.read(1).astype(np.float32)
        transform = apt.transform
        fijo = _alinear_rendimiento(fijo_path, apt)
        seguidor = _alinear_rendimiento(seguidor_path, apt)
        fijo_tilt0 = _alinear_rendimiento(fijo_tilt0_path, apt) if fijo_tilt0_path else None

    base_valida = (
        (aptitud != NODATA) & np.isfinite(aptitud)
        & np.isfinite(fijo) & np.isfinite(seguidor)
    )
    if fijo_tilt0 is not None:
        base_valida &= np.isfinite(fijo_tilt0)

    aptas = base_valida & (aptitud >= umbral)
    if not aptas.any():
        raise ValueError("No hay celdas aptas comunes a las configuraciones de montaje y aptitud.")

    fijo_aptas = fijo[aptas]
    seg_aptas = seguidor[aptas]
    fijo_tilt0_aptas = fijo_tilt0[aptas] if fijo_tilt0 is not None else None

    # Ganancia agregada (medias) y ganancia por celda (más honesta ante distribuciones sesgadas).
    # OJO: "por celda" trata cada píxel como independiente (ver docstring) — se mantiene como
    # referencia histórica, no como la cifra de dispersión más defendible.
    ganancia_media_pct = float(100.0 * (seg_aptas.mean() / fijo_aptas.mean() - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_por_celda = np.where(fijo_aptas > 0, seg_aptas / fijo_aptas - 1.0, np.nan)
    gain_finito = gain_por_celda[np.isfinite(gain_por_celda)]
    ganancia_mediana_celda_pct = float(100.0 * np.median(gain_finito))

    # --- Agregación por bloque espacial: promedia dentro de cada bloque primero, y solo
    # después resume entre bloques. `n_bloques` es el tamaño de muestra efectivo honesto. ---
    rows, cols = np.where(aptas)  # mismo orden row-major que fijo[aptas]/seguidor[aptas]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    bloque_m = tamano_bloque_km * 1000.0
    bx = np.floor(np.asarray(xs) / bloque_m).astype(np.int64)
    by = np.floor(np.asarray(ys) / bloque_m).astype(np.int64)
    df_bloques = pd.DataFrame({"block_id": bx * 1_000_003 + by, "gain": gain_por_celda})
    df_bloques = df_bloques[np.isfinite(df_bloques["gain"])]
    gain_por_bloque = df_bloques.groupby("block_id")["gain"].mean()
    n_bloques = int(gain_por_bloque.size)
    ganancia_mediana_bloque_pct = (
        round(float(100.0 * gain_por_bloque.median()), 2) if n_bloques > 0 else None
    )

    entradas = {"fijo_tilt23": fijo_path, "seguidor": seguidor_path}
    if fijo_tilt0_path:
        entradas["fijo_tilt0"] = fijo_tilt0_path

    stats = {
        "umbral_probabilidad": float(umbral),
        "celdas_aptas": int(aptas.sum()),
        "tamano_bloque_km": tamano_bloque_km,
        "n_bloques_espaciales": n_bloques,
        "rendimiento_fijo_aptas": _resumen(fijo_aptas),
        "rendimiento_seguidor_aptas": _resumen(seg_aptas),
        "ganancia_seguidor_media_pct": round(ganancia_media_pct, 2),
        "ganancia_seguidor_mediana_celda_pct": round(ganancia_mediana_celda_pct, 2),
        "ganancia_seguidor_mediana_bloque_pct": ganancia_mediana_bloque_pct,
        "referencia_autor_pct": 36.0,
        "interpretacion": _interpretar(ganancia_media_pct, n_bloques, int(aptas.sum())),
        "entradas": entradas,
    }

    if fijo_tilt0_aptas is not None:
        stats["rendimiento_fijo_tilt0_aptas"] = _resumen(fijo_tilt0_aptas)
        gan_tilt23_vs_tilt0 = float(100.0 * (fijo_aptas.mean() / fijo_tilt0_aptas.mean() - 1))
        gan_seg_vs_tilt0 = float(100.0 * (seg_aptas.mean() / fijo_tilt0_aptas.mean() - 1))
        stats["ganancia_tilt23_vs_tilt0_media_pct"] = round(gan_tilt23_vs_tilt0, 2)
        stats["ganancia_seguidor_vs_tilt0_media_pct"] = round(gan_seg_vs_tilt0, 2)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


def _interpretar(ganancia_pct, n_bloques, n_celdas):
    """Lectura para los perfiles de inversión a partir de la ganancia del seguidor."""
    base = (
        f"El seguidor de un eje rinde {ganancia_pct:+.1f}% sobre el montaje fijo en las "
        "zonas aptas. El perfil 'agresivo' (prioriza radiación/rendimiento) se beneficia "
        "más de esta ganancia; el 'conservador' (prioriza cercanía a infraestructura) la "
        "pondera contra el mayor CAPEX y mantenimiento del seguidor."
    )
    if ganancia_pct < 25:
        base += " La ganancia medida queda por debajo del ~36% de referencia del " \
                "autor: conviene discutir por qué (terreno, latitud, sombreado)."
    base += (
        f" Nota metodológica: {n_celdas:,} celdas aptas caen en solo {n_bloques} bloques "
        "espaciales independientes (pixeles vecinos comparten terreno/sombreado); "
        f"{n_bloques} es el tamaño de muestra efectivo honesto para cualquier lectura de "
        "dispersión — no las celdas."
    )
    return base
