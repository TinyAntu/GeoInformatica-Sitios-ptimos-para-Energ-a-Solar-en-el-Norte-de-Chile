"""Explicabilidad ESPACIAL con SHAP (Brecha 8, parte que faltaba tras T6).

T6 dio la explicabilidad global (qué variables importan en promedio). Esto la lleva al
mapa: calcula los valores SHAP en cada píxel de la grilla de inferencia y produce:

  - Un raster de contribución SHAP por variable (p. ej. "cuánto suma el GHI a la aptitud
    aquí", "cuánto penaliza la lejanía a transmisión aquí").
  - Un mapa de la variable dominante por píxel (qué factor manda en cada lugar).
  - Un waterfall SHAP para los Top-N sitios de mayor aptitud (por qué son aptos).

Reutiliza los helpers de scripts/generate_suitability_map.py para construir la matriz de
features en el MISMO orden canónico (FEATURES), con el mismo assert que protege el
invariante. La grilla suele ser más gruesa que el mapa de aptitud (config
shap_espacial.resolucion_m) porque calcular SHAP en decenas de millones de píxeles es caro.
"""

import os
import json

import numpy as np
import rasterio
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import FEATURES

NODATA = -9999.0


def _construir_features(config, resolucion_m):
    """Construye (crs, transform, width, height, valid_mask, X_pred) en el orden FEATURES.

    Reutiliza las funciones de generate_suitability_map (import perezoso para no exigir el
    path al importar el módulo). Replica el mismo apilado y el mismo assert de orden.
    """
    import geopandas as gpd
    from rasterio.warp import Resampling
    from scripts.generate_suitability_map import (
        construir_grilla_referencia, read_and_reproject_to_grid,
        calcular_distancia_a_capa, get_rasterized_mask,
    )

    paths_raw = config["paths"]["raw"]
    paths_processed = config["paths"]["processed"]
    dem_path = paths_processed["dem_32719"]
    slope_path = paths_processed["slope"]
    aspect_path = paths_processed["aspect"]
    ghi_path = paths_processed["ghi_32719"]

    crs, transform, width, height = construir_grilla_referencia(dem_path, resolucion_m)
    grid_shape = (height, width)
    print(f"  Grilla SHAP: {width}x{height} = {width*height:,} píxeles a ~{transform[0]:.0f} m")

    ghi = read_and_reproject_to_grid(ghi_path, crs, grid_shape, transform, nodata_val=np.nan)
    elev = read_and_reproject_to_grid(dem_path, crs, grid_shape, transform, nodata_val=np.nan)
    slope = read_and_reproject_to_grid(slope_path, crs, grid_shape, transform, nodata_val=np.nan)
    aspect = read_and_reproject_to_grid(aspect_path, crs, grid_shape, transform, nodata_val=np.nan,
                                        resampling=Resampling.nearest)
    with np.errstate(invalid="ignore"):
        northness = np.cos(np.radians(aspect))

    # Regiones antes que las distancias: calcular_distancia_a_capa necesita `regiones` para
    # recortar la infraestructura a Antofagasta+Atacama (mismo criterio que el entrenamiento
    # y que scripts/generate_suitability_map.py).
    regiones = gpd.read_file(paths_raw["vectores"]["regiones"])
    regiones = regiones[regiones["REGION"].isin(["Antofagasta", "Atacama"])].to_crs(crs)
    region_mask = get_rasterized_mask(regiones, grid_shape, transform, fill=0, default_value=1)

    px = transform[0]
    d_trans = calcular_distancia_a_capa(paths_raw["vectores"]["lineas"], crs, grid_shape, transform, px, regiones, nombre="transmisión")
    d_almac = calcular_distancia_a_capa(paths_raw["vectores"]["almacenamiento"], crs, grid_shape, transform, px, regiones, nombre="almacenamiento")
    d_subes = calcular_distancia_a_capa(paths_raw["vectores"]["subestaciones"], crs, grid_shape, transform, px, regiones, nombre="subestaciones")

    valid = ((region_mask == 1) & ~np.isnan(ghi) & ~np.isnan(elev) & ~np.isnan(slope)
             & ~np.isnan(northness) & ~np.isnan(d_trans) & ~np.isnan(d_almac) & ~np.isnan(d_subes))
    if not valid.any():
        return crs, transform, width, height, valid, np.empty((0, len(FEATURES)), np.float32)

    assert FEATURES == ["slope", "ghi", "elev", "northness",
                        "dist_transmision", "dist_almacen", "dist_subestaciones"], (
        "El orden de FEATURES cambió: actualiza el column_stack acorde.")
    X = np.column_stack((slope[valid], ghi[valid], elev[valid], northness[valid],
                         d_trans[valid], d_almac[valid], d_subes[valid]))
    return crs, transform, width, height, valid, X


def _escribir_raster(valores, valid, base_meta, path):
    """Reconstruye un raster 2D con `valores` en las celdas válidas (resto NODATA)."""
    grid = np.full((base_meta["height"], base_meta["width"]), NODATA, dtype=np.float32)
    grid[valid] = valores.astype(np.float32)
    meta = dict(base_meta, driver="GTiff", dtype=rasterio.float32, count=1, nodata=NODATA)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(grid, 1)


def generar_mapas_shap(config, model_path, resolucion_m, out_dir, figures_dir, top_n=5):
    """Genera los rasters SHAP por variable, el mapa dominante y los waterfall Top-N."""
    model = joblib.load(model_path)
    crs, transform, width, height, valid, X = _construir_features(config, resolucion_m)
    if len(X) == 0:
        raise ValueError("No hay píxeles válidos para calcular SHAP espacial.")

    base_meta = {"crs": crs, "transform": transform, "width": width, "height": height}

    print(f"  Calculando SHAP en {len(X):,} píxeles (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv1 = sv[:, :, 1] if sv.ndim == 3 else sv  # contribuciones hacia "apto"
    ev = explainer.expected_value
    ev1 = float(ev[1]) if np.ndim(ev) > 0 else float(ev)

    os.makedirs(out_dir, exist_ok=True)
    salidas = {}
    for i, feat in enumerate(FEATURES):
        path = os.path.join(out_dir, f"shap_espacial_{feat}.tif")
        _escribir_raster(sv1[:, i], valid, base_meta, path)
        salidas[feat] = path

    # Mapa de variable dominante por píxel (índice de la feature con mayor |SHAP|).
    dominante = np.argmax(np.abs(sv1), axis=1).astype(np.float32)
    dom_path = os.path.join(out_dir, "shap_espacial_dominante.tif")
    _escribir_raster(dominante, valid, base_meta, dom_path)
    salidas["dominante"] = dom_path

    # Waterfall de los Top-N sitios por probabilidad de aptitud.
    prob = model.predict_proba(X)[:, 1]
    top_idx = np.argsort(prob)[-top_n:][::-1]
    os.makedirs(figures_dir, exist_ok=True)
    waterfalls = []
    for rank, idx in enumerate(top_idx, 1):
        expl = shap.Explanation(values=sv1[idx], base_values=ev1,
                                data=X[idx], feature_names=list(FEATURES))
        shap.plots.waterfall(expl, show=False)
        fig_path = os.path.join(figures_dir, f"shap_waterfall_top{rank}.png")
        plt.title(f"Top {rank} — aptitud {prob[idx]:.3f}")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        waterfalls.append(fig_path)

    # Contribución media (con signo) por variable sobre la grilla → resumen del JSON.
    contrib_media = {f: round(float(sv1[:, i].mean()), 5) for i, f in enumerate(FEATURES)}
    dom_counts = {FEATURES[i]: int((dominante == i).sum()) for i in range(len(FEATURES))}
    resultado = {
        "resolucion_m": resolucion_m,
        "n_pixeles": int(len(X)),
        "contribucion_media_shap": contrib_media,
        "pixeles_por_variable_dominante": dom_counts,
        "leyenda_dominante": {i: f for i, f in enumerate(FEATURES)},
        "rasters": salidas,
        "waterfalls": waterfalls,
    }
    out_json = os.path.join(out_dir, "shap_espacial.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    return resultado
