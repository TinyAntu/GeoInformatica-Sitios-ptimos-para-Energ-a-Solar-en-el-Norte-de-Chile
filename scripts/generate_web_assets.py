"""Genera assets web livianos para el visor (T7): PNG + bounds a partir de los rasters.

Cada mapa de resultados (aptitud, rendimiento, cruce) se reproyecta a EPSG:4326, se
remuestrea a tamaño web, se colorea (nodata transparente) y se guarda como PNG de pocos
KB en `app/assets/`, junto a una barra de color y un `manifest.json` con los límites
lat/lon para superponerlo en Leaflet/folium.

La gracia: estos PNG son chicos, así que se pueden commitear y sirven **igual para el
visor local y para el deploy en la nube** (Streamlit Community Cloud), sin necesidad de
subir los .tif pesados (60–267 MB) ni de un computador encendido.

Uso:
    python scripts/generate_web_assets.py --config config.yaml
"""

import os
import sys
import json
import argparse

import numpy as np
import yaml
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MAX_PX = 1600  # lado máximo del PNG de salida (compromiso nitidez/tamaño)


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _reproyectar_para_web(src_path, resampling=Resampling.bilinear):
    """Reproyecta un raster a Web Mercator (EPSG:3857) remuestreado a MAX_PX.

    Se usa 3857 —no 4326— porque folium/Leaflet coloca el ImageOverlay estirándolo
    linealmente en el espacio de pantalla (que ES Web Mercator) sin reproyectar la imagen.
    Una imagen 4326 (lat/lon plano) estirada así queda distorsionada en el eje norte-sur en
    latitudes altas (norte de Chile ~-25°), lo que se percibe como un desfase. Generando la
    imagen ya en 3857, las esquinas caen exactas sobre el basemap.

    Devuelve (arr, (sur, oeste, norte, este)) con los bounds en lat/lon que espera folium.
    """
    with rasterio.open(src_path) as src:
        # Extensión en metros Web Mercator.
        w_m, s_m, e_m, n_m = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        ancho, alto = e_m - w_m, n_m - s_m
        escala = MAX_PX / max(ancho, alto)
        width = max(1, int(round(ancho * escala)))
        height = max(1, int(round(alto * escala)))
        dst_transform = from_bounds(w_m, s_m, e_m, n_m, width, height)
        destino = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=destino,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=dst_transform, dst_crs="EPSG:3857", dst_nodata=np.nan,
            resampling=resampling,
        )
    # Bounds lat/lon (esquinas de la caja 3857) para folium: [[S,W],[N,E]].
    oeste, sur, este, norte = transform_bounds("EPSG:3857", "EPSG:4326", w_m, s_m, e_m, n_m)
    return destino, (sur, oeste, norte, este)


def _colorear(arr, cmap_name, vmin, vmax, png_path, oculta_menor_a=None):
    """Colorea un array con un colormap (nodata/NaN transparente) y guarda PNG RGBA.

    `oculta_menor_a`: si se indica, las celdas válidas con valor < ese umbral quedan
    transparentes (se usa para el consenso: ocultar el 'no apto' = 0 y resaltar 1–3).
    """
    valido = np.isfinite(arr)
    norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = (colormaps[cmap_name](norm) * 255).astype(np.uint8)
    rgba[~valido, 3] = 0  # transparencia en nodata
    if oculta_menor_a is not None:
        rgba[valido & (arr < oculta_menor_a), 3] = 0
    Image.fromarray(rgba, mode="RGBA").save(png_path)


def _barra_color(cmap_name, vmin, vmax, unidad, out_path):
    """Genera una barra de color independiente para la leyenda del visor."""
    fig, ax = plt.subplots(figsize=(4, 0.5))
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap_name, extent=[vmin, vmax, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel(unidad, fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Genera assets web (PNG+bounds) para el visor')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    prefijo = config.get('solarpv', {}).get('out_prefix', 'data/results/rendimiento')

    # Definición de capas. modo: 'fijo01' (0–1), 'percentil' (contraste p2–p98),
    # 'categorico' (consenso 0–3, oculta el 0 = no apto y usa resampleo nearest).
    capas = [
        # Los 3 perfiles de aptitud (Brecha 6).
        {"id": "balanceado", "ruta": "data/results/mapa_probabilidad_aptitud.tif",
         "cmap": "viridis", "unidad": "Aptitud ML — probabilidad (0–1)", "modo": "fijo01"},
        {"id": "conservador", "ruta": "data/results/aptitud_conservador.tif",
         "cmap": "viridis", "unidad": "Aptitud conservador (0–1)", "modo": "fijo01"},
        {"id": "agresivo", "ruta": "data/results/aptitud_agresivo.tif",
         "cmap": "viridis", "unidad": "Aptitud agresivo (0–1)", "modo": "fijo01"},
        # Producción física y cruce.
        {"id": "rendimiento", "ruta": prefijo + "_fijo_specific_yield.tif",
         "cmap": "inferno", "unidad": "Rendimiento (kWh/kWp/año)", "modo": "percentil"},
        {"id": "cruce", "ruta": "data/results/aptitud_x_rendimiento.tif",
         "cmap": "magma", "unidad": "Ranking aptitud × rendimiento (0–1)", "modo": "fijo01"},
        # Consenso/divergencia entre perfiles (Brecha 6, punto 4).
        {"id": "consenso", "ruta": "data/results/consenso_perfiles.tif",
         "cmap": "RdYlGn", "unidad": "Perfiles aptos: 1 → 3 (3 = consenso)", "modo": "categorico"},
    ]

    assets_dir = _ruta_abs("app/assets")
    os.makedirs(assets_dir, exist_ok=True)
    manifest = {}

    for capa in capas:
        cid, ruta, cmap, unidad, modo = capa["id"], capa["ruta"], capa["cmap"], capa["unidad"], capa["modo"]
        src_path = _ruta_abs(ruta)
        if not os.path.exists(src_path):
            print(f"  [OMITIDA] {cid}: no existe {ruta}")
            continue
        print(f"  Procesando {cid} ({os.path.basename(ruta)})...")
        # El consenso es categórico: nearest para no interpolar entre categorías.
        resampling = Resampling.nearest if modo == "categorico" else Resampling.bilinear
        arr, (sur, oeste, norte, este) = _reproyectar_para_web(src_path, resampling=resampling)

        valido = arr[np.isfinite(arr)]
        oculta_menor_a = None
        if modo == "fijo01":
            vmin, vmax = 0.0, 1.0
        elif modo == "categorico":
            vmin, vmax, oculta_menor_a = 1.0, 3.0, 1.0  # oculta el 0 (no apto)
        else:  # percentil: mejor contraste para el rendimiento
            vmin, vmax = float(np.percentile(valido, 2)), float(np.percentile(valido, 98))

        png = os.path.join(assets_dir, f"{cid}.png")
        cbar = os.path.join(assets_dir, f"{cid}_colorbar.png")
        _colorear(arr, cmap, vmin, vmax, png, oculta_menor_a=oculta_menor_a)
        _barra_color(cmap, vmin, vmax, unidad, cbar)

        manifest[cid] = {
            "png": f"assets/{cid}.png",
            "colorbar": f"assets/{cid}_colorbar.png",
            "bounds": [[sur, oeste], [norte, este]],  # [[S,W],[N,E]] para folium
            "vmin": round(vmin, 2), "vmax": round(vmax, 2),
            "unidad": unidad, "cmap": cmap,
            "tamano_px": [arr.shape[1], arr.shape[0]],
        }
        print(f"    -> {png} ({os.path.getsize(png)//1024} KB), rango [{vmin:.2f}, {vmax:.2f}]")

    with open(os.path.join(assets_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest: {os.path.join(assets_dir, 'manifest.json')} ({len(manifest)} capas)")

    # Empaqueta las estadísticas (T2/T3/T6) en app/assets para que el visor las tenga
    # también en la nube (los JSON originales viven bajo data/, que está en .gitignore).
    stats = {}
    fuentes = {
        "cruce": "data/results/cruce_aptitud_rendimiento.json",
        "comparacion_montaje": "data/results/comparacion_montaje.json",
        "shap": "data/results/shap_importancias.json",
        "consenso": "data/results/consenso_perfiles.json",
    }
    for clave, ruta in fuentes.items():
        p = _ruta_abs(ruta)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                stats[clave] = json.load(f)
    with open(os.path.join(assets_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats:    {os.path.join(assets_dir, 'stats.json')} ({len(stats)} secciones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
