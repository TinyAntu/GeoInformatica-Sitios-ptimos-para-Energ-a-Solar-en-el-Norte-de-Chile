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
from matplotlib.colors import ListedColormap
from PIL import Image

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _asegurar_proj_lib
_asegurar_proj_lib()
from src.preprocessing import _obtener_crs
from src.rutas import ruta_resultado, prefijo_rendimiento

MAX_PX = 1600  # lado máximo del PNG de salida (compromiso nitidez/tamaño)

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7']
if 'okabe_ito_gs' not in colormaps:
    colormaps.register(ListedColormap(OKABE_ITO, name='okabe_ito_gs'), name='okabe_ito_gs')


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _reproyectar_para_web(src_path, resampling=Resampling.bilinear, clip_mask_path=None,
                          nodata_fuente=None):
    """Reproyecta un raster a Web Mercator (EPSG:3857) remuestreado a MAX_PX.

    `nodata_fuente` reemplaza al nodata declarado en el ráster. El rendimiento del motor no
    declara nodata y deja en 0 las celdas sin GHI: con `nodata_fuente=0` esas celdas quedan
    transparentes y el bilineal no las mezcla con sus vecinas en el borde.
    """
    crs_3857 = _obtener_crs(3857)
    crs_4326 = _obtener_crs(4326)
    with rasterio.open(src_path) as src:
        src_crs = src.crs if src.crs and getattr(src.crs, 'is_projected', False) else _obtener_crs(32719)
        # Extensión en metros Web Mercator.
        w_m, s_m, e_m, n_m = transform_bounds(src_crs, crs_3857, *src.bounds)
        ancho, alto = e_m - w_m, n_m - s_m
        escala = MAX_PX / max(ancho, alto)
        width = max(1, int(round(ancho * escala)))
        height = max(1, int(round(alto * escala)))
        dst_transform = from_bounds(w_m, s_m, e_m, n_m, width, height)
        destino = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=destino,
            src_transform=src.transform, src_crs=src_crs,
            src_nodata=src.nodata if nodata_fuente is None else nodata_fuente,
            dst_transform=dst_transform, dst_crs=crs_3857, dst_nodata=np.nan,
            resampling=resampling,
        )
        if clip_mask_path is not None:
            with rasterio.open(clip_mask_path) as mascara:
                datos_mascara = mascara.read(1, masked=True)
                mascara_valida = ~np.ma.getmaskarray(datos_mascara)
                mascara_destino = np.zeros((height, width), dtype=np.uint8)
                mask_crs = mascara.crs if mascara.crs and getattr(mascara.crs, 'is_projected', False) else _obtener_crs(32719)
                reproject(
                    source=mascara_valida.astype(np.uint8), destination=mascara_destino,
                    src_transform=mascara.transform, src_crs=mask_crs,
                    src_nodata=0, dst_transform=dst_transform, dst_crs=crs_3857,
                    dst_nodata=0, resampling=Resampling.nearest,
                )
                destino[mascara_destino == 0] = np.nan
    # Bounds lat/lon (esquinas de la caja 3857) para folium: [[S,W],[N,E]].
    oeste, sur, este, norte = transform_bounds(crs_3857, crs_4326, w_m, s_m, e_m, n_m)
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
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _barra_categorica(cmap_name, etiquetas, out_path):
    """Leyenda discreta (swatch de color + etiqueta) para una capa categórica.

    Usa la MISMA normalización que _colorear (color de la categoría i = cmap(i/(n-1))),
    para que los colores de la leyenda coincidan con los del mapa.
    """
    n = len(etiquetas)
    cmap = colormaps[cmap_name]
    fig, ax = plt.subplots(figsize=(3.2, 0.34 * n))
    for i, et in enumerate(etiquetas):
        y = n - 1 - i
        ax.add_patch(plt.Rectangle((0, y + 0.1), 0.5, 0.8, color=cmap(i / max(n - 1, 1))))
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Genera assets web (PNG+bounds) para el visor')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--zona', default=None,
                        help='Clave de config.yaml con otra zona (p. ej. "transferibilidad"): '
                             'lee sus resultados y escribe un manifest y unos PNG aparte.')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Con --zona todo se lee del directorio de esa zona y se escribe en assets/<zona>/ con un
    # manifest propio. Deliberadamente NO se mezcla con el manifest principal: así el visor
    # desplegado sigue funcionando aunque la zona no se haya calculado nunca.
    sufijo = ''
    subdir = ''
    if args.zona:
        if not config.get(args.zona):
            print(f"  [ERROR] config.yaml no tiene el bloque '{args.zona}'.")
            return 1
        sufijo = f'_{args.zona}'
        subdir = args.zona
    # Cada resultado en la subcarpeta de su tema, dentro de la zona si corresponde (src/rutas.py).
    res = lambda tema, nombre: ruta_resultado(config, tema, nombre, args.zona)
    prefijo = prefijo_rendimiento(config, args.zona)

    # Definición de capas. modo: 'fijo01' (0–1), 'percentil' (contraste p2–p98),
    # 'categorico' (consenso 0–3, oculta el 0 = no apto y usa resampleo nearest).
    capas = [
        # Los 3 perfiles de aptitud (Brecha 6).
        {"id": "balanceado", "ruta": res("aptitud", "mapa_probabilidad_aptitud.tif"),
         "cmap": "viridis", "unidad": "Aptitud ML — probabilidad (0–1)", "modo": "fijo01"},
        {"id": "conservador", "ruta": res("aptitud", "aptitud_conservador.tif"),
         "cmap": "viridis", "unidad": "Aptitud conservador (0–1)", "modo": "fijo01"},
        {"id": "agresivo", "ruta": res("aptitud", "aptitud_agresivo.tif"),
         "cmap": "viridis", "unidad": "Aptitud agresivo (0–1)", "modo": "fijo01"},
        # Producción física y cruce.
        {"id": "rendimiento", "ruta": prefijo + "_fijo_specific_yield.tif",
         "cmap": "inferno", "unidad": "Rendimiento (kWh/kWp/año)", "modo": "percentil"},
        {"id": "cruce", "ruta": res("cruce", "aptitud_x_rendimiento.tif"),
         "cmap": "magma", "unidad": "Ranking aptitud × rendimiento (0–1)", "modo": "fijo01"},
        # Consenso/divergencia entre perfiles (Brecha 6, punto 4). cividis: secuencial y
        # ordinal, apta para daltonismo (reemplaza RdYlGn, ilegible en rojo-verde).
        {"id": "consenso", "ruta": res("aptitud", "consenso_perfiles.tif"),
         "cmap": "cividis", "unidad": "Perfiles aptos: 1 → 3 (3 = consenso)", "modo": "categorico"},
        # Variable dominante por píxel según SHAP (Brecha 8). Orden = FEATURES.
        # Okabe-Ito: categórica nominal, apta para daltonismo (reemplaza tab10).
        {"id": "dominante", "ruta": res("explicabilidad", "shap_espacial_dominante.tif"),
         "cmap": "okabe_ito_gs", "unidad": "Variable dominante (SHAP)", "modo": "categorias",
         "categorias": ["Pendiente", "GHI", "Elevación", "Northness",
                        "Dist. transmisión", "Dist. almacenam.", "Dist. subestaciones"]},
    ]

    assets_dir = _ruta_abs(os.path.join("app/assets", subdir))
    os.makedirs(assets_dir, exist_ok=True)
    # Prefijo con el que el visor resuelve las rutas del manifest (siempre relativo a app/).
    rel_assets = f"assets/{subdir}/" if subdir else "assets/"
    manifest = {}

    for capa in capas:
        cid, ruta, cmap, unidad, modo = capa["id"], capa["ruta"], capa["cmap"], capa["unidad"], capa["modo"]
        src_path = _ruta_abs(ruta)
        if not os.path.exists(src_path):
            print(f"  [OMITIDA] {cid}: no existe {ruta}")
            continue
        print(f"  Procesando {cid} ({os.path.basename(ruta)})...")
        # Los mapas categóricos se remuestrean con nearest (no interpolar entre categorías).
        es_categorico = modo in ("categorico", "categorias")
        resampling = Resampling.nearest if es_categorico else Resampling.bilinear
        mascara, nodata_fuente = None, None
        if cid == "rendimiento":
            mascara = res("aptitud", "mapa_probabilidad_aptitud.tif")
            nodata_fuente = 0.0  # 0 = sin dato de GHI observado en el motor
        arr, (sur, oeste, norte, este) = _reproyectar_para_web(
            src_path, resampling=resampling, clip_mask_path=mascara,
            nodata_fuente=nodata_fuente,
        )

        valido = arr[np.isfinite(arr)]
        oculta_menor_a = None
        if modo == "fijo01":
            vmin, vmax = 0.0, 1.0
        elif modo == "categorico":
            vmin, vmax, oculta_menor_a = 1.0, 3.0, 1.0  # oculta el 0 (no apto)
        elif modo == "categorias":
            vmin, vmax = 0.0, float(len(capa["categorias"]) - 1)
        else:  # percentil: mejor contraste para el rendimiento
            vmin, vmax = float(np.percentile(valido, 2)), float(np.percentile(valido, 98))

        png = os.path.join(assets_dir, f"{cid}.png")
        cbar = os.path.join(assets_dir, f"{cid}_colorbar.png")
        _colorear(arr, cmap, vmin, vmax, png, oculta_menor_a=oculta_menor_a)
        if modo == "categorias":
            _barra_categorica(cmap, capa["categorias"], cbar)
        else:
            _barra_color(cmap, vmin, vmax, unidad, cbar)

        manifest[cid] = {
            "png": f"{rel_assets}{cid}.png",
            "colorbar": f"{rel_assets}{cid}_colorbar.png",
            "bounds": [[sur, oeste], [norte, este]],  # [[S,W],[N,E]] para folium
            "vmin": round(vmin, 2), "vmax": round(vmax, 2),
            "unidad": unidad, "cmap": cmap,
            "categorias": capa.get("categorias"),
            "tamano_px": [arr.shape[1], arr.shape[0]],
        }
        print(f"    -> {png} ({os.path.getsize(png)//1024} KB), rango [{vmin:.2f}, {vmax:.2f}]")

    # Los JSON van siempre en la raíz de app/assets (los PNG sí en el subdirectorio de la
    # zona), para que el visor los encuentre sin tener que conocer la estructura interna.
    dir_json = _ruta_abs("app/assets")
    ruta_manifest = os.path.join(dir_json, f"manifest{sufijo}.json")
    with open(ruta_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Manifest: {ruta_manifest} ({len(manifest)} capas)")

    # Empaqueta las estadísticas (T2/T3/T6) en app/assets para que el visor las tenga
    # también en la nube (los JSON originales viven bajo data/, que está en .gitignore).
    stats = {}
    if args.zona:
        fuentes = {
            "transferibilidad": res("transferibilidad", "metricas_transferibilidad.json"),
            "consenso": res("aptitud", "consenso_perfiles.json"),
            "shap_espacial": res("explicabilidad", "shap_espacial.json"),
            "cruce": res("cruce", "cruce_aptitud_rendimiento.json"),
        }
    else:
        fuentes = {
            "cruce": res("cruce", "cruce_aptitud_rendimiento.json"),
            "comparacion_montaje": res("rendimiento", "comparacion_montaje.json"),
            "shap": res("explicabilidad", "shap_importancias.json"),
            "consenso": res("aptitud", "consenso_perfiles.json"),
            "shap_espacial": res("explicabilidad", "shap_espacial.json"),
            # Necesario para dibujar los puntos A y B del gradiente de generalización en la
            # pestaña de transferencia: son las cifras del norte, no de la zona evaluada.
            "metricas_topk": res("modelo", "metricas_topk.json"),
        }
    for clave, ruta in fuentes.items():
        p = _ruta_abs(ruta)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                stats[clave] = json.load(f)
    ruta_stats = os.path.join(dir_json, f"stats{sufijo}.json")
    with open(ruta_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats:    {ruta_stats} ({len(stats)} secciones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
