"""Renderiza los GeoTIFF de aptitud como figuras listas para el informe PEP1.

Cada figura incluye los 7 elementos cartográficos obligatorios (clase 9 / rúbrica):
título, leyenda (colorbar), escala gráfica, norte, fuente/autor, CRS y fecha.
Paleta secuencial perceptualmente uniforme (viridis) — nunca rainbow/jet.

Uso:
    python scripts/generar_figuras_informe.py

Entradas (generadas por generate_suitability_map.py y profiles.py):
    data/results/mapa_probabilidad_aptitud.tif
    data/results/aptitud_conservador.tif
    data/results/aptitud_agresivo.tif
Salidas:
    figures/mapa_aptitud_rf.png / mapa_aptitud_conservador.png / mapa_aptitud_agresivo.png
"""
import os
import sys
from datetime import date

import numpy as np
import yaml
import rasterio
from rasterio.enums import Resampling
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo: solo guardamos figuras
import matplotlib.pyplot as plt
from matplotlib import patches

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

MAX_PIXELES_LADO = 2600  # tope de resolución de lectura: suficiente para 300 dpi en media página
COLOR_TINTA = '#333333'  # texto y ejes en tinta neutra (la identidad la lleva el raster)


def leer_raster_reducido(path):
    """Lee la banda 1 reducida a MAX_PIXELES_LADO (nearest: conserva valores y nodata)."""
    with rasterio.open(path) as src:
        factor = max(src.width, src.height) / MAX_PIXELES_LADO
        out_h = int(src.height / factor) if factor > 1 else src.height
        out_w = int(src.width / factor) if factor > 1 else src.width
        datos = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        nodata = src.nodata if src.nodata is not None else -9999.0
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    enmascarado = np.ma.masked_invalid(np.ma.masked_equal(datos, nodata))
    return enmascarado, extent


def dibujar_barra_escala(ax, largo_km=100):
    """Escala gráfica: barra de `largo_km` anclada abajo a la izquierda (coords de datos)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    largo_m = largo_km * 1000
    bx = x0 + 0.05 * (x1 - x0)
    by = y0 + 0.04 * (y1 - y0)
    alto = 0.008 * (y1 - y0)
    ax.add_patch(patches.Rectangle((bx, by), largo_m, alto,
                                   facecolor=COLOR_TINTA, edgecolor='none', zorder=5))
    ax.add_patch(patches.Rectangle((bx, by), largo_m / 2, alto,
                                   facecolor='white', edgecolor=COLOR_TINTA,
                                   linewidth=0.5, zorder=6))
    ax.text(bx, by + alto * 1.8, f'0        {largo_km // 2}        {largo_km} km',
            fontsize=8, color=COLOR_TINTA, zorder=6)


def dibujar_norte(ax):
    """Flecha de norte en la esquina superior derecha (fracción de ejes)."""
    ax.annotate('N', xy=(0.96, 0.985), xytext=(0.96, 0.925),
                xycoords='axes fraction', textcoords='axes fraction',
                ha='center', va='top', fontsize=13, fontweight='bold', color=COLOR_TINTA,
                arrowprops=dict(facecolor=COLOR_TINTA, edgecolor=COLOR_TINTA,
                                width=3, headwidth=10, headlength=8))


def renderizar_mapa(tif_path, titulo, etiqueta_leyenda, out_png, regiones):
    print(f"Renderizando {os.path.basename(tif_path)} ...")
    datos, extent = leer_raster_reducido(tif_path)

    fig, ax = plt.subplots(figsize=(8.5, 10))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(alpha=0.0)  # nodata transparente

    im = ax.imshow(datos, extent=extent, origin='upper', cmap=cmap,
                   vmin=0.0, vmax=1.0, interpolation='nearest')
    regiones.boundary.plot(ax=ax, color=COLOR_TINTA, linewidth=0.7, zorder=4)

    # Encuadre al área de estudio: algunos rasters (p. ej. GHI) cubren todo Chile y sin
    # este recorte la región quedaría como una esquina diminuta en un lienzo vacío.
    minx, miny, maxx, maxy = regiones.total_bounds
    margen_x, margen_y = 0.03 * (maxx - minx), 0.03 * (maxy - miny)
    ax.set_xlim(minx - margen_x, maxx + margen_x)
    ax.set_ylim(miny - margen_y, maxy + margen_y)

    # Leyenda (colorbar) con etiqueta y unidades
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(etiqueta_leyenda, fontsize=10, color=COLOR_TINTA)
    cbar.ax.tick_params(labelsize=8, colors=COLOR_TINTA)

    # Título
    ax.set_title(titulo, fontsize=12, color=COLOR_TINTA, pad=12)

    # Ejes recesivos en km UTM
    ax.set_xlabel('Este UTM (km)', fontsize=9, color=COLOR_TINTA)
    ax.set_ylabel('Norte UTM (km)', fontsize=9, color=COLOR_TINTA)
    ax.xaxis.set_major_formatter(lambda v, _: f'{v / 1000:,.0f}')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v / 1000:,.0f}')
    ax.tick_params(labelsize=8, colors=COLOR_TINTA)
    for spine in ax.spines.values():
        spine.set_color('#bbbbbb')
    ax.set_aspect('equal')

    # Escala y norte
    dibujar_barra_escala(ax, largo_km=100)
    dibujar_norte(ax)

    # Fuente, CRS y fecha (pie de mapa)
    pie = ("Fuente: IDE Energía, Explorador Solar (Min. Energía), NASA SRTM, BCN. "
           "Elaboración propia, Grupo Solar (USACH).\n"
           f"CRS: EPSG:32719 (WGS 84 / UTM zona 19S)  |  Fecha de elaboración: "
           f"{date.today().strftime('%d-%m-%Y')}  |  Zonas de exclusión (SNASPE, agua, "
           "urbano) con aptitud 0.")
    fig.text(0.5, 0.015, pie, ha='center', va='bottom', fontsize=7.5, color=COLOR_TINTA)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Guardado: {out_png}")


def main():
    print("=== GENERANDO FIGURAS CARTOGRÁFICAS PARA EL INFORME ===")
    with open(os.path.join(directorio_raiz, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    regiones_path = os.path.join(directorio_raiz, config['paths']['raw']['vectores']['regiones'])
    regiones = gpd.read_file(regiones_path)
    regiones = regiones[regiones['REGION'].isin(['Antofagasta', 'Atacama'])].to_crs(epsg=32719)

    figures_dir = os.path.join(directorio_raiz, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = os.path.join(directorio_raiz, 'data', 'results')

    mapas = [
        ('mapa_probabilidad_aptitud.tif',
         'Aptitud fotovoltaica — probabilidad Random Forest (perfil balanceado)',
         'Probabilidad de aptitud (0–1)',
         'mapa_aptitud_rf.png'),
        ('aptitud_conservador.tif',
         'Aptitud fotovoltaica — WLC con pesos AHP, perfil conservador (CR = 0.001)',
         'Índice de aptitud WLC (0–1)',
         'mapa_aptitud_conservador.png'),
        ('aptitud_agresivo.tif',
         'Aptitud fotovoltaica — WLC con pesos AHP, perfil agresivo (CR = 0.021)',
         'Índice de aptitud WLC (0–1)',
         'mapa_aptitud_agresivo.png'),
    ]

    for tif, titulo, etiqueta, png in mapas:
        tif_path = os.path.join(results_dir, tif)
        if not os.path.exists(tif_path):
            print(f"  [AVISO] No existe {tif_path}; se omite. "
                  "(¿Corriste generate_suitability_map.py / profiles.py?)")
            continue
        renderizar_mapa(tif_path, titulo, etiqueta, os.path.join(figures_dir, png), regiones)

    print("=== FIGURAS GENERADAS ===")


if __name__ == '__main__':
    main()
