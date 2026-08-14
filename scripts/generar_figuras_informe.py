"""Renderiza los GeoTIFF de resultados como figuras listas para el informe (PEP1 + PEP2/T8).

Cada mapa incluye los 7 elementos cartográficos obligatorios (clase 9 / rúbrica): título,
leyenda (colorbar o leyenda discreta), escala gráfica, norte, fuente/autor, CRS y fecha.

Paletas — todas aptas para daltonismo (deuteranopia/protanopia, la forma más común):
  - Continuas (aptitud, rendimiento, cruce): familia perceptualmente uniforme de matplotlib
    (viridis/plasma/magma/cividis), diseñada y validada para daltonismo. Nunca rainbow/jet.
  - Discretas (consenso, variable dominante SHAP): NO se usa RdYlGn (rojo-verde clásico,
    ilegible para deuteranopia/protanopia) ni tab10 (pares poco distinguibles). Se usa
    cividis muestreada (consenso, variable ordinal) y la paleta Okabe-Ito (dominante,
    variable nominal) — estándar recomendado en visualización científica accesible.

Uso:
    python scripts/generar_figuras_informe.py

Entradas (generadas por el pipeline y los scripts de T1/T2/T3/T6-espacial/Brecha 6):
    data/results/mapa_probabilidad_aptitud.tif, aptitud_conservador.tif, aptitud_agresivo.tif
    data/results/rendimiento_{fijo,seguidor}_specific_yield.tif
    data/results/aptitud_x_rendimiento.tif
    data/results/consenso_perfiles.tif
    data/results/shap_espacial_dominante.tif
    data/results/comparacion_montaje.json
Salidas: figures/*.png
"""
import os
import sys
import json
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
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

MAX_PIXELES_LADO = 2600  # tope de resolución de lectura: suficiente para 300 dpi en media página
COLOR_TINTA = '#333333'  # texto y ejes en tinta neutra (la identidad la lleva el raster)

# Paleta Okabe & Ito (2008): colores categóricos distinguibles para las formas más comunes
# de daltonismo. Referencia estándar en visualización científica accesible.
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
            '#0072B2', '#D55E00', '#CC79A7']


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


def calcular_percentil_combinado(paths, p_low=2, p_high=98):
    """Percentiles combinados de varios rasters, para compartir una misma escala de color
    entre mapas comparables (p. ej. rendimiento fijo vs. seguidor)."""
    partes = [leer_raster_reducido(p)[0].compressed() for p in paths]
    todos = np.concatenate(partes)
    return float(np.percentile(todos, p_low)), float(np.percentile(todos, p_high))


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


def dibujar_leyenda_discreta(ax, colores, etiquetas, titulo_leyenda):
    """Leyenda de parches para mapas categóricos/ordinales (reemplaza el colorbar continuo)."""
    handles = [patches.Patch(facecolor=c, edgecolor=COLOR_TINTA, linewidth=0.4, label=e)
              for c, e in zip(colores, etiquetas)]
    leg = ax.legend(handles=handles, loc='lower right', fontsize=8, title=titulo_leyenda,
                    title_fontsize=8.5, frameon=True, framealpha=0.92, edgecolor='#bbbbbb')
    leg.get_frame().set_facecolor('white')


def renderizar_mapa(tif_path, titulo, etiqueta_leyenda, out_png, regiones,
                    cmap='viridis', vmin=0.0, vmax=1.0,
                    discreto=False, categorias=None, colores_discretos=None,
                    ocultar_bajo=None, nota_extra=None):
    """Renderiza un GeoTIFF con los 7 elementos cartográficos obligatorios.

    `discreto=True` dibuja una leyenda de parches (categorías/orden) en vez de colorbar
    continuo; usar con `colores_discretos` (uno por categoría, vmin..vmax enteros inclusive)
    y `categorias` (etiquetas en el mismo orden). `ocultar_bajo` enmascara valores válidos
    por debajo del umbral (p. ej. el 0 = "no apto" del mapa de consenso).
    """
    print(f"Renderizando {os.path.basename(tif_path)} ...")
    datos, extent = leer_raster_reducido(tif_path)
    if ocultar_bajo is not None:
        datos = np.ma.masked_less(datos, ocultar_bajo)

    fig, ax = plt.subplots(figsize=(8.5, 10))

    if discreto:
        cmap_obj = ListedColormap(colores_discretos)
        cmap_obj.set_bad(alpha=0.0)
        bounds = np.arange(vmin - 0.5, vmax + 1.5, 1)
        norm = BoundaryNorm(bounds, cmap_obj.N)
        im = ax.imshow(datos, extent=extent, origin='upper', cmap=cmap_obj, norm=norm,
                       interpolation='nearest')
    else:
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)  # nodata transparente
        im = ax.imshow(datos, extent=extent, origin='upper', cmap=cmap_obj,
                       vmin=vmin, vmax=vmax, interpolation='nearest')

    regiones.boundary.plot(ax=ax, color=COLOR_TINTA, linewidth=0.7, zorder=4)

    # Encuadre al área de estudio: algunos rasters cubren más territorio que las 2 regiones
    # y sin este recorte quedarían como una esquina diminuta en un lienzo vacío.
    minx, miny, maxx, maxy = regiones.total_bounds
    margen_x, margen_y = 0.03 * (maxx - minx), 0.03 * (maxy - miny)
    ax.set_xlim(minx - margen_x, maxx + margen_x)
    ax.set_ylim(miny - margen_y, maxy + margen_y)

    # Leyenda: colorbar continuo o parches discretos, según el tipo de dato
    if discreto:
        dibujar_leyenda_discreta(ax, colores_discretos, categorias, etiqueta_leyenda)
    else:
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(etiqueta_leyenda, fontsize=10, color=COLOR_TINTA)
        cbar.ax.tick_params(labelsize=8, colors=COLOR_TINTA)

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
           f"{date.today().strftime('%d-%m-%Y')}")
    if nota_extra:
        pie += f"  |  {nota_extra}"
    fig.text(0.5, 0.015, pie, ha='center', va='bottom', fontsize=7.5, color=COLOR_TINTA)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Guardado: {out_png}")


def renderizar_comparacion_montaje(json_path, out_png):
    """Gráfico de barras fijo vs. seguidor (no es un mapa: sin norte/escala/CRS, pero con
    fuente y fecha). Colores Okabe-Ito azul/vermellón: par de alto contraste, distinguible
    también en daltonismo rojo-verde."""
    print(f"Renderizando comparación fijo vs. seguidor ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fijo = data['rendimiento_fijo_aptas']
    seguidor = data['rendimiento_seguidor_aptas']

    metricas = ['min', 'media', 'p50', 'p90', 'max']
    etiquetas_metricas = ['Mín', 'Media', 'Mediana', 'p90', 'Máx']
    vals_fijo = [fijo[m] for m in metricas]
    vals_seguidor = [seguidor[m] for m in metricas]

    color_fijo, color_seguidor = '#0072B2', '#D55E00'  # azul / vermellón (Okabe-Ito)

    x = np.arange(len(metricas))
    ancho = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - ancho / 2, vals_fijo, ancho, label='Montaje fijo (tilt 23°)', color=color_fijo)
    b2 = ax.bar(x + ancho / 2, vals_seguidor, ancho, label='Seguidor de un eje', color=color_seguidor)
    ax.bar_label(b1, fmt='%.0f', fontsize=7.5, color=COLOR_TINTA, padding=2)
    ax.bar_label(b2, fmt='%.0f', fontsize=7.5, color=COLOR_TINTA, padding=2)

    # Margen superior explícito: dos las barras y la leyenda quepan sin recortarse ni
    # solaparse (el único hueco libre real es sobre la columna "Mín", la más baja).
    ax.set_ylim(0, max(vals_fijo + vals_seguidor) * 1.18)

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas_metricas, fontsize=9)
    ax.set_ylabel('Rendimiento (kWh/kWp/año)', fontsize=10, color=COLOR_TINTA)
    ax.set_title('Rendimiento fotovoltaico: montaje fijo vs. seguidor de un eje\n'
                 '(zonas aptas, probabilidad RF ≥ 0.70)', fontsize=11.5, color=COLOR_TINTA, pad=10)

    ganancia = data.get('ganancia_seguidor_media_pct')
    ref_autor = data.get('referencia_autor_pct')
    if ganancia is not None:
        texto = f"Ganancia del seguidor: {ganancia:+.1f}%"
        if ref_autor is not None:
            texto += f"\n(referencia del autor: +{ref_autor:.0f}%)"
        ax.text(0.02, 0.97, texto, transform=ax.transAxes, ha='left', va='top', fontsize=9,
                color=COLOR_TINTA, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#bbbbbb', alpha=0.92))

    # Leyenda bajo la caja de ganancia, en el mismo cuadrante libre (sobre "Mín").
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.80),
             frameon=True, framealpha=0.92, edgecolor='#bbbbbb')
    ax.tick_params(labelsize=9, colors=COLOR_TINTA)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bbbbbb')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.grid(axis='y', color='#e5e5e5', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    pie = ("Fuente: solarpv-rs (motor Rust, validado contra pvlib ≤0.2 %), grilla 300 m. "
           "Elaboración propia, Grupo Solar (USACH).\n"
           f"Fecha de elaboración: {date.today().strftime('%d-%m-%Y')}")
    fig.text(0.5, 0.01, pie, ha='center', va='bottom', fontsize=7.5, color=COLOR_TINTA)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Guardado: {out_png}")


def main():
    print("=== GENERANDO FIGURAS CARTOGRÁFICAS PARA EL INFORME (PEP1 + T8) ===")
    with open(os.path.join(directorio_raiz, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    regiones_path = os.path.join(directorio_raiz, config['paths']['raw']['vectores']['regiones'])
    regiones = gpd.read_file(regiones_path)
    regiones = regiones[regiones['REGION'].isin(['Antofagasta', 'Atacama'])].to_crs(epsg=32719)

    figures_dir = os.path.join(directorio_raiz, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    results_dir = os.path.join(directorio_raiz, 'data', 'results')

    # --- Mapas de aptitud (perfiles) — paleta viridis, sin cambios respecto de PEP1 ---
    mapas_aptitud = [
        ('mapa_probabilidad_aptitud.tif',
         'Aptitud fotovoltaica — probabilidad Random Forest (perfil balanceado / ML)',
         'Probabilidad de aptitud (0–1)', 'mapa_aptitud_rf.png'),
        ('aptitud_conservador.tif',
         'Aptitud fotovoltaica — WLC con pesos AHP, perfil conservador (CR = 0.001)',
         'Índice de aptitud WLC (0–1)', 'mapa_aptitud_conservador.png'),
        ('aptitud_agresivo.tif',
         'Aptitud fotovoltaica — WLC con pesos AHP, perfil agresivo (CR = 0.021)',
         'Índice de aptitud WLC (0–1)', 'mapa_aptitud_agresivo.png'),
    ]
    for tif, titulo, etiqueta, png in mapas_aptitud:
        tif_path = os.path.join(results_dir, tif)
        if not os.path.exists(tif_path):
            print(f"  [AVISO] No existe {tif_path}; se omite.")
            continue
        renderizar_mapa(tif_path, titulo, etiqueta, os.path.join(figures_dir, png), regiones,
                        cmap='viridis', vmin=0.0, vmax=1.0)

    # --- Rendimiento físico (motor solarpv-rs) — paleta plasma, escala compartida (T1/T3) ---
    prefijo = config.get('solarpv', {}).get('out_prefix', 'data/results/rendimiento')
    tif_fijo = os.path.join(directorio_raiz, prefijo + '_fijo_specific_yield.tif')
    tif_seguidor = os.path.join(directorio_raiz, prefijo + '_seguidor_specific_yield.tif')
    existentes = [p for p in (tif_fijo, tif_seguidor) if os.path.exists(p)]
    if existentes:
        vmin_r, vmax_r = calcular_percentil_combinado(existentes)
        nota_motor = "Motor: solarpv-rs (validado vs. pvlib ≤0.2 %), grilla 300 m."
        if os.path.exists(tif_fijo):
            renderizar_mapa(tif_fijo, 'Rendimiento fotovoltaico — montaje fijo (tilt 23°)',
                            'Rendimiento (kWh/kWp/año)',
                            os.path.join(figures_dir, 'mapa_rendimiento_fijo.png'), regiones,
                            cmap='plasma', vmin=vmin_r, vmax=vmax_r, nota_extra=nota_motor)
        if os.path.exists(tif_seguidor):
            renderizar_mapa(tif_seguidor, 'Rendimiento fotovoltaico — seguidor de un eje',
                            'Rendimiento (kWh/kWp/año)',
                            os.path.join(figures_dir, 'mapa_rendimiento_seguidor.png'), regiones,
                            cmap='plasma', vmin=vmin_r, vmax=vmax_r, nota_extra=nota_motor)
    else:
        print("  [AVISO] No hay mapas de rendimiento; corre scripts/run_solar_yield.py primero.")

    # --- Cruce aptitud × rendimiento (T2) — paleta magma ---
    tif_cruce = os.path.join(results_dir, 'aptitud_x_rendimiento.tif')
    if os.path.exists(tif_cruce):
        renderizar_mapa(tif_cruce, 'Ranking combinado: aptitud × rendimiento físico',
                        'Score combinado (0–1)',
                        os.path.join(figures_dir, 'mapa_cruce_aptitud_rendimiento.png'), regiones,
                        cmap='magma', vmin=0.0, vmax=1.0,
                        nota_extra="Score = probabilidad RF × rendimiento normalizado (p1-p99).")
    else:
        print("  [AVISO] No existe el cruce aptitud×rendimiento; corre scripts/run_cruce.py.")

    # --- Consenso entre perfiles (Brecha 6) — cividis muestreada, discreta y ordinal ---
    tif_consenso = os.path.join(results_dir, 'consenso_perfiles.tif')
    if os.path.exists(tif_consenso):
        colores_consenso = [to_hex(plt.get_cmap('cividis')(t)) for t in (0.0, 0.5, 1.0)]
        renderizar_mapa(tif_consenso, 'Consenso vs. divergencia entre perfiles de inversión',
                        'Perfiles que consideran apta la celda',
                        os.path.join(figures_dir, 'mapa_consenso_perfiles.png'), regiones,
                        discreto=True, vmin=1, vmax=3,
                        categorias=['1 perfil (divergencia)', '2 perfiles (divergencia)',
                                    '3 perfiles (consenso)'],
                        colores_discretos=colores_consenso, ocultar_bajo=1,
                        nota_extra="'Apto' = top 10 % de cada perfil (comparación relativa entre escalas distintas).")
    else:
        print("  [AVISO] No existe el mapa de consenso; corre scripts/run_consenso.py.")

    # --- Variable dominante SHAP (Brecha 8) — paleta Okabe-Ito, discreta y nominal ---
    tif_dominante = os.path.join(results_dir, 'shap_espacial_dominante.tif')
    if os.path.exists(tif_dominante):
        etiquetas_shap = ['Pendiente', 'GHI', 'Elevación', 'Northness',
                          'Dist. transmisión', 'Dist. almacenam.', 'Dist. subestaciones']
        renderizar_mapa(tif_dominante, 'Variable dominante en la decisión de aptitud (SHAP espacial)',
                        'Variable con mayor |SHAP|',
                        os.path.join(figures_dir, 'mapa_shap_dominante.png'), regiones,
                        discreto=True, vmin=0, vmax=6,
                        categorias=etiquetas_shap, colores_discretos=OKABE_ITO[:7],
                        nota_extra="TreeExplainer sobre model_rf.pkl (sin re-entrenar), grilla 500 m.")
    else:
        print("  [AVISO] No existe el mapa de variable dominante; corre scripts/run_shap_spatial.py.")

    # --- Comparación fijo vs. seguidor (T3) — gráfico de barras, no es un mapa ---
    json_comparacion = os.path.join(results_dir, 'comparacion_montaje.json')
    if os.path.exists(json_comparacion):
        renderizar_comparacion_montaje(json_comparacion,
                                       os.path.join(figures_dir, 'comparacion_fijo_vs_seguidor.png'))
    else:
        print("  [AVISO] No existe comparacion_montaje.json; corre scripts/run_comparacion_montaje.py.")

    print("=== FIGURAS GENERADAS ===")


if __name__ == '__main__':
    main()
