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


def renderizar_metricas_validacion(json_topk, json_validacion, out_png):
    """Panel de las 5 métricas comprometidas en PEP1 §6.5 (no es un mapa: sin norte/escala).

    El mensaje central no es "cumple / no cumple" sino que el umbral de Precisión@K1 era
    INALCANZABLE por construcción: con ~144 km² de huella instalada sobre ~203.000 km² de
    área de estudio, ni un modelo perfecto supera el 7,1 % en el top 1 %. Por eso el panel C
    dibuja el techo teórico junto al valor logrado y al umbral, en la misma escala.
    """
    print("Renderizando panel de métricas de validación ...")
    with open(json_topk, 'r', encoding='utf-8') as f:
        topk = json.load(f)
    with open(json_validacion, 'r', encoding='utf-8') as f:
        val = json.load(f)

    loro = topk['out_of_sample_loro']['combinado_por_k']
    insample = topk['in_sample_por_region']['combinado_por_k']
    contraste = topk['contraste_umbrales_pep1']

    color_ok, color_falla = '#009E73', '#D55E00'      # verde / vermellón (Okabe-Ito)
    color_in, color_out = '#56B4E9', '#0072B2'        # celeste / azul
    color_techo = '#cccccc'

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15.5, 5.2))

    # --- Panel A: contraste con los 5 umbrales -------------------------------------
    # Cada fila se normaliza como "razón de cumplimiento": >=1 significa que se cumple.
    # Para los criterios donde MENOR es mejor (gap, Brier) la razón se invierte, de modo
    # que la lectura del eje sea la misma en las cinco filas.
    lorocv = val['leave_one_region_out_cv']
    sbcv = val['spatial_block_cv']
    filas = [
        ('AUC LORO-CV',    lorocv['auc_mean'],            0.75, 'mayor'),
        ('Gap AUC',        lorocv['gap_auc'],             0.15, 'menor'),
        ('Brier (SBCV)',   sbcv['brier_mean'],            0.20, 'menor'),
        ('Recall@K3',      contraste['recall_k3']['valor'],    0.85, 'mayor'),
        ('Precisión@K1',   contraste['precision_k1']['valor'], 0.15, 'mayor'),
    ]
    etiquetas, razones, colores, anotaciones = [], [], [], []
    for nombre, valor, umbral, sentido in filas:
        if sentido == 'mayor':
            razon, cumple, simbolo = valor / umbral, valor >= umbral, '≥'
        else:
            # Un valor ~0 daría una razón enorme; se acota para no romper la escala.
            razon = min(umbral / valor, 8.0) if valor > 0 else 8.0
            cumple, simbolo = valor <= umbral, '≤'
        etiquetas.append(nombre)
        razones.append(razon)
        colores.append(color_ok if cumple else color_falla)
        anotaciones.append(f"{valor:.4g}  (umbral {simbolo} {umbral:g})")

    y = np.arange(len(filas))
    ax_a.barh(y, razones, color=colores, height=0.62, zorder=3)
    # La línea punteada en x=1 no se rotula: el propio label del eje X ya dice qué significa,
    # y un texto flotante aquí se solapa con el título del panel.
    ax_a.axvline(1.0, color=COLOR_TINTA, linestyle='--', linewidth=1.2, zorder=4)
    for i, texto in enumerate(anotaciones):
        ax_a.text(max(razones[i], 1.0) + 0.12, i, texto, va='center', fontsize=8,
                  color=COLOR_TINTA)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(etiquetas, fontsize=9)
    ax_a.set_xlim(0, 8.9)
    ax_a.set_xlabel('Razón de cumplimiento  (≥ 1 cumple el umbral)', fontsize=9,
                    color=COLOR_TINTA)
    ax_a.set_title('A. Umbrales establecidos', fontsize=10.5,
                   color=COLOR_TINTA, pad=8)

    # --- Panel B: recall y precisión en función de K --------------------------------
    ks = sorted(float(k) for k in loro.keys())
    recall_out = [loro[f"{k:g}"]['recall'] * 100 for k in ks]
    recall_in = [insample[f"{k:g}"]['recall'] * 100 for k in ks]
    prec_out = [loro[f"{k:g}"]['precision_area']['precision'] * 100 for k in ks]

    ax_b.plot(ks, recall_in, 'o-', color=color_in, linewidth=1.8, markersize=4.5,
              label='Recall in-sample (A-500)')
    ax_b.plot(ks, recall_out, 'o-', color=color_out, linewidth=2.0, markersize=5,
              label='Recall out-of-sample (LORO)')
    ax_b.plot(ks, prec_out, 's--', color=color_falla, linewidth=1.6, markersize=4,
              label='Precisión-área out-of-sample')
    for k_marca in (1.0, 3.0):
        ax_b.axvline(k_marca, color='#bbbbbb', linewidth=0.9, linestyle=':', zorder=0)
    ax_b.set_xlabel('K — porcentaje del área de estudio seleccionada (%)', fontsize=9,
                    color=COLOR_TINTA)
    ax_b.set_ylabel('Porcentaje (%)', fontsize=9, color=COLOR_TINTA)
    ax_b.set_title('B. Recall y precisión en función de K', fontsize=10.5,
                   color=COLOR_TINTA, pad=8)
    ax_b.legend(fontsize=8, loc='center right', frameon=True, framealpha=0.92,
                edgecolor='#bbbbbb')

    # --- Panel C: logrado vs. techo vs. umbral de PEP1 ------------------------------
    ks_c = ['1', '3']
    x = np.arange(len(ks_c))
    ancho = 0.34
    techos = [loro[k]['precision_area']['precision_techo_modelo_perfecto'] * 100 for k in ks_c]
    log_in = [insample[k]['precision_area']['precision'] * 100 for k in ks_c]
    log_out = [loro[k]['precision_area']['precision'] * 100 for k in ks_c]

    # Techo como barra fantasma ancha detrás: deja ver cuánto del máximo se alcanzó.
    ax_c.bar(x, techos, ancho * 2.5, color=color_techo, zorder=1,
             label='Techo teórico')
    b_in = ax_c.bar(x - ancho / 2, log_in, ancho, color=color_in, zorder=3,
                    label='In-sample')
    b_out = ax_c.bar(x + ancho / 2, log_out, ancho, color=color_out, zorder=3,
                     label='Out-of-sample (LORO)')
    ax_c.bar_label(b_in, fmt='%.2f', fontsize=7.5, color=COLOR_TINTA, padding=2)
    ax_c.bar_label(b_out, fmt='%.2f', fontsize=7.5, color=COLOR_TINTA, padding=2)

    umbral_pep1 = contraste['precision_k1']['umbral_pep1'] * 100
    # xlim explícito: fija el borde izquierdo para poder anclar ahí la etiqueta del umbral
    # sin que se recorte (con el xlim automático quedaba fuera del área dibujable).
    ax_c.set_xlim(-0.5, len(ks_c) - 0.5)
    ax_c.axhline(umbral_pep1, color=color_falla, linestyle='--', linewidth=1.6, zorder=4)
    # Debajo de la línea: la banda sobre ella la ocupa la leyenda, y encima del umbral no
    # hay nada que anotar (ningún valor llega ahí, que es justamente el punto del panel).
    ax_c.text(-0.45, umbral_pep1 - 0.35, f'umbral = {umbral_pep1:.0f} %',
              ha='left', va='top', fontsize=8.5, color=color_falla, fontweight='bold')

    for i, k in enumerate(ks_c):
        pct = loro[k]['precision_area']['pct_del_techo_alcanzado']
        # Holgura amplia sobre la barra fantasma: con un margen chico la anotación chocaba
        # con las cifras que bar_label pone encima de las barras de valor logrado.
        ax_c.text(i, techos[i] + 0.9, f"{pct:.0f} % del techo", ha='center', va='bottom',
                  fontsize=8.5, color=COLOR_TINTA, fontweight='bold')

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f'K = {k} %' for k in ks_c], fontsize=9)
    ax_c.set_ylim(0, umbral_pep1 * 1.22)
    ax_c.set_ylabel('Precisión-área (%)', fontsize=9, color=COLOR_TINTA)
    ax_c.set_title('C. El umbral era inalcanzable por construcción', fontsize=10.5,
                   color=COLOR_TINTA, pad=8)
    # Centro-derecha: el único cuadrante libre (arriba está la línea del umbral, abajo las
    # barras, y arriba a la izquierda la etiqueta del umbral).
    ax_c.legend(fontsize=7.5, loc='center right', frameon=True, framealpha=0.92,
                edgecolor='#bbbbbb')

    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(labelsize=8.5, colors=COLOR_TINTA)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bbbbbb')
        ax.spines['bottom'].set_color('#bbbbbb')
        ax.set_axisbelow(True)
    ax_a.grid(axis='x', color='#e5e5e5', linewidth=0.7)
    ax_b.grid(color='#e5e5e5', linewidth=0.7)
    ax_c.grid(axis='y', color='#e5e5e5', linewidth=0.7)
    
    ha_mw = topk.get('supuesto_huella_ha_por_mw')
    n_plantas = topk['out_of_sample_loro']['n_plantas_evaluadas']
    resolucion = topk['out_of_sample_loro']['resolucion_m']
    pie = (
        "Fuente de datos: catastro de instalaciones de generación eléctrica en operación "
        f"(Ministerio de Energía, Chile; 716 registros, {n_plantas} en la zona de estudio, "
        "en operación hasta 11-2025).\n"
        f"Método: Random Forest (model_rf.pkl) validado con Leave-One-Region-Out CV sobre "
        f"{n_plantas} plantas, huella reconstruida por potencia instalada ({ha_mw} ha/MWac; "
        "rango publicado 1,45–3,60 según LBNL 2022 y NREL 2013).\n"
        "CRS: EPSG:32719 (WGS 84 / UTM zona 19S)  |  "
        f"Grilla de análisis: {resolucion} m  |  Elaboración propia, Grupo Solar (USACH)  |  "
        f"Fecha de elaboración: {date.today().strftime('%d-%m-%Y')}"
    )
    fig.text(0.5, 0.005, pie, ha='center', va='bottom', fontsize=7.5, color=COLOR_TINTA)

    fig.suptitle('Validación del modelo de aptitud frente a los umbrales establecidos',
                 fontsize=12.5, color=COLOR_TINTA, y=0.99)
    # Margen inferior ampliado (0.075 -> 0.10) para que la tercera línea del pie no pise el
    # eje X del panel A.
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
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

    # --- Métricas de validación vs. umbrales de PEP1 — panel de 3 gráficos, no es un mapa ---
    json_topk = os.path.join(results_dir, 'metricas_topk.json')
    json_validacion = os.path.join(results_dir, 'validacion_espacial.json')
    if os.path.exists(json_topk) and os.path.exists(json_validacion):
        renderizar_metricas_validacion(json_topk, json_validacion,
                                       os.path.join(figures_dir, 'metricas_validacion.png'))
    else:
        print("  [AVISO] Faltan metricas_topk.json o validacion_espacial.json; "
              "corre scripts/run_metricas_topk.py.")

    print("=== FIGURAS GENERADAS ===")


if __name__ == '__main__':
    main()
