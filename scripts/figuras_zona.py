"""Figuras cartográficas de una zona evaluada por transferencia (no la de entrenamiento).

Reutiliza `renderizar_mapa` de scripts/generar_figuras_informe.py, así que las figuras salen
con los mismos 7 elementos cartográficos obligatorios y la misma paleta que las del informe,
y son por tanto comparables lado a lado con las del norte.

No se parametriza `generar_figuras_informe.py` completo a propósito: la mitad de sus figuras
(comparación de montaje, métricas vs. umbrales de PEP1, cruce con rendimiento) dependen de
artefactos que una corrida de transferencia no produce.

Uso:
    python scripts/figuras_zona.py --config config.yaml --zona transferibilidad
"""

import os
import sys
import argparse

import yaml
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from scripts.generar_figuras_informe import (
    renderizar_mapa, calcular_percentil_combinado, OKABE_ITO,
)


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--zona', default='transferibilidad')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    bloque = config.get(args.zona) or {}
    if not bloque:
        raise SystemExit(f"config.yaml no tiene el bloque '{args.zona}'.")

    nombres = bloque['regiones']
    etiqueta_zona = ' + '.join(nombres)
    results_dir = _ruta_abs(bloque.get('dir_resultados', f'data/results/{args.zona}'))
    figures_dir = _ruta_abs(bloque.get('dir_figuras', f'figures/{args.zona}'))
    os.makedirs(figures_dir, exist_ok=True)

    regiones = gpd.read_file(_ruta_abs(config['paths']['raw']['vectores']['regiones']))
    regiones = regiones[regiones['REGION'].isin(nombres)].to_crs(epsg=32719)

    # La nota va en todas las figuras: sin ella, un lector podría creer que el modelo se
    # ajustó a esta región. El punto del ejercicio es exactamente lo contrario.
    nota_modelo = ("Modelo entrenado en Antofagasta+Atacama, aplicado sin reentrenar. "
                   f"Grilla {bloque.get('resolucion_m', 500)} m.")

    mapas = [
        ('mapa_probabilidad_aptitud.tif',
         f'Aptitud fotovoltaica en {etiqueta_zona} — Random Forest transferido',
         'Probabilidad de aptitud (0–1)', 'mapa_aptitud_rf.png', 'viridis'),
        ('aptitud_conservador.tif',
         f'Aptitud fotovoltaica en {etiqueta_zona} — WLC/AHP, perfil conservador',
         'Índice de aptitud WLC (0–1)', 'mapa_aptitud_conservador.png', 'viridis'),
        ('aptitud_agresivo.tif',
         f'Aptitud fotovoltaica en {etiqueta_zona} — WLC/AHP, perfil agresivo',
         'Índice de aptitud WLC (0–1)', 'mapa_aptitud_agresivo.png', 'viridis'),
    ]
    for tif, titulo, etiqueta, png, cmap in mapas:
        ruta = os.path.join(results_dir, tif)
        if not os.path.exists(ruta):
            print(f"  [AVISO] No existe {ruta}; se omite.")
            continue
        # Los perfiles AHP no llevan la nota del modelo: no dependen del RF entrenado.
        nota = nota_modelo if tif.startswith('mapa_probabilidad') else None
        renderizar_mapa(ruta, titulo, etiqueta, os.path.join(figures_dir, png), regiones,
                        cmap=cmap, vmin=0.0, vmax=1.0, nota_extra=nota,
                        recortar_a_regiones=True)

    # --- Rendimiento físico (motor solarpv-rs) ---
    # Escala por percentiles p2-p98 sobre la propia zona, NO compartida con la zona de
    # estudio: son rangos de latitud distintos y forzar una escala común aplastaría el
    # contraste interno de la región más homogénea. Se declara en el pie de la figura.
    tif_rend = os.path.join(results_dir, 'rendimiento_fijo_specific_yield.tif')
    if os.path.exists(tif_rend):
        vmin_r, vmax_r = calcular_percentil_combinado([tif_rend], regiones=regiones)
        renderizar_mapa(tif_rend,
                        f'Rendimiento fotovoltaico en {etiqueta_zona} — montaje fijo (tilt 23°)',
                        'Rendimiento (kWh/kWp/año)',
                        os.path.join(figures_dir, 'mapa_rendimiento_fijo.png'), regiones,
                        cmap='plasma', vmin=vmin_r, vmax=vmax_r,
                        nota_extra="Motor: solarpv-rs, grilla 300 m. Escala p2–p98 propia de "
                                   "esta zona (no comparable en color con otras regiones).",
                        recortar_a_regiones=True)
    else:
        print("  [AVISO] No existe el mapa de rendimiento; corre run_zona.py sin --sin-motor.")

    # --- Cruce aptitud x rendimiento ---
    tif_cruce = os.path.join(results_dir, 'aptitud_x_rendimiento.tif')
    if os.path.exists(tif_cruce):
        renderizar_mapa(tif_cruce,
                        f'Ranking combinado en {etiqueta_zona}: aptitud × rendimiento físico',
                        'Score combinado (0–1)',
                        os.path.join(figures_dir, 'mapa_cruce_aptitud_rendimiento.png'), regiones,
                        cmap='magma', vmin=0.0, vmax=1.0,
                        nota_extra="Score = probabilidad RF × rendimiento normalizado (p1–p99). "
                                   + nota_modelo,
                        recortar_a_regiones=True)
    else:
        print("  [AVISO] No existe el cruce aptitud×rendimiento.")

    # --- Consenso entre perfiles ---
    tif_consenso = os.path.join(results_dir, 'consenso_perfiles.tif')
    if os.path.exists(tif_consenso):
        colores = [to_hex(plt.get_cmap('cividis')(t)) for t in (0.0, 0.5, 1.0)]
        renderizar_mapa(tif_consenso,
                        f'Consenso vs. divergencia entre perfiles — {etiqueta_zona}',
                        'Perfiles que consideran apta la celda',
                        os.path.join(figures_dir, 'mapa_consenso_perfiles.png'), regiones,
                        discreto=True, vmin=1, vmax=3,
                        categorias=['1 perfil (divergencia)', '2 perfiles (divergencia)',
                                    '3 perfiles (consenso)'],
                        colores_discretos=colores, ocultar_bajo=1,
                        nota_extra="'Apto' = top 10 % de cada perfil (comparación relativa).",
                        recortar_a_regiones=True)
    else:
        print("  [AVISO] No existe el mapa de consenso.")

    # --- Variable dominante SHAP ---
    tif_dom = os.path.join(results_dir, 'shap_espacial_dominante.tif')
    if os.path.exists(tif_dom):
        etiquetas = ['Pendiente', 'GHI', 'Elevación', 'Northness',
                     'Dist. transmisión', 'Dist. almacenam.', 'Dist. subestaciones']
        renderizar_mapa(tif_dom,
                        f'Variable dominante en la decisión de aptitud — {etiqueta_zona}',
                        'Variable con mayor |SHAP|',
                        os.path.join(figures_dir, 'mapa_shap_dominante.png'), regiones,
                        discreto=True, vmin=0, vmax=6,
                        categorias=etiquetas, colores_discretos=OKABE_ITO[:7],
                        nota_extra="TreeExplainer sobre model_rf.pkl (sin reentrenar).",
                        recortar_a_regiones=True)
    else:
        print("  [AVISO] No existe el mapa de variable dominante.")

    print(f"=== FIGURAS DE {etiqueta_zona.upper()} EN {os.path.relpath(figures_dir, directorio_raiz)} ===")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
