"""CLI de Análisis de Desacople Aptitud–Rendimiento (punto 2 de la retroalimentación).

Análisis aparte, FUERA del pipeline: no entrena nada. Cruza el mapa oficial de aptitud del
pipeline con el rendimiento físico de solarpv-rs v0.4.0 y calcula:
  1. Rendimiento medio y mediano regional vs. zonas aptas (P >= prob_min, 0,70 por defecto).
  2. Correlación de Spearman aptitud–rendimiento por celda y agregada en bloques de 30 km
     del SBCV, con intervalo de confianza al 95 % por bootstrap espacial de bloques.
  3. Tabla CSV consolidada y JSON de metadatos en data/results/analisis/desacople/.

Requiere que el pipeline se haya corrido con `control_fuga.excluir_dedicadas: true`: el
mapa de aptitud debe venir del modelo sin infraestructura dedicada (ver src/infraestructura.py).

Excluye del análisis las celdas con rendimiento <= 0 (sin dato de GHI observado) y las zonas
de exclusión territorial (SNASPE, masas lacustres, áreas pobladas).

Uso:
    python scripts/run_desacople_aptitud_rendimiento.py --config config.yaml
    python scripts/run_desacople_aptitud_rendimiento.py --rendimiento data/results/rendimiento/rendimiento_seguidor_specific_yield.tif
"""

import os
import sys
import json
import argparse

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.infraestructura import control_fuga_activo
from src.utils import _resolver_rutas, _ruta_abs
from src.rutas import ruta_resultado, prefijo_rendimiento, dir_analisis
from src.desacople_aptitud_rendimiento import (
    mascara_zonas_exclusion,
    alinear_y_cruzar,
    calcular_estadisticas_desacople,
    exportar_tabla_csv,
)


def main():
    parser = argparse.ArgumentParser(description='Análisis de Desacople Aptitud-Rendimiento')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo config.yaml')
    parser.add_argument('--rendimiento', default=None,
                        help='Ráster de rendimiento (kWh/kWp/año). Por defecto, el fijo del '
                             'pipeline: data/results/rendimiento/rendimiento_fijo_specific_yield.tif')
    parser.add_argument('--bloque-km', type=float, default=None,
                        help='Tamaño del bloque espacial en km (por defecto, el del SBCV: '
                             'validacion.tamano_bloque_km de config.yaml)')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Número de réplicas bootstrap (default: 1000)')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla aleatoria (default: 42)')
    args = parser.parse_args()

    print("=" * 80)
    print("ANÁLISIS DE DESACOPLE APTITUD-RENDIMIENTO (MAPA OFICIAL SIN FUGA)")
    print("=" * 80)

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])
    # Los bloques deben ser los del SBCV (misma grilla UTM anclada en el origen y mismo lado),
    # como pide el análisis: si cambia el tamaño en el config, el desacople lo sigue.
    if args.bloque_km is None:
        args.bloque_km = float(config.get('validacion', {}).get('tamano_bloque_km', 30))

    # El análisis solo tiene sentido sobre el mapa sin fuga: el mapa con infraestructura
    # dedicada premia la cercanía a la propia conexión de cada planta (causalidad inversa).
    if not control_fuga_activo(config):
        print("[ERROR] control_fuga.excluir_dedicadas no está activo en el config: el mapa de "
              "aptitud del pipeline sería el modelo con fuga. Actívalo y corre el pipeline.")
        return 1

    aptitud_path = ruta_resultado(config, 'aptitud', 'mapa_probabilidad_aptitud.tif')
    rendimiento_path = (_ruta_abs(args.rendimiento) if args.rendimiento
                        else prefijo_rendimiento(config) + '_fijo_specific_yield.tif')
    for ruta, que in ((aptitud_path, 'el mapa de aptitud'), (rendimiento_path, 'el mapa de rendimiento')):
        if not os.path.exists(ruta):
            print(f"[ERROR] No existe {que}: {ruta}. Corre el pipeline primero.")
            return 1

    dir_salida = dir_analisis(config, 'desacople')
    out_csv = os.path.join(dir_salida, 'desacople_aptitud_rendimiento.csv')
    out_json = os.path.join(dir_salida, 'desacople_aptitud_rendimiento.json')
    print(f"  Aptitud:     {aptitud_path}")
    print(f"  Rendimiento: {rendimiento_path}")

    # 1. Zonas de exclusión sobre la grilla de aptitud (las mismas del mapa del pipeline)
    print("\n[1/4] Rasterizando zonas de exclusión territorial...")
    vectores = config['paths']['raw']['vectores']
    mascara = mascara_zonas_exclusion(
        [vectores.get('snap'), vectores.get('masas_lacustres'), vectores.get('areas_pobladas')],
        aptitud_path)

    # 2. Alineación y cruce
    print("\n[2/4] Alineando rásters y cruzando aptitud vs. rendimiento físico...")
    umbral = config.get('postgis_validation', {}).get('prob_min', 0.70)
    cruce = alinear_y_cruzar(aptitud_path, rendimiento_path, umbral=umbral, mascara_exclusion=mascara)

    # 3. Cálculo estadístico y bootstrap
    print(f"\n[3/4] Calculando métricas, correlaciones de Spearman y Bootstrap de bloques ({args.bloque_km} km, B={args.n_bootstrap})...")
    try:
        resultados = calcular_estadisticas_desacople(
            cruce, tamano_bloque_km=args.bloque_km,
            n_bootstrap=args.n_bootstrap, random_state=args.random_state
        )
    except ValueError as e:
        print(f"\n[SIN CELDAS APTAS]\n{e}")
        return 1
    resultados['entradas'] = {
        'mapa_aptitud': aptitud_path,
        'rendimiento': rendimiento_path,
        'control_fuga_excluir_dedicadas': True,
    }

    # 4. Exportar CSV y JSON
    print("\n[4/4] Exportando tabla CSV consolidada y metadatos...")
    exportar_tabla_csv(resultados, out_csv)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("RESULTADOS DEL DESACOPLE APTITUD-RENDIMIENTO")
    print("=" * 90)
    r = resultados['resumen_rendimiento']
    reg = resultados['region']
    apt = resultados['aptas']

    print(f"Celdas válidas evaluadas:         {r['celdas_validas']:,}  "
          f"(excluidas: {r['celdas_excluidas_sin_ghi']:,} sin GHI, "
          f"{r['celdas_excluidas_zonas_exclusion']:,} en zonas de exclusión)")
    print(f"Celdas aptas (P >= {umbral:.2f}):           {r['celdas_aptas']:,} ({r['pct_aptas']}%)")
    print(f"Bloques espaciales de 30 km:      {reg['n_bloques_30km']} en región | {apt['n_bloques_30km']} con celdas aptas")
    print("-" * 90)
    print(f"Rendimiento regional (kWh/kWp):    media = {r['rendimiento_region']['media']} | mediana = {r['rendimiento_region']['mediana']}")
    print(f"Rendimiento en aptas (kWh/kWp):    media = {r['rendimiento_aptas']['media']} | mediana = {r['rendimiento_aptas']['mediana']}")
    print(f"Ganancia relativa en aptas:        media = {r['ganancia_media_pct']:+}% | mediana = {r['ganancia_mediana_pct']:+}%")
    print("-" * 90)
    sc = reg['spearman_celda']
    sb = reg['spearman_bloque']
    sca = apt['spearman_celda']
    sba = apt['spearman_bloque']
    print(f"Spearman por celda (regional):    rho = {sc['rho']:.4f} (IC 95%: [{sc['ic_95_inferior']:.4f}, {sc['ic_95_superior']:.4f}])")
    print(f"Spearman por bloque (regional):   rho = {sb['rho']:.4f} (IC 95%: [{sb['ic_95_inferior']:.4f}, {sb['ic_95_superior']:.4f}]) | p = {sb['p_valor']}")
    print(f"Spearman por celda (aptas):       rho = {sca['rho']:.4f} (IC 95%: [{sca['ic_95_inferior']:.4f}, {sca['ic_95_superior']:.4f}])")
    print(f"Spearman por bloque (aptas):      rho = {sba['rho']:.4f} (IC 95%: [{sba['ic_95_inferior']:.4f}, {sba['ic_95_superior']:.4f}]) | p = {sba['p_valor']}")
    print("-" * 90)
    print(f"Tabla CSV generada en:            {out_csv}")
    print(f"JSON con metadatos en:            {out_json}")
    print("=" * 90)
    return 0


if __name__ == '__main__':
    sys.exit(main())
