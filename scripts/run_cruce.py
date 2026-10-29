"""CLI / Etapa 9: cruce del mapa de aptitud RF con el rendimiento físico (solarpv-rs).

Lee `config.yaml`, resuelve rutas y llama a `src.cruce_aptitud_rendimiento.cruzar`.
El mapa de rendimiento lo produce antes `scripts/run_solar_yield.py` (requiere el motor
Rust compilado); si no existe, esta etapa se omite con un aviso claro, sin romper el
pipeline —el motor es una dependencia externa opcional—.

Uso directo:
    python scripts/run_cruce.py --config config.yaml
"""

import os
import sys
import argparse
import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.cruce_aptitud_rendimiento import cruzar
from src.rutas import ruta_resultado, prefijo_rendimiento


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Cruce aptitud x rendimiento (Etapa 9)')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--zona', default=None,
                        help='Clave de config.yaml con otra zona (p. ej. "transferibilidad"): '
                             'cruza los artefactos de esa zona y escribe en su directorio.')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Rutas de entrada (mismas convenciones que el resto del pipeline). Ambos mapas deben
    # venir de la MISMA zona: cruzar la aptitud de una región con el rendimiento de otra
    # produciría un ranking sin correspondencia espacial.
    if args.zona and not config.get(args.zona):
        print(f"  [ERROR] config.yaml no tiene el bloque '{args.zona}'.")
        return 1

    aptitud_path = ruta_resultado(config, 'aptitud', 'mapa_probabilidad_aptitud.tif', args.zona)
    # El montaje fijo es el caso base para el cruce (T3 agregará el seguidor).
    rendimiento_path = prefijo_rendimiento(config, args.zona) + '_fijo_specific_yield.tif'

    # Umbral de aptitud: única fuente de verdad en postgis_validation.prob_min.
    umbral = config.get('postgis_validation', {}).get('prob_min', 0.70)

    if not os.path.exists(aptitud_path):
        print(f"  [OMITIDA] No existe el mapa de aptitud ({aptitud_path}). "
              "Corre el pipeline (etapa 5) primero.")
        return 0
    if not os.path.exists(rendimiento_path):
        print(f"  [OMITIDA] No existe el mapa de rendimiento ({rendimiento_path}). "
              "Genera primero con: python scripts/run_solar_yield.py --config config.yaml")
        return 0

    out_en_aptas = ruta_resultado(config, 'cruce', 'rendimiento_en_aptas.tif', args.zona)
    out_ranking = ruta_resultado(config, 'cruce', 'aptitud_x_rendimiento.tif', args.zona)
    out_json = ruta_resultado(config, 'cruce', 'cruce_aptitud_rendimiento.json', args.zona)

    print(f"Cruzando aptitud (>= {umbral}) x rendimiento...")
    stats = cruzar(aptitud_path, rendimiento_path, umbral, out_en_aptas, out_ranking, out_json)

    r, a = stats['rendimiento_region'], stats['rendimiento_aptas']
    print(f"  Celdas válidas: {stats['celdas_validas']:,} | aptas: {stats['celdas_aptas']:,} "
          f"({stats['pct_aptas']}%)")
    print(f"  Rendimiento región: media={r['media']} | aptas: media={a['media']} "
          f"(ganancia {stats['ganancia_aptas_pct']:+}%)")
    sens = stats.get('sensibilidad_formula_combinacion', {})
    if sens:
        print(f"  Sensibilidad a la fórmula (producto vs. media 50/50): "
              f"Spearman={sens['spearman_ranking_vs_alternativa']} | "
              f"overlap top-1%={sens['overlap_top_1pct_pct']}%")
    print(f"  Salidas: {os.path.basename(out_ranking)}, {os.path.basename(out_en_aptas)}, "
          f"{os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
