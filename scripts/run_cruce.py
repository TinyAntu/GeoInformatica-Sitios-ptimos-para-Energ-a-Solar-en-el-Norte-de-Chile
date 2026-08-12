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


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Cruce aptitud x rendimiento (Etapa 9)')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Rutas de entrada (mismas convenciones que el resto del pipeline).
    aptitud_path = _ruta_abs('data/results/mapa_probabilidad_aptitud.tif')
    cfg_solar = config.get('solarpv', {})
    # El montaje fijo es el caso base para el cruce (T3 agregará el seguidor).
    rendimiento_path = _ruta_abs(cfg_solar.get('out_prefix', 'data/results/rendimiento')
                                 + '_fijo_specific_yield.tif')

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

    out_en_aptas = _ruta_abs('data/results/rendimiento_en_aptas.tif')
    out_ranking = _ruta_abs('data/results/aptitud_x_rendimiento.tif')
    out_json = _ruta_abs('data/results/cruce_aptitud_rendimiento.json')

    print(f"Cruzando aptitud (>= {umbral}) x rendimiento...")
    stats = cruzar(aptitud_path, rendimiento_path, umbral, out_en_aptas, out_ranking, out_json)

    r, a = stats['rendimiento_region'], stats['rendimiento_aptas']
    print(f"  Celdas válidas: {stats['celdas_validas']:,} | aptas: {stats['celdas_aptas']:,} "
          f"({stats['pct_aptas']}%)")
    print(f"  Rendimiento región: media={r['media']} | aptas: media={a['media']} "
          f"(ganancia {stats['ganancia_aptas_pct']:+}%)")
    print(f"  Salidas: {os.path.basename(out_ranking)}, {os.path.basename(out_en_aptas)}, "
          f"{os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
