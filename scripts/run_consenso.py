"""CLI / Etapa 10: consenso vs. divergencia entre perfiles de aptitud (Brecha 6, punto 4).

Lee config.yaml y cruza los 3 mapas de perfil (conservador, balanceado/RF, agresivo) para
identificar zonas aptas en los 3 (consenso) vs. en 1–2 (divergencia). Depende de las
etapas 5 y 6 del pipeline (mapa RF y perfiles AHP); si falta alguno, se omite con aviso.

Uso:
    python scripts/run_consenso.py --config config.yaml
"""

import os
import sys
import argparse
import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.consenso_perfiles import analizar_consenso


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Consenso/divergencia entre perfiles (Etapa 10)')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base = _ruta_abs('data/results/mapa_probabilidad_aptitud.tif')
    conservador = _ruta_abs('data/results/aptitud_conservador.tif')
    agresivo = _ruta_abs('data/results/aptitud_agresivo.tif')
    percentil = config.get('consenso_perfiles', {}).get('percentil_apto', 90)

    for p, etiqueta in [(base, 'mapa RF (etapa 5)'), (conservador, 'perfil conservador (etapa 6)'),
                        (agresivo, 'perfil agresivo (etapa 6)')]:
        if not os.path.exists(p):
            print(f"  [OMITIDA] Falta {os.path.basename(p)} ({etiqueta}). Corre el pipeline primero.")
            return 0

    out_raster = _ruta_abs('data/results/consenso_perfiles.tif')
    out_json = _ruta_abs('data/results/consenso_perfiles.json')

    print(f"Analizando consenso entre perfiles (top {100 - percentil}% por perfil)...")
    stats = analizar_consenso(base, conservador, agresivo, percentil, out_raster, out_json)

    print(f"  Celdas válidas: {stats['celdas_validas']:,}")
    print(f"  Consenso (3 perfiles): {stats['consenso_3_perfiles']['celdas']:,} "
          f"({stats['consenso_3_perfiles']['pct']}%)")
    print(f"  Divergencia (2 perfiles): {stats['divergencia_2_perfiles']['pct']}% | "
          f"(1 perfil): {stats['divergencia_1_perfil']['pct']}%")
    print(f"  Salidas: {os.path.basename(out_raster)}, {os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
