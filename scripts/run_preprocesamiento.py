"""Fase 1 — Preprocesamiento: de datos crudos a rásters procesados.

Cubre las etapas 1 y 2 del pipeline:
    1. Fusiona las teselas DEM, reproyecta a EPSG:32719 y deriva slope/aspect con surtgis.
    2. Reproyecta el GHI a EPSG:32719.

Deliberadamente NO carga las capas vectoriales: ninguna de las dos etapas las necesita, y
cargarlas aquí las dejaría residentes sin motivo (ese era el costo que pagaba el orquestador
antes de separar esta fase, ver scripts/run_pipeline.py).

La idempotencia vive dentro de `procesar_dem` y `reproject_raster_to_utm`: ambas se omiten
solas si sus salidas ya existen, así que este script no replica el chequeo.

Uso:
    python scripts/run_preprocesamiento.py --config config.yaml
"""

import os
import sys
import argparse

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.preprocessing import procesar_dem, reproject_raster_to_utm
from src.utils import _resolver_rutas, _ruta_abs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])

    processed = config['paths']['processed']
    rasters_raw = config['paths']['raw']['rasters']

    # --- Etapa 1: DEM -> slope/aspect ---
    print("\n--- Etapa 1: DEM (fusión, reproyección y derivados) ---")
    procesar_dem(
        carpetas_dem=rasters_raw['dem_folders'],
        out_slope_path=processed['slope'],
        out_aspect_path=processed['aspect'],
        out_dem_path=processed['dem_32719'],
        resolucion_m=config.get('preprocesamiento', {}).get('dem_resolucion_m'),
    )

    # --- Etapa 2: Reproyección GHI ---
    print("\n--- Etapa 2: Reproyección GHI a EPSG:32719 ---")
    reproject_raster_to_utm(rasters_raw['ghi'], processed['ghi_32719'], epsg_code=32719)

    print("\nPreprocesamiento completo.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
