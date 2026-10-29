"""CLI del análisis aparte que compara el teselado del motor con GHI observado.

FUERA del pipeline (no está en FASES). Sobre los recortes de `config.yaml > comparar_tiles`
corre el motor sin --tile, con --tile y teselado desde Python, y escribe en
`comparar_tiles.dir_resultados` un `resumen.json` y la figura `comparacion_tiles.png`
(ver src/comparar_tiles.py). Usa las mismas opciones del motor que el pipeline
(sección `solarpv`, montaje fijo).

Ejemplos:
    python scripts/run_comparar_tiles.py --config config.yaml
    # agrega solarpv-rs v0.4.0 sin corregir para mostrar su error con --tile:
    python scripts/run_comparar_tiles.py --binario-original ruta/a/solarpv_v040.exe
"""

import argparse
import json
import os
import sys

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _asegurar_proj_lib
_asegurar_proj_lib()
from src.solar_yield import _resolver_binario, preparar_ghi
from src.comparar_tiles import ejecutar_comparacion


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Compara --tile del motor con GHI observado')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--binario-original', default=None,
                        help='Binario de solarpv-rs v0.4.0 sin corregir (opcional): agrega su '
                             '--tile a la comparación para mostrar el error')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    cfg, comp = config['solarpv'], config['comparar_tiles']
    if not cfg.get('ghi'):
        print("ERROR: la comparación requiere GHI observado (solarpv.ghi en config.yaml).")
        return 1

    montaje = cfg.get('montaje', {})
    opciones = dict(lat=cfg['lat'], lon=cfg['lon'], date=cfg['date'], mount='tilt',
                    tilt=montaje.get('tilt'), surface_azimuth=montaje.get('surface_azimuth'),
                    gcr=None, per_cell_lat=True, svf=True,
                    horizon_radius=cfg.get('horizon_radius'), ghi_unit=cfg.get('ghi_unit', 'kwh'))
    binario = _resolver_binario(_ruta_abs(cfg['binario']))
    original = _resolver_binario(args.binario_original) if args.binario_original else None

    print("Comparación de --tile con GHI observado (fuera del pipeline)")
    resumen = ejecutar_comparacion(
        recortes=comp['recortes'],
        dem_path=_ruta_abs(cfg['dem']),
        ghi_path=preparar_ghi(cfg, _ruta_abs),
        dir_salida=_ruta_abs(comp['dir_resultados']),
        binario=binario, opciones=opciones,
        tamano=int(comp['tamano']), tesela=int(comp['tesela']),
        binario_original=original,
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nResultados en {_ruta_abs(comp['dir_resultados'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
