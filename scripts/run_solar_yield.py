"""CLI para generar el mapa de rendimiento fotovoltaico con el motor solarpv-rs (T1).

Lee la configuración de `config.yaml` (sección `solarpv`) e invoca el wrapper de
`src/solar_yield.py`. La validación de CRS/extensión reutiliza `target_srid` y
`bounds_utm` de la sección `postgis_validation` para no duplicar la zona de estudio.

Ejemplos:
    # Montaje fijo (por defecto), ejecución real:
    python scripts/run_solar_yield.py --config config.yaml

    # Seguidor de un eje:
    python scripts/run_solar_yield.py --mount tracker

    # Sin binario compilado todavía: solo imprime el comando que se correría:
    python scripts/run_solar_yield.py --dry-run
"""

import os
import sys
import argparse
import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.solar_yield import generar_mapa_rendimiento, MotorNoDisponibleError
from src.utils import _esta_actualizado


def _ruta_abs(ruta: str) -> str:
    """Resuelve una ruta del config contra la raíz del proyecto (patrón del repo)."""
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Mapa de rendimiento PV con solarpv-rs')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--mount', choices=['tilt', 'tracker'], default='tilt',
                        help="Montaje: 'tilt' (fijo) o 'tracker' (seguidor de un eje)")
    parser.add_argument('--dry-run', action='store_true',
                        help='Imprime el comando sin ejecutar el motor (útil si aún no está compilado)')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza recalcular aunque el resultado ya esté actualizado')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg = config.get('solarpv')
    if not cfg:
        print("ERROR: falta la sección 'solarpv' en config.yaml.")
        return 1

    # Zona de estudio y CRS objetivo: única fuente de verdad en postgis_validation.
    pv = config.get('postgis_validation', {})
    target_srid = pv.get('target_srid', 32719)
    bounds_utm = pv.get('bounds_utm')

    montaje = cfg.get('montaje', {})
    # El sufijo distingue las salidas de montaje fijo vs seguidor (evita pisarse).
    out_prefix = _ruta_abs(cfg['out_prefix']) + ('_fijo' if args.mount == 'tilt' else '_seguidor')

    # El motor puede tardar minutos u horas (ver README): se salta si el rendimiento ya
    # existe y está más nuevo que el DEM y la config, igual que hace run_comparacion_montaje.py
    # (_asegurar_rendimiento) y run_pipeline.py para esta misma etapa.
    dem_path = _ruta_abs(cfg['dem'])
    salida_esperada = f"{out_prefix}_specific_yield.tif"
    if not args.regenerar and not args.dry_run and _esta_actualizado([ruta_config, dem_path], [salida_esperada]):
        print(f"  [OK] Ya existe y está actualizado: {salida_esperada} (usa --regenerar para forzar)")
        return 0

    try:
        salida = generar_mapa_rendimiento(
            dem_path=_ruta_abs(cfg['dem']),
            out_prefix=out_prefix,
            lat=cfg['lat'],
            lon=cfg['lon'],
            date=cfg['date'],
            binario=_ruta_abs(cfg['binario']),
            mount=args.mount,
            tilt=montaje.get('tilt'),
            surface_azimuth=montaje.get('surface_azimuth'),
            gcr=montaje.get('gcr'),
            target_srid=target_srid,
            bounds_utm=bounds_utm,
            resolucion_m=cfg.get('resolucion_m'),
            dry_run=args.dry_run,
        )
    except MotorNoDisponibleError as e:
        # Bloqueo esperado si el motor no está compilado: mensaje accionable, no traceback.
        print(f"\n[MOTOR NO DISPONIBLE]\n{e}")
        return 2

    print(f"\nRendimiento ({args.mount}) listo en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
