"""CLI: comparación de montaje fijo vs. seguidor de un eje (T3).

Asegura que existan ambos mapas de rendimiento (los genera con el wrapper de T1 si
faltan) y luego computa la ganancia del seguidor dentro de las zonas aptas, guardando
`data/results/comparacion_montaje.json`.

Uso:
    python scripts/run_comparacion_montaje.py --config config.yaml
    python scripts/run_comparacion_montaje.py --regenerar   # fuerza recalcular ambos rasters
"""

import os
import sys
import argparse
import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.solar_yield import generar_mapa_rendimiento, MotorNoDisponibleError
from src.comparacion_montaje import comparar


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _asegurar_rendimiento(cfg, target_srid, bounds_utm, mount, sufijo, regenerar):
    """Devuelve la ruta del *_specific_yield.tif del montaje, generándolo si falta."""
    out_prefix = _ruta_abs(cfg['out_prefix']) + sufijo
    salida = f"{out_prefix}_specific_yield.tif"
    if os.path.exists(salida) and not regenerar:
        print(f"  [OK] Ya existe el rendimiento '{mount}': {os.path.basename(salida)}")
        return salida

    montaje = cfg.get('montaje', {})
    print(f"  Generando rendimiento '{mount}' (puede tardar varios minutos)...")
    return generar_mapa_rendimiento(
        dem_path=_ruta_abs(cfg['dem']),
        out_prefix=out_prefix,
        lat=cfg['lat'], lon=cfg['lon'], date=cfg['date'],
        binario=_ruta_abs(cfg['binario']),
        mount=mount,
        tilt=montaje.get('tilt'),
        surface_azimuth=montaje.get('surface_azimuth'),
        gcr=montaje.get('gcr'),
        target_srid=target_srid, bounds_utm=bounds_utm,
        resolucion_m=cfg.get('resolucion_m'),
    )


def main():
    parser = argparse.ArgumentParser(description='Comparación montaje fijo vs seguidor (T3)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza regenerar ambos rasters de rendimiento con el motor')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg = config.get('solarpv')
    if not cfg:
        print("ERROR: falta la sección 'solarpv' en config.yaml.")
        return 1

    pv = config.get('postgis_validation', {})
    target_srid = pv.get('target_srid', 32719)
    bounds_utm = pv.get('bounds_utm')
    umbral = pv.get('prob_min', 0.70)

    aptitud_path = _ruta_abs('data/results/mapa_probabilidad_aptitud.tif')
    if not os.path.exists(aptitud_path):
        print(f"  [OMITIDA] No existe el mapa de aptitud ({aptitud_path}). Corre el pipeline primero.")
        return 0

    try:
        fijo = _asegurar_rendimiento(cfg, target_srid, bounds_utm, 'tilt', '_fijo', args.regenerar)
        seguidor = _asegurar_rendimiento(cfg, target_srid, bounds_utm, 'tracker', '_seguidor', args.regenerar)
    except MotorNoDisponibleError as e:
        print(f"\n[MOTOR NO DISPONIBLE]\n{e}")
        return 2

    out_json = _ruta_abs('data/results/comparacion_montaje.json')
    stats = comparar(fijo, seguidor, aptitud_path, umbral, out_json)

    f_, s_ = stats['rendimiento_fijo_aptas'], stats['rendimiento_seguidor_aptas']
    print(f"\n  Zonas aptas: {stats['celdas_aptas']:,} celdas")
    print(f"  Fijo     media={f_['media']}  p90={f_['p90']} kWh/kWp/año")
    print(f"  Seguidor media={s_['media']}  p90={s_['p90']} kWh/kWp/año")
    print(f"  Ganancia seguidor: media={stats['ganancia_seguidor_media_pct']:+}% | "
          f"mediana por celda={stats['ganancia_seguidor_mediana_celda_pct']:+}% "
          f"(referencia autor +{stats['referencia_autor_pct']}%)")
    print(f"  JSON: {os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
