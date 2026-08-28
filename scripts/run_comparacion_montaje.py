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

from src.utils import _asegurar_proj_lib
_asegurar_proj_lib()

from src.solar_yield import generar_mapa_rendimiento, MotorNoDisponibleError, _reestampar_crs
from src.comparacion_montaje import comparar


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _asegurar_rendimiento(cfg, target_srid, bounds_utm, mount, sufijo, regenerar, tilt=None):
    """Devuelve la ruta del *_specific_yield.tif del montaje, generándolo si falta."""
    out_prefix = _ruta_abs(cfg['out_prefix']) + sufijo
    salida = f"{out_prefix}_specific_yield.tif"
    if os.path.exists(salida) and not regenerar:
        _reestampar_crs(salida, target_srid)
        print(f"  [OK] Ya existe el rendimiento '{mount}' ({sufijo}): {os.path.basename(salida)}")
        return salida

    montaje = cfg.get('montaje', {})
    tilt_efectivo = tilt if tilt is not None else montaje.get('tilt')
    print(f"  Generando rendimiento '{mount}' (sufijo '{sufijo}')...")
    return generar_mapa_rendimiento(
        dem_path=_ruta_abs(cfg['dem']),
        out_prefix=out_prefix,
        lat=cfg['lat'], lon=cfg['lon'], date=cfg['date'],
        binario=_ruta_abs(cfg['binario']),
        mount=mount,
        tilt=tilt_efectivo,
        surface_azimuth=montaje.get('surface_azimuth'),
        gcr=montaje.get('gcr'),
        target_srid=target_srid, bounds_utm=bounds_utm,
        resolucion_m=cfg.get('resolucion_m'),
    )


def main():
    parser = argparse.ArgumentParser(description='Comparación montaje fijo vs seguidor (T3)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza regenerar los rasters de rendimiento con el motor')
    parser.add_argument('--incluir-tilt0', action='store_true',
                        help='Calcula e incluye el rendimiento con tilt=0 (plano) para compararlo contra tilt=23° y seguidor')
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

        ruta_tilt0_esperada = _ruta_abs(cfg['out_prefix']) + '_fijo_tilt0_specific_yield.tif'
        fijo_tilt0 = None
        if args.incluir_tilt0 or os.path.exists(ruta_tilt0_esperada):
            fijo_tilt0 = _asegurar_rendimiento(cfg, target_srid, bounds_utm, 'tilt', '_fijo_tilt0', args.regenerar, tilt=0)
    except MotorNoDisponibleError as e:
        print(f"\n[MOTOR NO DISPONIBLE]\n{e}")
        return 2

    tamano_bloque_km = config.get('validacion', {}).get('tamano_bloque_km', 15)
    out_json = _ruta_abs('data/results/comparacion_montaje.json')
    stats = comparar(fijo, seguidor, aptitud_path, umbral, out_json,
                     tamano_bloque_km=tamano_bloque_km, fijo_tilt0_path=fijo_tilt0)

    f_, s_ = stats['rendimiento_fijo_aptas'], stats['rendimiento_seguidor_aptas']
    print(f"\n  Zonas aptas: {stats['celdas_aptas']:,} celdas "
          f"({stats['n_bloques_espaciales']} bloques espaciales de {tamano_bloque_km:.0f} km)")
    if 'rendimiento_fijo_tilt0_aptas' in stats:
        t0_ = stats['rendimiento_fijo_tilt0_aptas']
        print(f"  Fijo (tilt=0°)  media={t0_['media']}  p90={t0_['p90']} kWh/kWp/año")
    print(f"  Fijo (tilt=23°) media={f_['media']}  p90={f_['p90']} kWh/kWp/año")
    print(f"  Seguidor (1-eje) media={s_['media']}  p90={s_['p90']} kWh/kWp/año")
    if 'ganancia_tilt23_vs_tilt0_media_pct' in stats:
        print(f"  Ganancia tilt=23° sobre tilt=0°: media={stats['ganancia_tilt23_vs_tilt0_media_pct']:+}%")
    print(f"  Ganancia seguidor (vs tilt=23°): media={stats['ganancia_seguidor_media_pct']:+}% | "
          f"mediana por celda={stats['ganancia_seguidor_mediana_celda_pct']:+}% | "
          f"mediana por bloque={stats['ganancia_seguidor_mediana_bloque_pct']}% "
          f"(referencia autor +{stats['referencia_autor_pct']}%)")
    if 'ganancia_seguidor_vs_tilt0_media_pct' in stats:
        print(f"  Ganancia seguidor (vs tilt=0°): media={stats['ganancia_seguidor_vs_tilt0_media_pct']:+}%")
    print(f"  JSON: {os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
