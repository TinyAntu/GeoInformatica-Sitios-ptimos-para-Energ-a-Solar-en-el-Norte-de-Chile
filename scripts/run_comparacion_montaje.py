"""CLI: comparación de montaje fijo vs. seguidor de un eje (T3).

Asegura que existan ambos mapas de rendimiento (los genera con el wrapper de T1 si
faltan) y luego computa la ganancia del seguidor dentro de las zonas aptas, guardando
`data/results/rendimiento/comparacion_montaje.json`.

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

from src.utils import _asegurar_proj_lib, _esta_actualizado
_asegurar_proj_lib()

from src.solar_yield import (generar_mapa_rendimiento, MotorNoDisponibleError, _reestampar_crs,
                             MotorSinCorreccionTileError, preparar_ghi)
from src.comparacion_montaje import comparar
from src.rutas import ruta_resultado, prefijo_rendimiento


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _asegurar_rendimiento(cfg, prefijo, target_srid, bounds_utm, mount, sufijo, regenerar,
                          tilt=None, entradas=(), ghi=None):
    """Devuelve la ruta del *_specific_yield.tif del montaje, generándolo si falta o si es
    más viejo que `entradas` (config, DEM, GHI): un raster de cielo despejado no debe
    compararse contra otro calculado con GHI observado. `ghi` es el GHI ya preparado por
    `main` (el mismo para los tres montajes)."""
    out_prefix = prefijo + sufijo
    salida = f"{out_prefix}_specific_yield.tif"
    if not regenerar and _esta_actualizado(list(entradas), [salida]):
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
        tile=cfg.get('tile'),
        horizon_radius=cfg.get('horizon_radius'),
        ghi=ghi,
        ghi_monthly=_ruta_abs(cfg['ghi_monthly']) if cfg.get('ghi_monthly') else None,
        ghi_unit=cfg.get('ghi_unit', 'kwh'),
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

    aptitud_path = ruta_resultado(config, 'aptitud', 'mapa_probabilidad_aptitud.tif')
    if not os.path.exists(aptitud_path):
        print(f"  [OMITIDA] No existe el mapa de aptitud ({aptitud_path}). Corre el pipeline primero.")
        return 0

    # Las tres variantes deben salir de la misma config, DEM y GHI para ser comparables.
    prefijo = prefijo_rendimiento(config)
    try:
        ghi = preparar_ghi(cfg, _ruta_abs)
        entradas = [_ruta_abs(args.config), _ruta_abs(cfg['dem'])] + ([ghi] if ghi else [])
        fijo = _asegurar_rendimiento(cfg, prefijo, target_srid, bounds_utm, 'tilt', '_fijo',
                                     args.regenerar, entradas=entradas, ghi=ghi)
        seguidor = _asegurar_rendimiento(cfg, prefijo, target_srid, bounds_utm, 'tracker',
                                         '_seguidor', args.regenerar, entradas=entradas, ghi=ghi)

        ruta_tilt0_esperada = prefijo + '_fijo_tilt0_specific_yield.tif'
        fijo_tilt0 = None
        if args.incluir_tilt0 or os.path.exists(ruta_tilt0_esperada):
            fijo_tilt0 = _asegurar_rendimiento(cfg, prefijo, target_srid, bounds_utm, 'tilt',
                                               '_fijo_tilt0', args.regenerar, tilt=0,
                                               entradas=entradas, ghi=ghi)
    except MotorNoDisponibleError as e:
        print(f"\n[MOTOR NO DISPONIBLE]\n{e}")
        return 2
    except MotorSinCorreccionTileError as e:
        print(f"\n[MOTOR SIN CORREGIR]\n{e}")
        return 3
    except FileNotFoundError as e:
        print(f"\n[FALTA UN ARCHIVO DE ENTRADA]\n{e}")
        return 4

    tamano_bloque_km = config.get('validacion', {}).get('tamano_bloque_km', 15)
    out_json = ruta_resultado(config, 'rendimiento', 'comparacion_montaje.json')
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
          f"mediana por bloque={stats['ganancia_seguidor_mediana_bloque_pct']}%")
    # La referencia del autor es contra el fijo horizontal: se muestra junto a esa ganancia.
    if 'ganancia_seguidor_vs_tilt0_media_pct' in stats:
        print(f"  Ganancia seguidor (vs tilt=0°): media={stats['ganancia_seguidor_vs_tilt0_media_pct']:+}% "
              f"(referencia autor +{stats['referencia_autor_pct']}%)")
    print(f"  JSON: {os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
