"""Cadena completa de evaluación de una zona por transferencia (sin reentrenar el modelo).

Encadena las ocho etapas que producen, para una región que NO participó del entrenamiento,
los mismos productos que el pipeline genera para la zona de estudio: mapa de aptitud, perfiles
AHP, consenso, SHAP espacial, rendimiento físico, cruce, métricas y figuras.

Todo se escribe bajo el `dir_resultados` de la zona (data/results/<zona>/) y su `dir_figuras`.
Ningún artefacto de la zona de entrenamiento se toca: los scripts del pipeline escriben en
rutas fijas de data/results/, y correrlos sobre otra región sin redirigir la salida
sobrescribiría los mapas que cita el informe.

Uso:
    python scripts/run_zona.py --config config.yaml --zona transferibilidad
    python scripts/run_zona.py --zona transferibilidad --desde 2   # reanudar sin rehacer el DEM
"""

import os
import sys
import argparse

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _asegurar_proj_lib
_asegurar_proj_lib()

from src.utils import _correr_etapa
from src.preprocessing import procesar_dem


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def preparar_dem(config, bloque):
    """Etapa 0: DEM/slope/aspect de la zona a la MISMA resolución que la de entrenamiento.

    `procesar_dem` omite el trabajo si las tres salidas ya existen. La resolución sale de
    `preprocesamiento.dem_resolucion_m` a propósito y no de un valor propio de la zona: la
    pendiente depende de la resolución del DEM del que se deriva, así que si las zonas
    usaran resoluciones distintas la comparación mediría procesamiento en vez de geografía.
    """
    procesar_dem(
        carpetas_dem=[_ruta_abs(c) for c in bloque['dem_folders']],
        out_slope_path=_ruta_abs(bloque['slope']),
        out_aspect_path=_ruta_abs(bloque['aspect']),
        out_dem_path=_ruta_abs(bloque['dem']),
        resolucion_m=config.get('preprocesamiento', {}).get('dem_resolucion_m'),
        patron=bloque.get('dem_patron', '*.hgt'),
        nodata_entrada=bloque.get('dem_nodata'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--zona', default='transferibilidad',
                        help='Clave del bloque de config.yaml con la zona a evaluar.')
    parser.add_argument('--desde', type=int, default=0, metavar='N',
                        help='Reanuda desde la etapa N (0 = DEM). Útil tras un fallo puntual.')
    parser.add_argument('--sin-motor', action='store_true',
                        help='Omite el rendimiento físico y el cruce (etapas 5 y 6).')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    bloque = config.get(args.zona) or {}
    if not bloque:
        print(f"ERROR: config.yaml no tiene el bloque '{args.zona}'.")
        return 1

    regiones = ', '.join(bloque.get('regiones', []))
    print("=" * 70)
    print(f"CADENA DE ZONA: {regiones}  (bloque '{args.zona}')")
    print("El modelo NO se reentrena: se aplica model_rf.pkl tal cual.")
    print("=" * 70)

    zargs = ['--config', args.config, '--zona', args.zona]

    # (numero, etiqueta, script, args, best_effort). La etapa 0 no es un script: llama
    # directamente a procesar_dem, igual que hace run_pipeline.py con su etapa 1.
    etapas = [
        (1, "Mapa de aptitud + métricas + covariate shift", "run_transferibilidad.py", zargs, False),
        (2, "Perfiles de inversión AHP/WLC", "profiles.py", zargs, False),
        (3, "Consenso/divergencia entre perfiles", "run_consenso.py", zargs, False),
        (4, "Explicabilidad SHAP espacial", "run_shap_spatial.py", zargs + ['--regenerar'], False),
        (5, "Rendimiento PV fijo (motor Rust)", "run_solar_yield.py", zargs, True),
        (6, "Cruce aptitud x rendimiento", "run_cruce.py", zargs, True),
        (7, "Figuras cartográficas de la zona", "figuras_zona.py", zargs, False),
    ]

    if args.desde <= 0:
        print("\n--- Etapa 0: DEM/slope/aspect de la zona ---")
        preparar_dem(config, bloque)

    motor_ok = True
    for numero, etiqueta, script, extra, best_effort in etapas:
        if numero < args.desde:
            continue
        if args.sin_motor and numero in (5, 6):
            print(f"\n--- Etapa {numero}: {etiqueta} --- (omitida por --sin-motor)")
            continue
        # El cruce no tiene sentido sin el mapa de rendimiento: si el motor no corrió,
        # saltarlo evita un error confuso sobre un archivo que nunca se generó.
        if numero == 6 and not motor_ok:
            print(f"\n--- Etapa {numero}: {etiqueta} --- (omitida: falta el rendimiento)")
            continue
        ok = _correr_etapa(f"Etapa {numero}: {etiqueta}", script, extra,
                           motor_best_effort=best_effort)
        if numero == 5:
            motor_ok = ok

    dir_res = bloque.get('dir_resultados', f'data/results/{args.zona}')
    dir_fig = bloque.get('dir_figuras', f'figures/{args.zona}')
    print("\n" + "=" * 70)
    print(f"Cadena de {regiones} completada.")
    print(f"  Resultados: {dir_res}")
    print(f"  Figuras:    {dir_fig}")
    if not motor_ok:
        print("  [AVISO] El motor Rust no estaba disponible: sin rendimiento ni cruce.")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
