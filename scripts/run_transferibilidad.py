"""CLI: generalización del modelo sobre una región que no participó del entrenamiento.

Aplica `model_rf.pkl` TAL CUAL (sin reentrenar) a la grilla de otra zona, guarda el mapa de
aptitud resultante y calcula Recall@K / Precisión@K contra las plantas reales de esa región,
más el diagnóstico de desplazamiento de covariables.

Corre además una segunda pasada de SENSIBILIDAD con el filtro regional en modo 'compuesto':
la capa de transmisión codifica las líneas que cruzan fronteras como 'ATACAMA;COQUIMBO', y el
filtro exacto —el que usó el entrenamiento— las descarta. El resultado principal usa el modo
exacto por consistencia con el modelo; la sensibilidad mide cuánto de la caída se debe a ese
defecto de datos y cuánto a la geografía.

Uso:
    python scripts/run_transferibilidad.py --config config.yaml --zona transferibilidad
"""

import os
import sys
import json
import argparse
import warnings

import yaml
import joblib

warnings.filterwarnings("ignore")

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.metricas_topk import KS_DEFECTO
from src.explainability_spatial import zona_de_config
from src.transferibilidad import (
    evaluar_zona, escribir_mapa_aptitud,
    cargar_muestras_entrenamiento, diagnostico_covariate_shift,
)


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--zona', default='transferibilidad',
                        help='Clave del bloque de config.yaml con la zona a evaluar.')
    parser.add_argument('--ks', default=None,
                        help='Lista de K (%% del área) separados por coma. Por defecto '
                             'barre 0.5,1,2,3,5,7.5,10.')
    parser.add_argument('--sin-sensibilidad', action='store_true',
                        help='Omite la segunda pasada con el filtro regional compuesto.')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    bloque = config.get(args.zona) or {}
    if not bloque:
        raise SystemExit(f"config.yaml no tiene el bloque '{args.zona}'.")

    ks = tuple(float(k) for k in args.ks.split(',')) if args.ks else KS_DEFECTO
    resolucion_m = bloque.get('resolucion_m', 500)
    ha_por_mw = bloque.get('ha_por_mw', 2.0)
    dir_res = _ruta_abs(bloque.get('dir_resultados', f'data/results/{args.zona}'))
    os.makedirs(dir_res, exist_ok=True)

    modelo = joblib.load(_ruta_abs(config['paths']['results']['model_rf']))
    zona = zona_de_config(config, args.zona)

    print(f"\n=== Generalización sobre {zona['regiones']} ===")
    print(f"  Modelo: {config['paths']['results']['model_rf']} (sin reentrenar)")
    print(f"  Grilla: {resolucion_m} m | huella: {ha_por_mw} ha/MW")

    # --- Pasada principal: filtro regional EXACTO, igual que el entrenamiento ---
    res, X, extras = evaluar_zona(config, directorio_raiz, modelo, zona,
                                  resolucion_m, ks=ks, ha_por_mw=ha_por_mw)

    ruta_mapa = os.path.join(dir_res, 'mapa_probabilidad_aptitud.tif')
    escribir_mapa_aptitud(extras, ruta_mapa)
    print(f"  Mapa de aptitud -> {os.path.relpath(ruta_mapa, directorio_raiz)}")

    muestras = cargar_muestras_entrenamiento(config, directorio_raiz)
    shift = diagnostico_covariate_shift(muestras, X)

    salida = {
        'zona': args.zona,
        'regiones': zona['regiones'],
        'modelo': 'model_rf.pkl (entrenado en Antofagasta+Atacama, NO reentrenado)',
        'resolucion_m': resolucion_m,
        'supuesto_huella_ha_por_mw': ha_por_mw,
        'metricas': res,
        'covariate_shift': shift,
        'mapa_aptitud': os.path.relpath(ruta_mapa, directorio_raiz),
        'nota_lectura': (
            "El techo de precisión depende de la razón huella/área propuesta, que es propia "
            "de cada zona: comparar la precisión cruda entre regiones no informa. Lo "
            "comparable es 'pct_del_techo_alcanzado' y el factor de enriquecimiento."
        ),
        'nota_extrapolacion': (
            "Un Random Forest no extrapola: fuera del rango de entrenamiento devuelve el "
            "valor de la hoja más cercana. Leer 'covariate_shift.pct_fuera_de_rango' antes "
            "de atribuir cualquier caída a la geografía."
        ),
    }

    # --- Sensibilidad: filtro regional COMPUESTO (rescata las líneas multi-región) ---
    if not args.sin_sensibilidad:
        print("\n  [sensibilidad] Repitiendo con filtro regional compuesto...")
        zona_comp = dict(zona, modo_region='compuesto')
        res_comp, _, _ = evaluar_zona(config, directorio_raiz, modelo, zona_comp,
                                      resolucion_m, ks=ks, ha_por_mw=ha_por_mw)
        salida['sensibilidad_filtro_lineas'] = {
            'descripcion': (
                "El filtro exacto descarta las líneas cuyo campo REGION es una cadena "
                "compuesta ('ATACAMA;COQUIMBO'). El modo compuesto las rescata. La "
                "diferencia acota cuánto del resultado principal es artefacto del filtro."
            ),
            'metricas': res_comp,
        }

    ruta_json = os.path.join(dir_res, 'metricas_transferibilidad.json')
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # --- Resumen legible ---
    print(f"\n  Plantas evaluadas : {res['n_plantas_evaluadas']}"
          f" ({res['potencia_total_mw']} MW)")
    print(f"  Área de la zona   : {res['area_zona_km2']:,.0f} km²")
    print(f"\n  {'K%':>5} {'recall':>8} {'precisión':>10} {'techo':>8} {'% techo':>9}")
    for k, v in res['por_k'].items():
        pa = v.get('precision_area') or {}
        print(f"  {k:>5} {v['recall']:>8.4f} {pa.get('precision', float('nan')):>10.4f}"
              f" {pa.get('precision_techo_modelo_perfecto', float('nan')):>8.4f}"
              f" {pa.get('pct_del_techo_alcanzado', float('nan')):>8.1f}%")

    print("\n  Desplazamiento de covariables (% de píxeles fuera del rango de entrenamiento):")
    for f_, d in sorted(shift.items(), key=lambda kv: -kv[1]['pct_fuera_de_rango']):
        print(f"    {f_:<20} {d['pct_fuera_de_rango']:>6.2f}%"
              f"   (desplazamiento {d['desplazamiento_medias_sd']} sd)")

    print(f"\n  JSON -> {os.path.relpath(ruta_json, directorio_raiz)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
