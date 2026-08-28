"""CLI: métricas Recall@K% y Precisión@K% contra los umbrales comprometidos en PEP1 §6.5.

Calcula tres cifras y las guarda en data/results/metricas_topk.json:
  A-100  in-sample sobre el mapa entregado (100 m)  -> la cifra del producto final
  A-500  in-sample por región (grilla de comparación) -> ancla de comparabilidad
  B-500  out-of-sample LORO (hold-out espacial)     -> la cifra defendible

La comparación válida es A-500 vs B-500: misma grilla y misma estructura de zonas, de modo
que la única variable que cambia es qué datos vio el modelo.

Uso:
    python scripts/run_metricas_topk.py --config config.yaml
"""

import os
import sys
import json
import argparse
import warnings

import yaml
import joblib
import geopandas as gpd

warnings.filterwarnings("ignore")

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.metricas_topk import (
    KS_DEFECTO,
    evaluar_mapa_entregado,
    evaluar_por_region,
    contrastar_con_umbrales_pep1,
)
from src.spatial_validation import asignar_region_a_muestras
from src.utils import _esta_actualizado


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _cargar_muestras(config):
    """Carga el dataset de entrenamiento con REGION asignada y nombres de columna restaurados."""
    paths_results = config['paths']['results']
    muestras = gpd.read_file(_ruta_abs(paths_results['dataset_ml']))

    # ESRI trunca los nombres de columna a 10 caracteres en shapefile (invariante del repo).
    rename_map = {
        'dist_trans': 'dist_transmision',
        'dist_almac': 'dist_almacen',
        'dist_subs':  'dist_subestaciones',
    }
    muestras = muestras.rename(columns={k: v for k, v in rename_map.items()
                                        if k in muestras.columns})

    regiones = gpd.read_file(
        _ruta_abs(config['paths']['raw']['vectores']['regiones'])).to_crs(muestras.crs)
    regiones = regiones[regiones['REGION'].isin(['Antofagasta', 'Atacama'])]
    return asignar_region_a_muestras(muestras, regiones)


def _imprimir_bloque(titulo, resultado):
    print(f"\n  {titulo}  (resolución {resultado['resolucion_m']} m, "
          f"{resultado['n_plantas_evaluadas']} plantas)")
    for k, v in resultado['por_k'].items():
        z = v['precision_zonas']
        linea = (f"    K={k:>2}%  recall {v['recall']:.3f} "
                 f"({v['plantas_capturadas']}/{resultado['n_plantas_evaluadas']})  |  "
                 f"área real {v['pct_area_seleccionada']:.2f}%")
        if z:
            linea += (f"  |  precisión-zona {z['precision']*100:.1f}% "
                      f"({z['n_zonas_con_planta']}/{z['n_zonas_candidatas']} zonas)  |  "
                      f"enriq. {z['factor_enriquecimiento']}x")
        print(linea)


def _imprimir_combinado(titulo, resultado):
    print(f"\n  {titulo}  (resolución {resultado['resolucion_m']} m, "
          f"{resultado['n_plantas_evaluadas']} plantas)")
    for k, v in resultado['combinado_por_k'].items():
        print(f"    K={k:>2}%  recall {v['recall']:.3f} "
              f"({v['plantas_capturadas']}/{resultado['n_plantas_evaluadas']})  |  "
              f"precisión-zona {v['precision_zonas']*100:.1f}% "
              f"({v['n_zonas_con_planta']}/{v['n_zonas_candidatas']} zonas)  |  "
              f"enriq. {v['factor_enriquecimiento']}x")
        a = v.get('precision_area')
        if a:
            print(f"          precisión-área {a['precision']*100:.3f}%  "
                  f"(techo con modelo perfecto {a['precision_techo_modelo_perfecto']*100:.2f}%, "
                  f"alcanzado {a['pct_del_techo_alcanzado']}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Recall@K / Precisión@K contra los umbrales de PEP1')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza recalcular aunque el resultado ya esté actualizado')
    parser.add_argument('--ks', default=None,
                        help='Lista de K (%% del área) separados por coma, p. ej. "1,3". '
                             'Por defecto barre 0.5,1,2,3,5,7.5,10 para poder graficar la '
                             'curva; 1 y 3 son obligatorios para el contraste con PEP1.')
    parser.add_argument('--ha-por-mw', type=float, default=2.0,
                        help='Ocupación de suelo supuesta para reconstruir la huella de las '
                             'plantas por buffer, en ha/MWac. Por defecto 2.0; el rango '
                             'publicado va de 1.45 (LBNL 2022, fijo) a 3.60 (NREL 2013, '
                             'área total). Usa 0 para omitir la precisión de área.')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    paths_results = config['paths']['results']
    model_path = _ruta_abs(paths_results.get('model_rf', 'data/results/model_rf.pkl'))
    mapa_path = _ruta_abs('data/results/mapa_probabilidad_aptitud.tif')
    salida = _ruta_abs('data/results/metricas_topk.json')

    for ruta, que in ((model_path, 'el modelo'), (mapa_path, 'el mapa de aptitud')):
        if not os.path.exists(ruta):
            print(f"  [ERROR] No existe {que} ({ruta}). Corre el pipeline primero.")
            return 1

    if not args.regenerar and _esta_actualizado([ruta_config, model_path, mapa_path], [salida]):
        print(f"  [OK] Ya existe y está actualizado: {salida} (usa --regenerar para forzar)")
        return 0

    # Misma resolución que la etapa SHAP espacial: es la grilla de análisis del proyecto.
    resolucion_m = config.get('shap_espacial', {}).get('resolucion_m', 500)

    if args.ks:
        ks = tuple(sorted(float(x) for x in args.ks.split(',') if x.strip()))
        # 1 y 3 sostienen el contraste con PEP1: sin ellos el reporte queda incompleto.
        faltantes = [k for k in (1.0, 3.0) if k not in ks]
        if faltantes:
            print(f"  [AVISO] --ks no incluye {faltantes}; el contraste con los umbrales "
                  f"de PEP1 quedará vacío para esos K.")
    else:
        ks = KS_DEFECTO

    modelo = joblib.load(model_path)

    ha_por_mw = args.ha_por_mw if args.ha_por_mw > 0 else None

    print("Calculando métricas top-K contra las plantas reales...")
    if ha_por_mw:
        print(f"  Huella de plantas reconstruida por buffer a {ha_por_mw} ha/MW "
              f"(Paneles.gdb es una capa de puntos, sin polígonos).")
    print("\n[1/3] (A-100) In-sample sobre el mapa entregado...")
    a100 = evaluar_mapa_entregado(config, directorio_raiz, ks, ha_por_mw=ha_por_mw)
    _imprimir_bloque("A-100 in-sample (mapa entregado)", a100)

    metrics_path = os.path.splitext(model_path)[0] + '_metrics.json'
    with open(metrics_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)['best_params']
    params_rf = {
        'n_estimators':     best_params['n_estimators'],
        'max_depth':        best_params['max_depth'],
        'min_samples_leaf': best_params['min_samples_leaf'],
        'class_weight':     'balanced',
    }
    muestras = _cargar_muestras(config)
    random_state = config.get('ml_params', {}).get('random_state', 42)

    # A-500 se evalúa POR REGIÓN igual que LORO: si el ancla tomara el top-K% sobre las dos
    # regiones juntas, la comparación mezclaría generalización con geometría de las zonas.
    print(f"\n[2/3] (A-500) In-sample por región ({resolucion_m} m)...")
    a500 = evaluar_por_region(config, directorio_raiz, muestras, params_rf, resolucion_m,
                              random_state=random_state, ks=ks, modelo_fijo=modelo,
                              ha_por_mw=ha_por_mw)
    _imprimir_combinado("A-500 in-sample por región", a500)

    print(f"\n[3/3] (B-500) Out-of-sample LORO ({resolucion_m} m)...")
    b500 = evaluar_por_region(config, directorio_raiz, muestras, params_rf, resolucion_m,
                              random_state=random_state, ks=ks, ha_por_mw=ha_por_mw)
    _imprimir_combinado("B-500 out-of-sample LORO", b500)

    contraste = contrastar_con_umbrales_pep1(b500['combinado_por_k'])

    resultado = {
        'ks_evaluados_pct': list(ks),
        'interpretacion_k': 'K = porcentaje del área de estudio (K1 = top 1%, K3 = top 3%)',
        'supuesto_huella_ha_por_mw': ha_por_mw,
        'in_sample_mapa_entregado': a100,
        'in_sample_por_region': a500,
        'out_of_sample_loro': b500,
        'contraste_umbrales_pep1': contraste,
        'nota_metodologica': (
            "Las variantes in-sample (A) son optimistas: las plantas entrenaron el modelo "
            "que generó el mapa (PEP1 §6.3 reentrena al 100% para la etapa cartográfica). "
            "La cifra defendible es la LORO (B), donde las plantas de cada región nunca "
            "vieron el modelo que las evalúa. La comparación válida es A-500 vs B-500: "
            "misma grilla, única variable = qué vio el modelo."
        ),
        'nota_precision': (
            "La precisión se mide a nivel de ZONA CANDIDATA (fracción de zonas conexas del "
            "top-K% que contienen al menos una planta real) y como SOLAPAMIENTO DE ÁREA "
            "contra una huella reconstruida por potencia instalada. Paneles.gdb es una capa "
            "de puntos (716 registros, área 0), no de polígonos: PEP1 §6.5 justificó el "
            "umbral 0,15 suponiendo huellas de ~20 km² que no existen en los datos."
        ),
        'sensibilidad_supuesto_huella': {
            'unidad': 'ha/MWac',
            'potencia_total_mw': 7195.7,
            'area_top_1pct_km2': 2028.0,
            'techos_por_supuesto': {
                '1.45 (LBNL 2022, fijo)':            {'huella_km2': 104.0, 'techo_k1_pct': 5.13},
                '2.00 (usado por defecto)':          {'huella_km2': 143.9, 'techo_k1_pct': 7.10},
                '2.25 (LBNL 2022, seguidor)':        {'huella_km2': 161.8, 'techo_k1_pct': 7.98},
                '2.95 (NREL 2013, área directa)':    {'huella_km2': 212.6, 'techo_k1_pct': 10.48},
                '3.60 (NREL 2013, área total)':      {'huella_km2': 259.2, 'techo_k1_pct': 12.78},
            },
            'ha_por_mw_necesario_para_umbral_pep1': 4.23,
            'conclusion': (
                "Ninguna cifra publicada de ocupación de suelo alcanza el umbral de 0,15 de "
                "PEP1: el techo va de 5,13% a 12,78%. Se necesitarían 4,23 ha/MW, por encima "
                "de todo benchmark existente, y además con captura perfecta. La conclusión "
                "es invariante al supuesto elegido, no un artefacto de usar 2,0."
            ),
        },
    }

    with open(salida, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print("\n  Contraste con los umbrales declarados en PEP1 §6.5 (sobre LORO):")
    for nombre in ('recall_k3', 'precision_k1'):
        c = contraste[nombre]
        estado = "CUMPLE" if c['cumple'] else "NO CUMPLE"
        print(f"    {nombre:<13} {c['valor']:.4f}  (umbral {c['umbral_pep1']})  -> {estado}")
        if not c['umbral_alcanzable']:
            print(f"      [!] Umbral INALCANZABLE por construcción: el techo con un modelo "
                  f"perfecto es {c['techo_modelo_perfecto']*100:.2f}%. "
                  f"Se alcanzó el {c['pct_del_techo_alcanzado']}% de ese máximo.")

    print(f"\n  Guardado en: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
