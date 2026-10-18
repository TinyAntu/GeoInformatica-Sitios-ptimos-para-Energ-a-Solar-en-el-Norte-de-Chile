"""Estudio aparte (fuera del pipeline): auditoría de fuga temporal en las distancias a la infraestructura.

Se corre a mano, después de que el pipeline haya producido el dataset y el modelo (fase 2).

Reentrena el modelo con distintas definiciones de `dist_subestaciones`, `dist_transmision` y
`dist_almacen` (todas / sin dedicadas / solo las previas a cada planta), por separado y en
conjunto, dejando todo lo demás fijo, y compara desempeño e importancias. Ver la explicación
completa en src/auditoria_infraestructura.py.

No modifica ningún artefacto del pipeline (model_rf.pkl, dataset, mapas): todo lo que
produce (JSON, tabla comparativa CSV, figuras y capas filtradas) queda en una sola carpeta,
`auditoria_infraestructura.dir_resultados` (por defecto data/results/auditoria_infraestructura/).

Uso:
    python scripts/run_auditoria_infraestructura.py --config config.yaml
    python scripts/run_auditoria_infraestructura.py --sin-mapa      # solo la parte por puntos
    python scripts/run_auditoria_infraestructura.py --regenerar     # fuerza recalcular
"""

import os
import sys
import copy
import json
import argparse
import warnings

import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # backend no interactivo: solo guardamos figuras
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _esta_actualizado, _resolver_rutas, _ruta_abs, _shapefile_paths
from src.spatial_validation import asignar_region_a_muestras
from src.auditoria_infraestructura import (
    CAPA_POR_FEATURE,
    EXPERIMENTOS,
    variantes_del_experimento,
    tiene_mapa,
    asignar_tipo_por_subestacion,
    preparar_infraestructura,
    asignar_fechas_referencia,
    filtrar_infraestructura,
    distancia_a_infraestructura,
    diagnosticar_fuga,
    optimizar_hiperparametros,
    evaluar_variante,
    resumir_distancias,
)

KS_MAPA = (1.0, 3.0, 5.0)
# Tolerancias de los controles de validez (ver main): la línea base debe reproducir el
# pipeline, o las diferencias entre experimentos medirían también un error de montaje.
TOLERANCIA_DISTANCIA_M = 1.0
TOLERANCIA_RECALL = 1e-4
SUBCARPETA_OPTUNA = 'anexo_optuna'


def entradas_esperadas(config, ruta_config, optimizar=False) -> list:
    """Rutas cuya fecha decide si hay que recalcular (chequeo incremental del proyecto).

    Incluye el código del propio estudio, como la etapa 16 del pipeline: sin él, agregar un
    experimento o una capa no dispararía el recálculo, porque los datos no cambiaron. El
    anexo Optuna depende además del JSON principal, contra el que arma la comparativa.
    """
    results = config['paths']['results']
    vectores = config['paths']['raw']['vectores']
    codigo = [os.path.abspath(__file__),
              os.path.join(directorio_raiz, 'src', 'auditoria_infraestructura.py')]
    entradas = ([ruta_config, os.path.splitext(results['model_rf'])[0] + '_metrics.json',
                 vectores['fotovoltaicas']] + codigo
                + [vectores[capa] for capa in CAPA_POR_FEATURE.values()]
                + _shapefile_paths(results['dataset_ml']))
    if optimizar:
        entradas.append(ruta_salida(config, optimizar=False))
    return entradas


def ruta_salida(config, optimizar=False) -> str:
    """JSON de resultados. El anexo Optuna vive en su propia subcarpeta, separado del
    estudio con hiperparámetros fijos, para poder comparar ambos sin pisarse."""
    bloque = config.get('auditoria_infraestructura') or {}
    directorio = _ruta_abs(bloque.get('dir_resultados', 'data/results/auditoria_infraestructura'))
    if optimizar:
        return os.path.join(directorio, SUBCARPETA_OPTUNA, 'ablacion_infraestructura_optuna.json')
    return os.path.join(directorio, 'ablacion_infraestructura.json')


def _cargar_muestras(config):
    """Dataset de entrenamiento con nombres restaurados (truncado ESRI) y REGION asignada."""
    muestras = gpd.read_file(config['paths']['results']['dataset_ml'])
    rename_map = {'dist_trans': 'dist_transmision', 'dist_almac': 'dist_almacen',
                  'dist_subs': 'dist_subestaciones'}
    muestras = muestras.rename(columns={k: v for k, v in rename_map.items() if k in muestras.columns})
    regiones = gpd.read_file(config['paths']['raw']['vectores']['regiones']).to_crs(muestras.crs)
    regiones = regiones[regiones['REGION'].isin(config['zona_estudio']['regiones'])]
    return asignar_region_a_muestras(muestras, regiones)


def _params_rf(config) -> dict:
    ruta = os.path.splitext(config['paths']['results']['model_rf'])[0] + '_metrics.json'
    with open(ruta, 'r', encoding='utf-8') as f:
        best = json.load(f)['best_params']
    return {'n_estimators': best['n_estimators'], 'max_depth': best['max_depth'],
            'min_samples_leaf': best['min_samples_leaf'], 'class_weight': 'balanced'}


def _recall_loro_cartografico(config, muestras, params_rf, capas, experimento, dir_salida,
                              random_state):
    """Recall LORO top-K sobre el mapa, con las capas filtradas del experimento.

    Reutiliza `evaluar_por_region` (la misma función que produce metricas_topk.json) con
    una copia del config cuyas rutas de infraestructura apuntan a las capas filtradas. Así
    la línea base debe reproducir la cifra LORO ya publicada, lo que valida el montaje.
    """
    from src.metricas_topk import evaluar_por_region

    config_experimento = copy.deepcopy(config)
    for feature, variante in variantes_del_experimento(experimento).items():
        if variante == 'todas':
            continue
        nombre_capa = CAPA_POR_FEATURE[feature]
        ruta_capa = os.path.join(dir_salida, f'{nombre_capa}_{variante}.gpkg')
        if not os.path.exists(ruta_capa):
            filtrada = filtrar_infraestructura(capas[nombre_capa], variante)
            filtrada.drop(columns=['f_op']).to_file(ruta_capa, driver='GPKG')
        config_experimento['paths']['raw']['vectores'][nombre_capa] = ruta_capa

    resolucion_m = config.get('shap_espacial', {}).get('resolucion_m', 500)
    loro = evaluar_por_region(config_experimento, directorio_raiz, muestras, params_rf,
                              resolucion_m, random_state=random_state, ks=KS_MAPA)
    return {
        'resolucion_m': resolucion_m,
        'recall_por_k': {k: {'recall': v['recall'], 'plantas_capturadas': v['plantas_capturadas']}
                         for k, v in loro['combinado_por_k'].items()},
        'n_plantas_evaluadas': loro['n_plantas_evaluadas'],
    }


def _controlar_recall_publicado(config, recall_linea_base, estricto=True):
    """La línea base cartográfica debe reproducir out_of_sample_loro de metricas_topk.json.

    Solo es exigible (`estricto`) cuando la línea base usa los mismos hiperparámetros que el
    pipeline. En el anexo Optuna, si la búsqueda eligiera otros, la diferencia se registra
    como resultado en vez de abortar.
    """
    ruta = _ruta_abs('data/results/metricas_topk.json')
    if not os.path.exists(ruta):
        print("  [AVISO] No existe metricas_topk.json: se omite el control cartográfico.")
        return None
    with open(ruta, 'r', encoding='utf-8') as f:
        publicado = json.load(f)['out_of_sample_loro']['combinado_por_k']
    diferencias = {k: abs(recall_linea_base[k]['recall'] - publicado[k]['recall'])
                   for k in recall_linea_base if k in publicado}
    if any(d > TOLERANCIA_RECALL for d in diferencias.values()):
        if estricto:
            raise RuntimeError(f"La línea base no reproduce el recall LORO publicado: {diferencias}")
        print(f"  [AVISO] La línea base optimizada difiere del recall publicado: {diferencias}")
    return round(max(diferencias.values()), 6)


# Colores fijos por variable: el mismo color significa lo mismo en todas las figuras.
COLOR_FEATURE = {
    'dist_subestaciones': '#d35400', 'dist_transmision': '#e6a157', 'dist_almacen': '#8e44ad',
    'slope': '#2a7f62', 'ghi': '#f1c40f', 'elev': '#7f8c8d', 'northness': '#95a5a6',
}
NOTA_FIGURAS = {
    False: "Mismos 420 puntos y mismos hiperparámetros; solo cambia la infraestructura visible.",
    True: ("Anexo Optuna: mismos 420 puntos; hiperparámetros reoptimizados por experimento "
           "(TPE, semilla 42, 50 intentos, Brier SBCV)."),
}


def _guardar_figura(fig, ruta, resultado):
    optimizado = bool(resultado.get('hiperparametros_optimizados'))
    fig.text(0.5, -0.02, NOTA_FIGURAS[optimizado], ha='center', fontsize=8, style='italic')
    plt.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _figura_desempeno(resultado, ruta):
    """AUC SBCV y LORO del modelo, y AUC de cada distancia auditada por sí sola."""
    experimentos = list(resultado['experimentos'])
    datos = [resultado['experimentos'][e] for e in experimentos]
    x = np.arange(len(experimentos))

    series = [('AUC SBCV (30 km)', [d['auc_sbcv_media'] for d in datos], '#2a7f62'),
              ('AUC LORO', [d['auc_loro_media'] for d in datos], '#1f78b4')]
    series += [(f'AUC solo {f}', [d['auc_univariado'][f] for d in datos], COLOR_FEATURE[f])
               for f in CAPA_POR_FEATURE]
    ancho = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (etiqueta, valores, color) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * ancho, valores, ancho, label=etiqueta, color=color)
    # El piso del eje queda bajo 0,5 a propósito: una distancia puede separar PEOR que el azar
    # (las plantas más lejos de esa infraestructura que los negativos) y debe verse.
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, label='Azar (AUC = 0,5)')
    ax.set_xticks(x, experimentos, rotation=30, ha='right')
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel('AUC')
    ax.set_title('Desempeño del modelo y poder separador de cada distancia, por experimento')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    _guardar_figura(fig, ruta, resultado)


def _figura_shap(resultado, ruta):
    """Reparto de la importancia SHAP global entre las 7 variables, por experimento."""
    experimentos = list(resultado['experimentos'])
    datos = [resultado['experimentos'][e] for e in experimentos]
    x = np.arange(len(experimentos))

    # Primero las tres distancias auditadas (abajo), después el resto.
    orden = list(CAPA_POR_FEATURE) + [f for f in COLOR_FEATURE if f not in CAPA_POR_FEATURE]
    fig, ax = plt.subplots(figsize=(13, 6))
    base = np.zeros(len(experimentos))
    for feature in orden:
        valores = np.array([d['importancia_shap_pct'][feature] for d in datos])
        ax.bar(x, valores, 0.6, bottom=base, label=feature, color=COLOR_FEATURE[feature])
        base += valores
    ax.set_xticks(x, experimentos, rotation=30, ha='right')
    ax.set_ylabel('% de la importancia SHAP global')
    ax.set_title('Reparto de la importancia SHAP por experimento')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    _guardar_figura(fig, ruta, resultado)


def _figura_recall_mapa(resultado, ruta):
    """Recall LORO top-K sobre el mapa, para los experimentos que admiten mapa."""
    recall = {e: r for e, r in (resultado.get('recall_loro_cartografico') or {}).items()
              if isinstance(r, dict)}
    fig, ax = plt.subplots(figsize=(11, 6))
    if recall:
        con_mapa = list(recall)
        ks = list(next(iter(recall.values()))['recall_por_k'])
        colores = ['#1f78b4', '#6baed6', '#c6dbef']
        ancho = 0.8 / len(ks)
        xm = np.arange(len(con_mapa))
        for i, k in enumerate(ks):
            valores = [recall[e]['recall_por_k'][k]['recall'] for e in con_mapa]
            barras = ax.bar(xm + (i - (len(ks) - 1) / 2) * ancho, valores, ancho,
                            label=f'Recall K{k} (top {k} % del territorio)', color=colores[i % 3])
            ax.bar_label(barras, fmt='%.2f', fontsize=7, padding=2)
        ax.set_xticks(xm, con_mapa, rotation=30, ha='right')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'Corrida con --sin-mapa: sin recall cartográfico',
                ha='center', va='center')
        ax.set_xticks([])
    ax.set_ylabel('Fracción de las 105 plantas capturadas')
    ax.set_title('Recall LORO sobre el mapa (top-K), experimentos sin variante "previas"')
    _guardar_figura(fig, ruta, resultado)


# Figuras del estudio: nombre de archivo -> función que la dibuja.
FIGURAS = {
    'figura_desempeno_auc.png': _figura_desempeno,
    'figura_importancia_shap.png': _figura_shap,
    'figura_recall_mapa.png': _figura_recall_mapa,
}


def _guardar_tabla_comparativa(resultado, ruta_tabla):
    """Una fila por experimento con las cifras que se comparan (lista para el informe)."""
    recall_mapa = resultado.get('recall_loro_cartografico') or {}
    filas = []
    for experimento, v in resultado['experimentos'].items():
        fila = {'experimento': experimento, **{f'variante_{f}': var for f, var in
                                               variantes_del_experimento(experimento).items()}}
        fila.update({
            'auc_sbcv': v['auc_sbcv_media'],
            'auc_sbcv_std': v['auc_sbcv_std'],
            'brier_sbcv': v['brier_sbcv_media'],
            'auc_loro': v['auc_loro_media'],
            **{f'auc_solo_{f}': v['auc_univariado'][f] for f in CAPA_POR_FEATURE},
            **{f'shap_pct_{f}': p for f, p in v['importancia_shap_pct'].items()},
            'shap_pct_acceso_red': v['shap_pct_acceso_red'],
            'shap_pct_infraestructura': v['shap_pct_infraestructura'],
            'variable_mas_importante': v['variable_mas_importante_shap'],
        })
        for feature, d in v['distancias'].items():
            fila[f'mediana_{feature}_positivas_m'] = d['mediana_positivas_m']
            fila[f'mediana_{feature}_negativas_m'] = d['mediana_negativas_m']
        mapa = recall_mapa.get(experimento)
        for k, valores in (mapa or {}).get('recall_por_k', {}).items():
            fila[f'recall_loro_mapa_k{k}'] = valores['recall']
        filas.append(fila)
    pd.DataFrame(filas).to_csv(ruta_tabla, index=False, encoding='utf-8')


def _metricas_comparables(resultado, experimento) -> dict:
    """Cifras de un experimento que se comparan entre hiperparámetros fijos y optimizados."""
    v = resultado['experimentos'][experimento]
    metricas = {
        'auc_sbcv': v['auc_sbcv_media'],
        'brier_sbcv': v['brier_sbcv_media'],
        'auc_loro': v['auc_loro_media'],
        'shap_pct_acceso_red': v['shap_pct_acceso_red'],
        'shap_pct_infraestructura': v['shap_pct_infraestructura'],
        'shap_pct_slope': v['importancia_shap_pct']['slope'],
        'shap_pct_ghi': v['importancia_shap_pct']['ghi'],
    }
    mapa = (resultado.get('recall_loro_cartografico') or {}).get(experimento)
    for k, valores in (mapa or {}).get('recall_por_k', {}).items():
        metricas[f'recall_loro_mapa_k{k}'] = valores['recall']
    return metricas


def _guardar_comparativa_fijos_vs_optuna(resultado_optuna, ruta_json_fijos, ruta_csv):
    """Una fila por experimento: hiperparámetros y cada métrica con fijos, optimizados y Δ.

    Se arma contra el JSON del estudio con hiperparámetros fijos ya existente: este anexo
    no lo recalcula ni lo modifica.
    """
    if not os.path.exists(ruta_json_fijos):
        print(f"  [AVISO] No existe {ruta_json_fijos}: se omite la comparativa con los fijos.")
        return None
    with open(ruta_json_fijos, 'r', encoding='utf-8') as f:
        resultado_fijos = json.load(f)

    fijos = resultado_fijos['hiperparametros']
    filas = []
    for experimento in resultado_optuna['experimentos']:
        if experimento not in resultado_fijos['experimentos']:
            continue
        optimizados = resultado_optuna['experimentos'][experimento]['hiperparametros']
        fila = {'experimento': experimento}
        for p in ('n_estimators', 'max_depth', 'min_samples_leaf'):
            fila[f'{p}_fijo'] = fijos[p]
            fila[f'{p}_optuna'] = optimizados[p]
        con_fijos = _metricas_comparables(resultado_fijos, experimento)
        con_optuna = _metricas_comparables(resultado_optuna, experimento)
        for metrica, valor_fijo in con_fijos.items():
            if metrica not in con_optuna:
                continue
            fila[f'{metrica}_fijo'] = valor_fijo
            fila[f'{metrica}_optuna'] = con_optuna[metrica]
            fila[f'{metrica}_delta'] = round(con_optuna[metrica] - valor_fijo, 4)
        filas.append(fila)
    pd.DataFrame(filas).to_csv(ruta_csv, index=False, encoding='utf-8')
    return ruta_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--sin-mapa', action='store_true',
                        help='Omite el recall LORO cartográfico (la parte más lenta)')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza recalcular aunque el resultado ya esté actualizado')
    parser.add_argument('--optimizar', action='store_true',
                        help='Reoptimiza con Optuna los hiperparámetros de CADA experimento '
                             '(misma semilla y espacio de búsqueda que el pipeline) y guarda '
                             f'todo en la subcarpeta {SUBCARPETA_OPTUNA}/, sin tocar el estudio '
                             'con hiperparámetros fijos')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])

    salida = ruta_salida(config, optimizar=args.optimizar)
    if not args.regenerar and _esta_actualizado(
            entradas_esperadas(config, ruta_config, args.optimizar), [salida]):
        print(f"  [OK] Ya existe y está actualizado: {salida} (usa --regenerar para forzar)")
        return 0
    dir_salida = os.path.dirname(salida)
    os.makedirs(dir_salida, exist_ok=True)
    # Las capas filtradas se regeneran en cada corrida: una capa vieja no debe colarse.
    for nombre in os.listdir(dir_salida):
        if nombre.endswith('.gpkg'):
            os.remove(os.path.join(dir_salida, nombre))

    bloque = config.get('auditoria_infraestructura') or {}
    margen_meses = int(bloque.get('margen_meses', 12))
    random_state = config['ml_params']['random_state']
    tamano_bloque_km = config.get('validacion', {}).get('tamano_bloque_km', 30)
    vectores = config['paths']['raw']['vectores']
    codigos = config['zona_estudio']['codigos']

    print("Cargando dataset de entrenamiento, infraestructura y plantas...")
    muestras = _cargar_muestras(config)
    params_fijos = _params_rf(config)
    crudas = {nombre: gpd.read_file(vectores[nombre]) for nombre in CAPA_POR_FEATURE.values()}
    # El almacenamiento no trae TIPO: se hereda de la subestación a la que se conecta.
    crudas['almacenamiento'] = asignar_tipo_por_subestacion(crudas['almacenamiento'],
                                                            crudas['subestaciones'])
    capas = {nombre: preparar_infraestructura(capa, codigos) for nombre, capa in crudas.items()}
    plantas = gpd.read_file(vectores['fotovoltaicas'])
    fechas = asignar_fechas_referencia(muestras, plantas, random_state=random_state)

    emparejamiento_max = float(fechas['dist_emparejamiento_m'].max())
    print(f"  {len(muestras)} muestras | emparejamiento planta-positiva: máx {emparejamiento_max:.1f} m")
    for nombre, capa in capas.items():
        n_dedicados = int((capa['TIPO'].astype(str).str.upper().str.strip() == 'DEDICADO').sum())
        print(f"  {nombre}: {len(capa)} elementos en la zona ({n_dedicados} dedicados, "
              f"{int(capa['f_op'].isna().sum())} sin fecha)")

    diagnostico = {}
    for nombre, capa in capas.items():
        diagnostico[nombre] = diagnosticar_fuga(muestras, capa, fechas, margen_meses)
        d = diagnostico[nombre]
        print(f"  {nombre}: más cercana DEDICADA {d['pct_mas_cercana_dedicada']}% | no previa "
              f"{d['pct_mas_cercana_no_previa']}% | mismo dueño {d['pct_mas_cercana_mismo_dueno']}%")

    if args.optimizar:
        descripcion = ("Anexo Optuna de la ablación de distancias a la infraestructura: mismos "
                       "420 puntos y mismas otras variables; los hiperparámetros se reoptimizan "
                       "en cada experimento con el mismo procedimiento del pipeline (TPE, "
                       "misma semilla, mismo espacio de búsqueda, Brier SBCV).")
    else:
        descripcion = ("Ablación de las distancias a la red: mismos 420 puntos, mismas otras "
                       "variables y mismos hiperparámetros (model_rf_metrics.json); solo "
                       "cambia qué subestaciones, líneas y almacenamiento se consideran.")
    resultado = {
        'descripcion': descripcion,
        'hiperparametros_optimizados': args.optimizar,
        'margen_meses': margen_meses,
        'random_state': random_state,
        'hiperparametros': params_fijos,
        'n_negativos_con_fecha_sorteada': int((fechas['fecha_imputada'] & (muestras['clase'] == 0)).sum()),
        'n_positivas_con_fecha_sorteada': int((fechas['fecha_imputada'] & (muestras['clase'] == 1)).sum()),
        'emparejamiento_planta_max_m': round(emparejamiento_max, 2),
        'diagnostico_fuga': diagnostico,
        'definicion_experimentos': {e: variantes_del_experimento(e) for e in EXPERIMENTOS},
        'experimentos': {},
    }

    # Las distancias de cada (feature, variante) se calculan una sola vez y se reutilizan
    # en todos los experimentos que las combinan.
    distancias = {}
    for feature, nombre_capa in CAPA_POR_FEATURE.items():
        for variante in sorted({variantes_del_experimento(e)[feature] for e in EXPERIMENTOS}):
            distancias[(feature, variante)] = distancia_a_infraestructura(
                muestras, capas[nombre_capa], variante, fechas['fecha_ref'], margen_meses)

    # Control: la línea base debe reproducir las variables del dataset, o la comparación
    # entre experimentos mediría también un error de cálculo.
    resultado['control_reconstruccion_max_dif_m'] = {}
    for feature in CAPA_POR_FEATURE:
        diferencia = float(np.nanmax(np.abs(distancias[(feature, 'todas')]
                                            - muestras[feature].values)))
        resultado['control_reconstruccion_max_dif_m'][feature] = round(diferencia, 3)
        print(f"  Control {feature}: diferencia máxima con el dataset = {diferencia:.3f} m")
        if diferencia > TOLERANCIA_DISTANCIA_M:
            raise RuntimeError(
                f"La variante 'todas' no reproduce {feature} del dataset (diferencia máx "
                f"{diferencia:.1f} m): la ablación no sería comparable.")

    muestras_por_experimento = {}
    params_por_experimento = {}
    optuna_config = config.get('optuna_params', {})
    for experimento in EXPERIMENTOS:
        print(f"\n=== Experimento '{experimento}' {variantes_del_experimento(experimento)} ===")
        muestras_e = muestras.copy()
        for feature, variante in variantes_del_experimento(experimento).items():
            muestras_e[feature] = distancias[(feature, variante)]
        muestras_por_experimento[experimento] = muestras_e

        if args.optimizar:
            print(f"  Optimizando hiperparámetros con Optuna "
                  f"({optuna_config.get('n_trials', 30)} intentos, semilla {random_state})...")
            params_e, mejor_brier = optimizar_hiperparametros(
                muestras_e, optuna_config, tamano_bloque_km, random_state)
            print(f"  -> {params_e} | mejor Brier SBCV {mejor_brier:.4f}")
        else:
            params_e, mejor_brier = params_fijos, None
        params_por_experimento[experimento] = params_e

        evaluacion = evaluar_variante(muestras_e, params_e, tamano_bloque_km, random_state)
        evaluacion['distancias'] = {f: resumir_distancias(muestras_e, f) for f in CAPA_POR_FEATURE}
        if args.optimizar:
            evaluacion['hiperparametros'] = params_e
            evaluacion['mejor_brier_optuna'] = round(mejor_brier, 4)
            evaluacion['hiperparametros_iguales_a_fijos'] = params_e == params_fijos
        resultado['experimentos'][experimento] = evaluacion

    if args.optimizar:
        # Control de reproducibilidad: la línea base es el mismo problema que optimizó el
        # pipeline, así que con la misma semilla debería elegir los mismos hiperparámetros.
        reproduce = params_por_experimento['linea_base'] == params_fijos
        resultado['control_optuna_linea_base_reproduce_pipeline'] = reproduce
        if not reproduce:
            print(f"  [AVISO] La línea base optimizada eligió {params_por_experimento['linea_base']}"
                  f" y no {params_fijos} (model_rf_metrics.json).")

    if not args.sin_mapa:
        print("\n=== Recall LORO cartográfico (top-K sobre el mapa) ===")
        resultado['recall_loro_cartografico'] = {}
        for experimento in (e for e in EXPERIMENTOS if tiene_mapa(e)):
            print(f"\n  Experimento '{experimento}'...")
            resultado['recall_loro_cartografico'][experimento] = _recall_loro_cartografico(
                config, muestras_por_experimento[experimento], params_por_experimento[experimento],
                capas, experimento, dir_salida, random_state)
        resultado['control_recall_publicado_max_dif'] = _controlar_recall_publicado(
            config, resultado['recall_loro_cartografico']['linea_base']['recall_por_k'],
            estricto=params_por_experimento['linea_base'] == params_fijos)
        resultado['recall_loro_cartografico']['nota'] = (
            "Los experimentos con 'previas' no tienen mapa: la red visible depende de la fecha "
            "de cada planta. 'linea_base' reproduce out_of_sample_loro de metricas_topk.json.")

    # Todos los productos del estudio viven juntos en su carpeta: JSON, tabla, figuras y
    # capas filtradas. No se mezclan con figures/ ni con los artefactos del pipeline.
    resultado['figuras'] = {}
    for nombre, dibujar in FIGURAS.items():
        ruta_figura = os.path.join(dir_salida, nombre)
        dibujar(resultado, ruta_figura)
        resultado['figuras'][nombre] = ruta_figura

    ruta_tabla = os.path.join(dir_salida, 'comparativa_variantes.csv')
    _guardar_tabla_comparativa(resultado, ruta_tabla)
    resultado['tabla_comparativa'] = ruta_tabla

    if args.optimizar:
        resultado['tabla_fijos_vs_optuna'] = _guardar_comparativa_fijos_vs_optuna(
            resultado, ruta_salida(config, optimizar=False),
            os.path.join(dir_salida, 'comparativa_fijos_vs_optuna.csv'))

    with open(salida, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)

    abreviatura = {'dist_subestaciones': 'sub', 'dist_transmision': 'tra', 'dist_almacen': 'alm'}
    print("\n" + "=" * 120)
    print("RESUMEN — AUDITORÍA DE LAS DISTANCIAS A LA INFRAESTRUCTURA")
    print("=" * 120)
    encabezado = f"{'experimento':<22}{'AUC SBCV':>9}{'AUC LORO':>9}"
    encabezado += ''.join(f"{'AUC ' + abreviatura[f]:>9}" for f in CAPA_POR_FEATURE)
    encabezado += ''.join(f"{'SHAP ' + abreviatura[f]:>9}" for f in CAPA_POR_FEATURE)
    encabezado += f"{'SHAP red':>9}{'SHAP infra':>11}{'SHAP slope':>11}{'SHAP ghi':>9}"
    print(encabezado)
    for experimento, v in resultado['experimentos'].items():
        s = v['importancia_shap_pct']
        fila = f"{experimento:<22}{v['auc_sbcv_media']:>9.3f}{v['auc_loro_media']:>9.3f}"
        fila += ''.join(f"{v['auc_univariado'][f]:>9.3f}" for f in CAPA_POR_FEATURE)
        fila += ''.join(f"{s[f]:>9.1f}" for f in CAPA_POR_FEATURE)
        fila += (f"{v['shap_pct_acceso_red']:>9.1f}{v['shap_pct_infraestructura']:>11.1f}"
                 f"{s['slope']:>11.1f}{s['ghi']:>9.1f}")
        print(fila)
    for experimento, r in (resultado.get('recall_loro_cartografico') or {}).items():
        if isinstance(r, dict):
            texto = ' | '.join(f"K{k}: {v['recall']:.3f}" for k, v in r['recall_por_k'].items())
            print(f"  Recall LORO mapa '{experimento}': {texto}")
    print(f"\nResultados en: {dir_salida}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
