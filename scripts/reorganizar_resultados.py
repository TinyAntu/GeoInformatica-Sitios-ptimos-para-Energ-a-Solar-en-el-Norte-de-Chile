"""Mueve los resultados y figuras de la estructura plana anterior a las subcarpetas por tema.

Se corre UNA vez por copia del repositorio (sirve también para que el resto del equipo migre
la suya). La estructura nueva la fija src/rutas.py; aquí solo está la tabla de qué archivo
antiguo va a qué tema. Por defecto solo muestra lo que movería; `--aplicar` mueve.

Usa os.replace, que en el mismo disco conserva la fecha de modificación: el chequeo
incremental del pipeline (_esta_actualizado) no ve los archivos movidos como nuevos. No pisa
un destino que ya exista y al final lista lo que quedó sin clasificar.

Uso:
    python scripts/reorganizar_resultados.py --config config.yaml            # vista previa
    python scripts/reorganizar_resultados.py --config config.yaml --aplicar
"""

import argparse
import fnmatch
import os
import sys

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

# (patrón del nombre, tema) para los resultados del pipeline, en el norte y en cada zona.
# El primer patrón que calce gana.
TEMAS_RESULTADOS = [
    ('model_rf*', 'modelo'), ('dataset_entrenamiento_rf.*', 'modelo'),
    ('validacion_espacial.json', 'modelo'), ('metricas_topk.json', 'modelo'),
    ('mapa_probabilidad_aptitud.tif', 'aptitud'), ('aptitud_conservador.tif', 'aptitud'),
    ('aptitud_agresivo.tif', 'aptitud'), ('consenso_perfiles.*', 'aptitud'),
    ('rendimiento_en_aptas.tif', 'cruce'), ('aptitud_x_rendimiento.tif', 'cruce'),
    ('cruce_aptitud_rendimiento.json', 'cruce'),
    ('rendimiento_*.tif', 'rendimiento'), ('comparacion_montaje.json', 'rendimiento'),
    ('shap_*', 'explicabilidad'),
    ('instalaciones_optimas_validadas.*', 'postgis'), ('reporte_validacion_postgis.json', 'postgis'),
    ('cargar_instalaciones_postgis.sql', 'postgis'),
    ('metricas_transferibilidad.json', 'transferibilidad'), ('aoa.tif', 'transferibilidad'),
    ('di.tif', 'transferibilidad'),
]
TEMAS_FIGURAS = [
    ('feature_importances_rf.png', 'modelo'), ('roc_curve_ratio_*.png', 'modelo'),
    ('precision_recall_ratio_*.png', 'modelo'), ('sbcv_distribucion_folds.png', 'modelo'),
    ('metricas_validacion.png', 'modelo'),
    ('mapa_aptitud_*.png', 'aptitud'), ('mapa_consenso_perfiles.png', 'aptitud'),
    ('mapa_rendimiento_*.png', 'rendimiento'), ('comparacion_fijo_vs_seguidor.png', 'rendimiento'),
    ('mapa_cruce_aptitud_rendimiento.png', 'cruce'),
    ('shap_*.png', 'explicabilidad'), ('mapa_shap_dominante.png', 'explicabilidad'),
    ('mapa_aoa.png', 'transferibilidad'), ('mapa_di.png', 'transferibilidad'),
]


def _tema(nombre, tabla):
    return next((tema for patron, tema in tabla if fnmatch.fnmatch(nombre, patron)), None)


def _abs(ruta):
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def planificar(config) -> tuple:
    """Lista de movimientos (origen, destino) y de archivos sin clasificar."""
    res = config['paths']['results']
    base, figuras = _abs(res['base']), _abs(res['figuras'])
    analisis, historico = _abs(res['analisis']), _abs(res['historico'])
    fig_analisis = _abs(res['figuras_analisis'])
    movimientos, sin_clasificar = [], []

    def archivos_sueltos(carpeta):
        if not os.path.isdir(carpeta):
            return []
        return sorted(f for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f)))

    def por_tema(carpeta, tabla):
        for nombre in archivos_sueltos(carpeta):
            tema = _tema(nombre, tabla)
            if tema:
                movimientos.append((os.path.join(carpeta, nombre), os.path.join(carpeta, tema, nombre)))
            else:
                sin_clasificar.append(os.path.join(carpeta, nombre))

    # 1. Resultados y figuras del norte y de cada zona (misma tabla de temas).
    zonas = [b for b in config.values() if isinstance(b, dict) and b.get('dir_resultados')
             and b.get('dir_figuras')]
    for carpeta_res, carpeta_fig in [(base, figuras)] + [(_abs(z['dir_resultados']), _abs(z['dir_figuras']))
                                                          for z in zonas]:
        # La curva PR la escribía el mapa de aptitud en data/results/: es una figura del modelo.
        curva = os.path.join(carpeta_res, 'curva_precision_recall.png')
        if os.path.isfile(curva):
            movimientos.append((curva, os.path.join(carpeta_fig, 'modelo', 'curva_precision_recall.png')))
        por_tema(carpeta_res, TEMAS_RESULTADOS)
        por_tema(carpeta_fig, TEMAS_FIGURAS)

    # 2. Estudios fuera del pipeline: carpeta completa a analisis/.
    for nombre in ('auditoria_infraestructura', 'desacople', 'comparar_tiles'):
        movimientos.append((os.path.join(base, nombre), os.path.join(analisis, nombre)))
    for nombre, estudio in (('sensibilidad_muestreo.json', 'sensibilidad_muestreo'),
                            ('aoa_comparacion.json', 'aoa_lorocv'),
                            ('ablation_study.json', 'ablacion')):
        movimientos.append((os.path.join(base, nombre), os.path.join(analisis, estudio, nombre)))
    for nombre in ('sensibilidad_distribucion_negativos.png', 'sensibilidad_shap_comparacion.png'):
        movimientos.append((os.path.join(figuras, nombre),
                            os.path.join(fig_analisis, 'sensibilidad_muestreo', nombre)))

    # 3. Histórico: respaldo de cielo despejado y figuras que ya no genera ningún script.
    movimientos.append((os.path.join(base, 'cielo_despejado'), os.path.join(historico, 'cielo_despejado')))
    for nombre in ('mapa_bala.png', 'comparacion_mapas.png'):
        movimientos.append((os.path.join(figuras, nombre), os.path.join(figuras, 'historico', nombre)))

    # Los movimientos de la sección 1 ya clasificaron estos nombres como "sin tema": se
    # quitan de la lista de pendientes los que sí tienen destino en las secciones 2 y 3.
    con_destino = {o for o, _ in movimientos}
    sin_clasificar = [s for s in sin_clasificar if s not in con_destino]
    movimientos = [(o, d) for o, d in movimientos if os.path.exists(o)]
    return movimientos, sin_clasificar


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--aplicar', action='store_true', help='Mueve de verdad (sin esto, solo muestra)')
    args = parser.parse_args()
    with open(_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    movimientos, sin_clasificar = planificar(config)
    rel = lambda p: os.path.relpath(p, directorio_raiz)
    conflictos = 0
    for origen, destino in movimientos:
        if os.path.exists(destino):
            conflictos += 1
            print(f"  [OMITIDO, ya existe] {rel(origen)} -> {rel(destino)}")
            continue
        print(f"  {rel(origen)} -> {rel(destino)}")
        if args.aplicar:
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            os.replace(origen, destino)

    print(f"\n{len(movimientos) - conflictos} movimientos"
          f"{'' if args.aplicar else ' (vista previa: usa --aplicar para mover)'}; "
          f"{conflictos} omitidos porque el destino ya existe.")
    if sin_clasificar:
        print("Sin clasificar (quedan donde están):")
        for s in sin_clasificar:
            print(f"  {rel(s)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
