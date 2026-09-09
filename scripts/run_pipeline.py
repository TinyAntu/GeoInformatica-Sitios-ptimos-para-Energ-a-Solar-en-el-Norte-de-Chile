"""Orquestador del pipeline solar, organizado en 6 fases.

Las fases siguen el flujo canónico de un pipeline geoespacial
(datos crudos -> limpieza -> transformación -> análisis -> visualización):

    Fase 1  Preprocesamiento          raw -> processed (DEM, slope, aspect, GHI)
    Fase 2  Features y modelamiento   muestreo, Random Forest y validación espacial
    Fase 3  Inferencia y superficies  mapas de aptitud (RF y AHP) y rendimiento físico
    Fase 4  Análisis integrado        cruces, consenso, explicabilidad y métricas
    Fase 5  Transferibilidad          el modelo ya entrenado sobre otra región (opt-in)
    Fase 6  Persistencia y difusión   PostGIS, figuras del informe y assets del visor

Cada etapa corre en su PROPIO PROCESO (`src/utils.py::_correr_etapa`). Eso no es un detalle
de estilo: es lo que hace que la memoria de una etapa pesada se devuelva al sistema cuando
termina. Un hilo no serviría —los hilos comparten el heap del proceso—, y por eso este
orquestador no retiene ningún dato geoespacial: ni siquiera las capas vectoriales, que hoy
carga `run_entrenamiento.py` dentro de su propio proceso.

Las etapas se declaran como DATOS (la tabla FASES) y un único bucle las ejecuta. Agregar una
etapa es agregar una fila, no otro bloque if/else.

Uso:
    python scripts/run_pipeline.py --config config.yaml
    python scripts/run_pipeline.py --listar-fases            # ver el plan sin ejecutar
    python scripts/run_pipeline.py --desde-fase 3            # reanudar tras un fallo
    python scripts/run_pipeline.py --con-transferencia       # incluir la fase 5
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Callable

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _esta_actualizado, _correr_etapa, _resolver_rutas, _ruta_abs


# --------------------------------------------------------------------------------------
# Modelo de datos
# --------------------------------------------------------------------------------------

@dataclass
class Etapa:
    """Una etapa del pipeline.

    Las rutas se declaran como funciones de `ctx` (no como valores) porque dependen de la
    configuración, que se resuelve en tiempo de ejecución.

    `guarda` devuelve el motivo por el que la etapa debe omitirse, o None si procede.
    `funcion` es para las etapas que no son un script externo (el resumen final).
    """
    numero: str
    nombre: str
    script: str | None = None
    args_extra: Callable[[dict], list] = field(default=lambda ctx: [])
    entradas: Callable[[dict], list] = field(default=lambda ctx: [])
    salidas: Callable[[dict], list] = field(default=lambda ctx: [])
    best_effort: bool = False
    guarda: Callable[[dict], str | None] = field(default=lambda ctx: None)
    funcion: Callable[[dict], bool] | None = None


@dataclass
class Fase:
    numero: int
    nombre: str
    proposito: str
    etapas: list


# --------------------------------------------------------------------------------------
# Guardas y helpers de la fase 5 (transferibilidad)
# --------------------------------------------------------------------------------------

def _guarda_transferencia(ctx) -> str | None:
    """Las tres condiciones que pueden dejar fuera la evaluación de otra región."""
    if not ctx['args'].con_transferencia:
        return "omitida; usa --con-transferencia"
    if not ctx['bloque_transf']:
        return "config.yaml no tiene el bloque 'transferibilidad'."
    faltantes = [c for c in ctx['carpetas_transf'] if not os.path.isdir(c)]
    if faltantes:
        return f"falta el DEM crudo: {', '.join(faltantes)}"
    return None


def _guarda_assets_transferencia(ctx) -> str | None:
    if not ctx['args'].con_transferencia:
        return "omitida; usa --con-transferencia"
    if not (ctx['metricas_transf'] and os.path.exists(ctx['metricas_transf'])):
        return "la zona transferida no produjo métricas"
    return None


def _resumen_final(ctx) -> bool:
    """Etapa 18: cierre informativo, sin efectos sobre el disco."""
    resultados = ctx['resultados']
    motor_ok = resultados.get('8', True)
    comparacion_ok = resultados.get('10', True)
    print("\n" + "=" * 70)
    print("Pipeline ejecutado correctamente.")
    if not motor_ok or not comparacion_ok:
        print("[AVISO] El motor Rust (solarpv-rs) no estaba disponible: se omitieron el "
              "rendimiento PV y/o la comparación fijo/seguidor. El resto de los resultados "
              "(mapa RF, perfiles AHP, SHAP, consenso, assets) sí se generaron completos.")
    if not ctx['args'].con_transferencia:
        print("[NOTA] La evaluación de generalización sobre otra región no se ejecutó. "
              "Añade --con-transferencia para incluirla.")
    print("Para ver los resultados en el visor local:")
    print("    streamlit run app/visor.py")
    print("=" * 70)
    return True


# --------------------------------------------------------------------------------------
# La tabla de fases: el pipeline completo declarado como datos
# --------------------------------------------------------------------------------------
#
# Nota sobre `--config`: tres scripts (run_spatial_validation, generate_suitability_map y
# generar_figuras_informe) no tienen argparse y abren 'config.yaml' literal, por eso se
# invocan sin argumentos. Corregirlo es un cambio aparte; aquí se refleja el estado real.

FASES = [
    Fase(1, "Preprocesamiento", "datos crudos -> rásters procesados en EPSG:32719", [
        Etapa('1-2', "DEM, slope/aspect y reproyección del GHI",
              script="run_preprocesamiento.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config]),
    ]),

    Fase(2, "Features y modelamiento", "muestreo espacial, Random Forest y validación espacial", [
        # La frescura del dataset/modelo la decide el propio script: es el único que conoce
        # las 5 rutas del shapefile ESRI y las capas vectoriales de entrada.
        Etapa('3', "Muestreo y entrenamiento del Random Forest",
              script="run_entrenamiento.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config]),
        Etapa('4', "Validación espacial (SBCV + LOROCV)",
              script="run_spatial_validation.py"),
    ]),

    Fase(3, "Inferencia y superficies", "mapas de aptitud y rendimiento físico celda a celda", [
        Etapa('5', "Mapa de probabilidad RF",
              script="generate_suitability_map.py",
              entradas=lambda ctx: [ctx['ruta_config']] + ctx['rasters_procesados'] +
                                   ([ctx['model_out']] if ctx['model_out'] else []),
              salidas=lambda ctx: [ctx['mapa_rf']]),
        Etapa('6', "Mapas de perfiles de inversión (AHP/WLC)",
              script="profiles.py",
              entradas=lambda ctx: [ctx['ruta_config']] + ctx['rasters_procesados'],
              salidas=lambda ctx: ctx['mapas_perfiles']),
        # Dependencia externa opcional (ver AGENTS.md sección 6): best-effort, no rompe el
        # pipeline si el binario Rust no está compilado.
        Etapa('8', "Rendimiento PV fijo (motor Rust)",
              script="run_solar_yield.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: [ctx['ruta_config'], ctx['dem_out']],
              salidas=lambda ctx: [ctx['rendimiento_fijo']],
              best_effort=True),
    ]),

    Fase(4, "Análisis integrado", "cruces entre capas, consenso, explicabilidad y métricas", [
        Etapa('9', "Cruce aptitud x rendimiento",
              script="run_cruce.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: [ctx['mapa_rf'], ctx['rendimiento_fijo']],
              salidas=lambda ctx: ctx['salidas_cruce']),
        # run_comparacion_montaje.py genera el rendimiento seguidor por su cuenta si falta
        # (ver src/comparacion_montaje.py -> _asegurar_rendimiento); también best-effort.
        Etapa('10', "Comparación fijo vs. seguidor",
              script="run_comparacion_montaje.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: [ctx['ruta_config'], ctx['mapa_rf']],
              salidas=lambda ctx: [ctx['comparacion_json']],
              best_effort=True),
        Etapa('11', "Consenso/divergencia entre perfiles",
              script="run_consenso.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: [ctx['mapa_rf']] + ctx['mapas_perfiles'],
              salidas=lambda ctx: ctx['salida_consenso']),
        Etapa('12', "Explicabilidad SHAP global (T6)",
              script="run_shap.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: ctx['entradas_shap'],
              salidas=lambda ctx: [ctx['shap_json']]),
        Etapa('13', "Explicabilidad SHAP espacial (Brecha 8)",
              script="run_shap_spatial.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: ctx['entradas_shap'],
              salidas=lambda ctx: [ctx['shap_espacial_json']]),
        # Cierra los dos umbrales de PEP1 §6.5 que no se calculaban.
        Etapa('14', "Métricas Recall@K / Precisión@K",
              script="run_metricas_topk.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config],
              entradas=lambda ctx: [ctx['ruta_config'], ctx['mapa_rf']] +
                                   ([ctx['model_out']] if ctx['model_out'] else []),
              salidas=lambda ctx: [ctx['metricas_topk_json']]),
    ]),

    # Va ANTES de la fase 6 porque el visor y el manifest necesitan sus artefactos. Es
    # opt-in: requiere un DEM que no está en el repo y añade varios minutos.
    Fase(5, "Transferibilidad", "aplicar el modelo ya entrenado a una región no vista (Coquimbo)", [
        Etapa('15', "Transferencia a otra región",
              script="run_zona.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config, '--zona', 'transferibilidad'],
              entradas=lambda ctx: [ctx['ruta_config'], ctx['dem_transf']] +
                                   ([ctx['model_out']] if ctx['model_out'] else []),
              salidas=lambda ctx: [ctx['metricas_transf']],
              guarda=_guarda_transferencia),
    ]),

    Fase(6, "Persistencia y difusión", "ingesta a PostGIS, cartografía del informe y visor web", [
        # Validación de coordenadas e ingesta: intenta conexión PostGIS en vivo; si no hay
        # BD, cae de vuelta a solo generar el SQL (validate_and_load_postgis.py lo maneja).
        Etapa('7', "Validación de coordenadas e ingesta PostGIS",
              script="validate_and_load_postgis.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config]),
        # 5 de sus 8 figuras se dibujan sobre rásters/JSON que recién existen tras las fases
        # 3 y 4. El propio script entra como dependencia: sin él, un cambio de código en las
        # figuras (colores, escala gráfica) no dispararía la regeneración, porque todos los
        # datos de entrada seguirían siendo más viejos que los PNG ya dibujados.
        Etapa('16', "Figuras cartográficas del informe",
              script="generar_figuras_informe.py",
              entradas=lambda ctx: [p for p in ([ctx['mapa_rf']] + ctx['mapas_perfiles'] +
                                                ctx['salidas_cruce'] + ctx['salida_consenso'] +
                                                [ctx['comparacion_json'], ctx['metricas_topk_json'],
                                                 ctx['script_figuras']])
                                    if os.path.exists(p)],
              salidas=lambda ctx: ctx['figuras']),
        # Siempre se regenera: lee todos los resultados anteriores y es liviana (reproyecta
        # y reduce a PNG chicos), así que no vale la pena mantener una lista de dependencias.
        Etapa('17', "Assets del visor",
              script="generate_web_assets.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config]),
        # Los assets de la zona transferida van a un manifest aparte: así el visor desplegado
        # sigue funcionando aunque esa zona no se haya calculado nunca.
        Etapa('17b', "Assets del visor (transferencia)",
              script="generate_web_assets.py",
              args_extra=lambda ctx: ['--config', ctx['args'].config, '--zona', 'transferibilidad'],
              guarda=_guarda_assets_transferencia),
        Etapa('18', "Resumen final", funcion=_resumen_final),
    ]),
]


# --------------------------------------------------------------------------------------
# Construcción del contexto
# --------------------------------------------------------------------------------------

def construir_contexto(args) -> dict:
    """Resuelve toda la configuración a rutas absolutas. No abre ni un solo dataset."""
    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])

    processed = config['paths']['processed']
    results = config['paths']['results']

    dem_out = processed['dem_32719']
    mapa_rf = _ruta_abs('data/results/mapa_probabilidad_aptitud.tif')
    out_prefix = _ruta_abs(config.get('solarpv', {}).get('out_prefix', 'data/results/rendimiento'))

    bloque_transf = config.get('transferibilidad') or {}
    dir_transf = bloque_transf.get('dir_resultados', 'data/results/transferibilidad')

    ctx = {
        'args': args,
        'config': config,
        'ruta_config': ruta_config,
        'model_out': results.get('model_rf'),
        'dem_out': dem_out,
        'rasters_procesados': [dem_out, processed['slope'], processed['aspect'],
                               processed['ghi_32719']],
        'mapa_rf': mapa_rf,
        'mapas_perfiles': [_ruta_abs(f'data/results/aptitud_{p}.tif')
                           for p in ('conservador', 'agresivo')],
        'rendimiento_fijo': out_prefix + '_fijo_specific_yield.tif',
        'salidas_cruce': [_ruta_abs('data/results/aptitud_x_rendimiento.tif'),
                          _ruta_abs('data/results/rendimiento_en_aptas.tif')],
        'comparacion_json': _ruta_abs('data/results/comparacion_montaje.json'),
        'salida_consenso': [_ruta_abs('data/results/consenso_perfiles.tif')],
        'shap_json': _ruta_abs('data/results/shap_importancias.json'),
        'shap_espacial_json': _ruta_abs('data/results/shap_espacial.json'),
        'metricas_topk_json': _ruta_abs('data/results/metricas_topk.json'),
        'script_figuras': os.path.join(directorio_raiz, 'scripts', 'generar_figuras_informe.py'),
        'figuras': [os.path.join(directorio_raiz, 'figures', nombre) for nombre in
                    ('mapa_aptitud_rf.png', 'mapa_aptitud_conservador.png',
                     'mapa_aptitud_agresivo.png', 'mapa_cruce_aptitud_rendimiento.png',
                     'mapa_consenso_perfiles.png', 'mapa_shap_dominante.png',
                     'comparacion_fijo_vs_seguidor.png', 'metricas_validacion.png')],
        'bloque_transf': bloque_transf,
        'carpetas_transf': [_ruta_abs(c) for c in bloque_transf.get('dem_folders', [])],
        'dem_transf': _ruta_abs(bloque_transf['dem']) if bloque_transf.get('dem') else '',
        'metricas_transf': _ruta_abs(os.path.join(dir_transf, 'metricas_transferibilidad.json')),
        'resultados': {},
    }
    # Las etapas 12 y 13 comparten deliberadamente la misma lista de entradas: ambas dependen
    # solo del modelo y de la configuración.
    ctx['entradas_shap'] = [ruta_config] + ([ctx['model_out']] if ctx['model_out'] else [])
    return ctx


# --------------------------------------------------------------------------------------
# Ejecución
# --------------------------------------------------------------------------------------

def fases_seleccionadas(args) -> list:
    """Aplica --solo-fase / --desde-fase / --sin-mapas sobre la tabla FASES."""
    fases = FASES
    if args.solo_fase:
        return [f for f in fases if f.numero == args.solo_fase]
    if args.desde_fase:
        fases = [f for f in fases if f.numero >= args.desde_fase]
    if args.sin_mapas:
        fases = [f for f in fases if f.numero <= 2]
    return fases


def listar_fases(args) -> None:
    print("\nPlan del pipeline\n" + "=" * 70)
    for fase in fases_seleccionadas(args):
        print(f"\nFASE {fase.numero} — {fase.nombre}")
        print(f"  ({fase.proposito})")
        for etapa in fase.etapas:
            destino = etapa.script or 'función interna'
            marca = '  [motor opcional]' if etapa.best_effort else ''
            print(f"    Etapa {etapa.numero:<4} {etapa.nombre:<52} {destino}{marca}")
    print("\n" + "=" * 70)


def ejecutar_etapa(etapa: Etapa, ctx: dict) -> None:
    etiqueta = f"Etapa {etapa.numero}: {etapa.nombre}"

    motivo = etapa.guarda(ctx)
    if motivo:
        print(f"\n--- {etiqueta} --- [OMITIDA] {motivo}")
        return

    if etapa.funcion is not None:
        ctx['resultados'][etapa.numero] = etapa.funcion(ctx)
        return

    if _esta_actualizado(etapa.entradas(ctx), etapa.salidas(ctx)):
        print(f"\n--- {etiqueta} --- (ya actualizado, se omite)")
        ctx['resultados'][etapa.numero] = True
        return

    if ctx['args'].dry_run:
        print(f"\n--- {etiqueta} --- [DRY-RUN] correría: {etapa.script} "
              f"{' '.join(etapa.args_extra(ctx))}")
        ctx['resultados'][etapa.numero] = True
        return

    ctx['resultados'][etapa.numero] = _correr_etapa(
        etiqueta, etapa.script, etapa.args_extra(ctx), motor_best_effort=etapa.best_effort)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pipeline Solar Norte de Chile')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--sin-mapas', action='store_true',
                        help='Corre solo hasta la fase 2 (modelamiento); omite mapas y figuras. '
                             'Útil para iterar solo en el modelo')
    parser.add_argument('--con-transferencia', action='store_true',
                        help='Habilita la fase 5: aplica el modelo ya entrenado sobre la región '
                             'declarada en el bloque "transferibilidad" del config, sin '
                             'reentrenar, para medir su generalización. Se omite por defecto '
                             'porque requiere un DEM adicional que no todos tienen descargado.')
    parser.add_argument('--desde-fase', type=int, metavar='N',
                        help='Reanuda desde la fase N (útil tras un fallo o un OOM)')
    parser.add_argument('--solo-fase', type=int, metavar='N',
                        help='Corre únicamente la fase N')
    parser.add_argument('--listar-fases', action='store_true',
                        help='Imprime el plan de fases y etapas, sin ejecutar nada')
    parser.add_argument('--dry-run', action='store_true',
                        help='Muestra qué etapas correrían y cuáles están al día, sin ejecutarlas')
    args = parser.parse_args(argv)

    if args.listar_fases:
        listar_fases(args)
        return 0


    print("Iniciando Pipeline Solar...")
    ctx = construir_contexto(args)
    print("Configuración cargada exitosamente.")

    for fase in fases_seleccionadas(args):
        print("\n" + "=" * 70)
        print(f"FASE {fase.numero} — {fase.nombre.upper()}")
        print(f"({fase.proposito})")
        print("=" * 70)
        for etapa in fase.etapas:
            ejecutar_etapa(etapa, ctx)

    if args.sin_mapas:
        print("\nFases posteriores al modelamiento omitidas (--sin-mapas).")
        print("Pipeline ejecutado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
