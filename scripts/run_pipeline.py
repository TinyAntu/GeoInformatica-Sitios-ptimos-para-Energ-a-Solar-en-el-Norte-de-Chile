import sys
import os
import argparse
import subprocess
import yaml

# Permite importar módulos desde la raíz del proyecto
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.preprocessing import cargar_capas_vectoriales, procesar_dem, reproject_raster_to_utm
from src.sampling import generar_dataset_muestras
from src.modeling import entrenar_modelo_rf
from src.utils import _esta_actualizado


def _resolver_rutas(obj, base_dir: str):
    """Convierte recursivamente todas las rutas relativas del config a absolutas."""
    if isinstance(obj, str):
        return os.path.join(base_dir, obj) if not os.path.isabs(obj) else obj
    if isinstance(obj, dict):
        return {k: _resolver_rutas(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolver_rutas(item, base_dir) for item in obj]
    return obj


def _shapefile_paths(shp_path: str) -> list:
    base, _ = os.path.splitext(shp_path)
    return [base + ext for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']]


def _correr_etapa(nombre_etapa: str, script: str, args_extra: list | None = None,
                  motor_best_effort: bool = False) -> bool:
    """Corre `scripts/<script>` como subproceso independiente.
    Cada etapa se ejecuta en su propio proceso (no como import + llamada a `main()` en el
    mismo proceso de run_pipeline.py, para evitar problemas de memoria y dependencias compartidas).

    Devuelve True si la etapa corrió con éxito (código 0), False si se omitió en modo
    best-effort. Si no es best-effort y falla, aborta el proceso completo.
    """
    ruta_script = os.path.join(directorio_raiz, 'scripts', script)
    cmd = [sys.executable, ruta_script] + (args_extra or [])
    print(f"\n--- {nombre_etapa} ---")
    resultado = subprocess.run(cmd, cwd=directorio_raiz)
    if resultado.returncode != 0:
        if motor_best_effort:
            print(f"  [AVISO] '{nombre_etapa}' terminó con código {resultado.returncode} "
                  "(probablemente el motor Rust solarpv-rs no está compilado, ver AGENTS.md "
                  "sección 6). Se omite y el pipeline continúa con el resto de las etapas.")
            return False
        print(f"ERROR: '{nombre_etapa}' falló con código {resultado.returncode}.")
        sys.exit(resultado.returncode)
    return True


def main():
    parser = argparse.ArgumentParser(description='Pipeline Solar Norte de Chile')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--sin-mapas', action='store_true',
                        help='Omite todas las etapas posteriores al entrenamiento (5 en adelante); '
                             'útil para iterar solo en el modelo')
    args = parser.parse_args()

    print("Iniciando Pipeline Solar...")

    ruta_config = os.path.join(directorio_raiz, args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resuelve todas las rutas relativas del config contra la raíz del proyecto
    config['paths'] = _resolver_rutas(config['paths'], directorio_raiz)

    print("Configuración cargada exitosamente.")

    vectores = cargar_capas_vectoriales(config['paths']['raw']['vectores'])
    print(f"Se cargaron {len(vectores)} capas vectoriales.")

    processed = config['paths']['processed']
    results = config['paths']['results']

    dem_out    = processed['dem_32719']
    slope_out  = processed['slope']
    aspect_out = processed['aspect']
    ghi_utm_out = processed['ghi_32719']

    # --- Etapa 1: DEM ---
    # procesar_dem() omite el proceso si los archivos ya existen en procesados.
    procesar_dem(
        carpetas_dem=config['paths']['raw']['rasters']['dem_folders'],
        out_slope_path=slope_out,
        out_aspect_path=aspect_out,
        out_dem_path=dem_out,
        resolucion_m=config.get('preprocesamiento', {}).get('dem_resolucion_m'),
    )

    # --- Etapa 2: Reproyección GHI ---
    # reproject_raster_to_utm() omite el proceso si el archivo ya existe en procesados.
    raw_ghi = config['paths']['raw']['rasters']['ghi']
    reproject_raster_to_utm(raw_ghi, ghi_utm_out, epsg_code=32719)

    # --- Etapa 3: Muestreo y entrenamiento ---
    dataset_out = results['dataset_ml']
    model_out   = results.get('model_rf')

    entradas_dataset = [ruta_config, raw_ghi, ghi_utm_out, slope_out, aspect_out, dem_out]
    entradas_dataset.extend(config['paths']['raw']['vectores'].values())
    salidas_dataset = _shapefile_paths(dataset_out)
    if model_out:
        salidas_dataset.append(model_out)

    if _esta_actualizado(entradas_dataset, salidas_dataset):
        print("El dataset y modelo ya están actualizados. No se requiere reprocesar entrenamiento.")
    else:
        os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
        if model_out:
            os.makedirs(os.path.dirname(model_out), exist_ok=True)

        rutas_rasters = {
            'ghi_32719':  ghi_utm_out,
            'slope':      slope_out,
            'aspect':     aspect_out,
            'dem_32719':  dem_out,
        }

        ml = config['ml_params']
        positivas, pool_negativos = generar_dataset_muestras(
            vectores, rutas_rasters, config['criterios'],
            ratio=ml['ratio_negativos'],
            random_state=ml['random_state'],
        )
        entrenar_modelo_rf(
            positivas=positivas,
            pool_negativos=pool_negativos,
            ratio=ml['ratio_negativos'],
            out_shp=dataset_out,
            optuna_config=config.get('optuna_params', {}),
            random_state=ml['random_state'],
            out_model_path=model_out,
            n_estimators=ml['n_estimators'],
            ratio_alt=ml.get('ratio_alt'),
            tamano_bloque_km=config.get('validacion', {}).get('tamano_bloque_km', 15),
            # Filtrado a Antofagasta+Atacama antes de pasarlo: el shapefile completo de
            # regiones incluye la costa patagónica (miles de vértices) y reproyectar/hacer
            # sjoin contra Chile completo es innecesariamente caro en memoria/tiempo.
            regiones_gdf=vectores['regiones'][
                vectores['regiones']['REGION'].isin(['Antofagasta', 'Atacama'])
            ],
        )

    # --- Etapa 4: Validación espacial ---
    _correr_etapa("Etapa 4: Validación espacial (SBCV + LOROCV)", "run_spatial_validation.py")

    if args.sin_mapas:
        print("\nEtapas posteriores al entrenamiento omitidas (--sin-mapas).")
        print("Pipeline ejecutado correctamente.")
        return

    # Las rutas de salida de las etapas 5-7 están definidas dentro de cada script;
    # aquí se replican solo para el chequeo incremental (_esta_actualizado).
    rasters_procesados = [dem_out, slope_out, aspect_out, ghi_utm_out]

    # --- Etapa 5: Mapa de probabilidad RF ---
    mapa_rf = os.path.join(directorio_raiz, 'data/results/mapa_probabilidad_aptitud.tif')
    entradas_mapa = [ruta_config] + rasters_procesados
    if model_out:
        entradas_mapa.append(model_out)
    if _esta_actualizado(entradas_mapa, [mapa_rf]):
        print("\n--- Etapa 5: Mapa de probabilidad RF --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 5: Mapa de probabilidad RF", "generate_suitability_map.py")

    # --- Etapa 6: Mapas de perfiles AHP/WLC ---
    mapas_perfiles = [os.path.join(directorio_raiz, f'data/results/aptitud_{p}.tif')
                      for p in ('conservador', 'agresivo')]
    if _esta_actualizado([ruta_config] + rasters_procesados, mapas_perfiles):
        print("\n--- Etapa 6: Mapas de perfiles de inversión (AHP/WLC) --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 6: Mapas de perfiles de inversión (AHP/WLC)", "profiles.py")

    # --- Etapa 7: Validación de Coordenadas e Ingesta PostGIS ---
    # Comportamiento sin cambios: intenta conexión PostGIS en vivo; si no hay BD, cae de
    # vuelta a solo generar el SQL (la propia validate_and_load_postgis.py ya lo maneja).
    _correr_etapa("Etapa 7: Validación de Coordenadas e Ingesta PostGIS",
                 "validate_and_load_postgis.py", ['--config', args.config])

    # --- Etapa 8: Rendimiento PV fijo (motor Rust solarpv-rs) ---
    # Dependencia externa opcional (ver AGENTS.md sección 6): best-effort, no rompe el
    # pipeline si el binario no está compilado.
    out_prefix_rendimiento = os.path.join(
        directorio_raiz, config.get('solarpv', {}).get('out_prefix', 'data/results/rendimiento'))
    rendimiento_fijo = out_prefix_rendimiento + '_fijo_specific_yield.tif'
    if _esta_actualizado([ruta_config, dem_out], [rendimiento_fijo]):
        print("\n--- Etapa 8: Rendimiento PV fijo (motor Rust) --- (ya actualizado, se omite)")
        motor_ok = True
    else:
        motor_ok = _correr_etapa("Etapa 8: Rendimiento PV fijo (motor Rust)",
                                 "run_solar_yield.py", ['--config', args.config],
                                 motor_best_effort=True)

    # --- Etapa 9: Cruce aptitud RF x rendimiento físico ---
    # run_cruce.py ya se omite solo (con aviso, código 0) si falta el rendimiento fijo.
    salidas_cruce = [os.path.join(directorio_raiz, 'data/results/aptitud_x_rendimiento.tif'),
                     os.path.join(directorio_raiz, 'data/results/rendimiento_en_aptas.tif')]
    if os.path.exists(rendimiento_fijo) and _esta_actualizado([mapa_rf, rendimiento_fijo], salidas_cruce):
        print("\n--- Etapa 9: Cruce aptitud x rendimiento --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 9: Cruce aptitud x rendimiento", "run_cruce.py", ['--config', args.config])

    # --- Etapa 10: Comparación fijo vs. seguidor ---
    # run_comparacion_montaje.py genera el rendimiento seguidor por su cuenta si falta
    # (ver src/comparacion_montaje.py -> _asegurar_rendimiento); también best-effort.
    comparacion_json = os.path.join(directorio_raiz, 'data/results/comparacion_montaje.json')
    if _esta_actualizado([ruta_config, mapa_rf], [comparacion_json]):
        print("\n--- Etapa 10: Comparación fijo vs. seguidor --- (ya actualizado, se omite)")
        comparacion_ok = True
    else:
        comparacion_ok = _correr_etapa("Etapa 10: Comparación fijo vs. seguidor",
                                       "run_comparacion_montaje.py", ['--config', args.config],
                                       motor_best_effort=True)

    # --- Etapa 11: Consenso vs. divergencia entre perfiles (Brecha 6) ---
    perfiles_tif = [mapa_rf] + mapas_perfiles  # RF (balanceado) + conservador + agresivo
    salida_consenso = [os.path.join(directorio_raiz, 'data/results/consenso_perfiles.tif')]
    if _esta_actualizado(perfiles_tif, salida_consenso):
        print("\n--- Etapa 11: Consenso/divergencia entre perfiles --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 11: Consenso/divergencia entre perfiles", "run_consenso.py",
                      ['--config', args.config])

    # --- Etapa 12: Explicabilidad SHAP global (T6) ---
    shap_json = os.path.join(directorio_raiz, 'data/results/shap_importancias.json')
    entradas_shap = [ruta_config]
    if model_out:
        entradas_shap.append(model_out)
    if _esta_actualizado(entradas_shap, [shap_json]):
        print("\n--- Etapa 12: Explicabilidad SHAP global (T6) --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 12: Explicabilidad SHAP global (T6)", "run_shap.py", ['--config', args.config])

    # --- Etapa 13: Explicabilidad SHAP espacial (Brecha 8) ---
    shap_espacial_json = os.path.join(directorio_raiz, 'data/results/shap_espacial.json')
    if _esta_actualizado(entradas_shap, [shap_espacial_json]):
        print("\n--- Etapa 13: Explicabilidad SHAP espacial (Brecha 8) --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 13: Explicabilidad SHAP espacial (Brecha 8)", "run_shap_spatial.py",
                      ['--config', args.config])

    # --- Etapa 14: Métricas Recall@K / Precisión@K contra los umbrales de PEP1 ---
    # Cierra los dos umbrales de PEP1 §6.5 que no se calculaban. Necesita el modelo y el
    # mapa RF ya generados (etapas 3 y 5).
    metricas_topk_json = os.path.join(directorio_raiz, 'data/results/metricas_topk.json')
    entradas_topk = [ruta_config, mapa_rf] + ([model_out] if model_out else [])
    if _esta_actualizado(entradas_topk, [metricas_topk_json]):
        print("\n--- Etapa 14: Métricas Recall@K / Precisión@K --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 14: Métricas Recall@K / Precisión@K", "run_metricas_topk.py",
                      ['--config', args.config])

    # --- Etapa 15: Figuras cartográficas del informe ---
    # Va DESPUÉS de las etapas 8-13 a propósito: 5 de sus 7 figuras (rendimiento fijo/seguidor,
    # cruce, consenso, variable dominante SHAP y comparación de montaje) se dibujan sobre
    # rásters/JSON que recién existen a esta altura. Cuando esta etapa corría antes (era la 7),
    # generar_figuras_informe.py las omitía con [AVISO] en toda corrida limpia y nunca se
    # regeneraban, porque el chequeo incremental solo miraba las 3 figuras de aptitud.
    figuras = [os.path.join(directorio_raiz, 'figures', nombre)
               for nombre in ('mapa_aptitud_rf.png', 'mapa_aptitud_conservador.png',
                              'mapa_aptitud_agresivo.png', 'mapa_cruce_aptitud_rendimiento.png',
                              'mapa_consenso_perfiles.png', 'mapa_shap_dominante.png',
                              'comparacion_fijo_vs_seguidor.png', 'metricas_validacion.png')]
    entradas_figuras = [p for p in ([mapa_rf] + mapas_perfiles + salidas_cruce +
                                    salida_consenso + [comparacion_json, metricas_topk_json])
                        if os.path.exists(p)]
    if _esta_actualizado(entradas_figuras, figuras):
        print("\n--- Etapa 15: Figuras cartográficas (7 elementos) --- (ya actualizado, se omite)")
    else:
        _correr_etapa("Etapa 15: Figuras cartográficas (7 elementos)", "generar_figuras_informe.py")

    # --- Etapa 16: Assets del visor ---
    # Siempre se regenera: lee todos los resultados de las etapas anteriores y es liviana
    # (reprojecta/reduce a PNG chicos), así que no vale la pena mantener una lista larga de
    # dependencias para el chequeo incremental.
    _correr_etapa("Etapa 16: Assets del visor", "generate_web_assets.py", ['--config', args.config])

    # --- Etapa 17: Resumen final ---
    print("\n" + "=" * 70)
    print("Pipeline ejecutado correctamente.")
    if not motor_ok or not comparacion_ok:
        print("[AVISO] El motor Rust (solarpv-rs) no estaba disponible: se omitieron el "
              "rendimiento PV y/o la comparación fijo/seguidor. El resto de los resultados "
              "(mapa RF, perfiles AHP, SHAP, consenso, assets) sí se generaron completos.")
    print("Para ver los resultados en el visor local:")
    print("    streamlit run app/visor.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
