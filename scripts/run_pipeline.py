import sys
import os
import argparse
import yaml

# Permite importar módulos desde la raíz del proyecto
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.preprocessing import cargar_capas_vectoriales, procesar_dem, reproject_raster_to_utm
from src.sampling import generar_dataset_muestras
from src.modeling import entrenar_modelo_rf
from src.utils import _esta_actualizado
from scripts.run_spatial_validation import main as run_spatial_validation
from scripts.generate_suitability_map import main as generar_mapa_rf
from scripts.profiles import main as generar_perfiles_ahp
from scripts.generar_figuras_informe import main as generar_figuras_informe
from scripts.validate_and_load_postgis import main as validar_postgis
from scripts.run_cruce import main as cruzar_aptitud_rendimiento
from scripts.run_consenso import main as analizar_consenso_perfiles

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


def main():
    parser = argparse.ArgumentParser(description='Pipeline Solar Norte de Chile')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--sin-mapas', action='store_true',
                        help='Omite las etapas de mapas y figuras (5-7); útil para iterar solo en el modelo')
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
        )

    # --- Etapa 4: Validación espacial ---
    print("Ejecutando validación espacial...")
    run_spatial_validation()

    if args.sin_mapas:
        print("\nEtapas de mapas y figuras omitidas (--sin-mapas).")
        print("Pipeline ejecutado correctamente.")
        return

    # Las rutas de salida de las etapas 5-7 están definidas dentro de cada script;
    # aquí se replican solo para el chequeo incremental (_esta_actualizado).
    rasters_procesados = [dem_out, slope_out, aspect_out, ghi_utm_out]

    # --- Etapa 5: Mapa de probabilidad RF ---
    print("\n--- Etapa 5: Mapa de probabilidad RF ---")
    mapa_rf = os.path.join(directorio_raiz, 'data/results/mapa_probabilidad_aptitud.tif')
    entradas_mapa = [ruta_config] + rasters_procesados
    if model_out:
        entradas_mapa.append(model_out)
    if _esta_actualizado(entradas_mapa, [mapa_rf]):
        print("El mapa de probabilidad ya está actualizado. Se omite.")
    else:
        generar_mapa_rf()

    # --- Etapa 6: Mapas de perfiles AHP/WLC ---
    print("\n--- Etapa 6: Mapas de perfiles de inversión (AHP/WLC) ---")
    mapas_perfiles = [os.path.join(directorio_raiz, f'data/results/aptitud_{p}.tif')
                      for p in ('conservador', 'agresivo')]
    if _esta_actualizado([ruta_config] + rasters_procesados, mapas_perfiles):
        print("Los mapas de perfiles ya están actualizados. Se omiten.")
    else:
        generar_perfiles_ahp()

    # --- Etapa 7: Figuras cartográficas del informe ---
    print("\n--- Etapa 7: Figuras cartográficas (7 elementos) ---")
    figuras = [os.path.join(directorio_raiz, 'figures', nombre)
               for nombre in ('mapa_aptitud_rf.png', 'mapa_aptitud_conservador.png',
                              'mapa_aptitud_agresivo.png')]
    if _esta_actualizado([mapa_rf] + mapas_perfiles, figuras):
        print("Las figuras cartográficas ya están actualizadas. Se omiten.")
    else:
        generar_figuras_informe()

    # --- Etapa 8: Validación de Coordenadas e Ingesta PostGIS ---
    print("\n--- Etapa 8: Validación de Coordenadas e Ingesta PostGIS ---")
    validar_postgis()

    # --- Etapa 9: Cruce aptitud RF x rendimiento físico (solarpv-rs) ---
    # Requiere el mapa de rendimiento (scripts/run_solar_yield.py, motor Rust compilado).
    # Si no existe, run_cruce.main() se omite solo con un aviso y no rompe el pipeline.
    print("\n--- Etapa 9: Cruce aptitud x rendimiento ---")
    rendimiento_fijo = os.path.join(directorio_raiz,
        config.get('solarpv', {}).get('out_prefix', 'data/results/rendimiento')
        + '_fijo_specific_yield.tif')
    salidas_cruce = [os.path.join(directorio_raiz, 'data/results/aptitud_x_rendimiento.tif'),
                     os.path.join(directorio_raiz, 'data/results/rendimiento_en_aptas.tif')]
    if os.path.exists(rendimiento_fijo) and _esta_actualizado([mapa_rf, rendimiento_fijo], salidas_cruce):
        print("El cruce ya está actualizado. Se omite.")
    else:
        cruzar_aptitud_rendimiento()

    # --- Etapa 10: Consenso vs. divergencia entre perfiles (Brecha 6) ---
    print("\n--- Etapa 10: Consenso/divergencia entre perfiles ---")
    perfiles_tif = [mapa_rf] + mapas_perfiles  # RF (balanceado) + conservador + agresivo
    salida_consenso = [os.path.join(directorio_raiz, 'data/results/consenso_perfiles.tif')]
    if _esta_actualizado(perfiles_tif, salida_consenso):
        print("El consenso de perfiles ya está actualizado. Se omite.")
    else:
        analizar_consenso_perfiles()

    print("\nPipeline ejecutado correctamente.")


if __name__ == "__main__":
    main()
