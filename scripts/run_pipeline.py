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

    # Ejecutamos la validacion espacial propuesta
    print("Ejecutando validación espacial...")
    run_spatial_validation()

    print("Pipeline ejecutado correctamente.")


if __name__ == "__main__":
    main()
