import sys
import os
import yaml

# 1. TRUCO DE RUTAS: Le decimos a Python que busque módulos en la carpeta principal (raíz)
# Esto soluciona el "ModuleNotFoundError: No module named 'src'"
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.preprocessing import cargar_capas_vectoriales, procesar_dem, reproject_raster_to_utm
from src.sampling import generar_dataset_muestras
from src.modeling import entrenar_modelo_rf


def _shapefile_paths(shp_path: str) -> list:
    base, _ = os.path.splitext(shp_path)
    return [base + ext for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg', '.qpj']]


def _esta_actualizado(paths_entrada: list, paths_salida: list) -> bool:
    # Verifica que todos los archivos de salida existan
    if not paths_salida or any(not os.path.exists(path) for path in paths_salida):
        return False
    
    # Obtiene la fecha más reciente de los archivos de entrada
    if not paths_entrada or any(not os.path.exists(path) for path in paths_entrada):
        return False
    
    tiempo_entrada_max = max(os.path.getmtime(path) for path in paths_entrada if os.path.exists(path))
    
    # Obtiene la fecha más antigua de los archivos de salida
    tiempo_salida_min = min(os.path.getmtime(path) for path in paths_salida if os.path.exists(path))
    
    # Retorna True si la salida es más reciente que la entrada
    return tiempo_salida_min > tiempo_entrada_max


def _archivos_existentes(paths: list) -> bool:
    return all(os.path.exists(path) for path in paths)


def main():
    print("Iniciando Pipeline Solar...")

    ruta_config = os.path.join(directorio_raiz, 'config.yaml')
    with open(ruta_config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    print("Configuración cargada exitosamente.")

    vectores = cargar_capas_vectoriales(config['paths']['raw']['vectores'])
    print(f"Se cargaron {len(vectores)} capas vectoriales.")

    processed = config['paths']['processed']
    results = config['paths']['results']

    dem_out = processed['dem_32718']
    slope_out = processed['slope']
    aspect_out = processed['aspect']
    ghi_utm_out = processed['ghi_32718']

    # Verificar independientemente cada archivo DEM
    slope_existe = os.path.exists(slope_out)
    aspect_existe = os.path.exists(aspect_out)
    dem_existe = os.path.exists(dem_out)
    
    if slope_existe and aspect_existe and dem_existe:
        print("Los archivos DEM ya existen. Se omite la fusión y el reprocesado.")
    else:
        if slope_existe:
            print(f"✓ Slope existe: {slope_out}")
        if aspect_existe:
            print(f"✓ Aspect existe: {aspect_out}")
        if not dem_existe:
            print(f"✗ DEM no existe: {dem_out} - será procesado")
        
        procesar_dem(
            carpetas_dem=config['paths']['raw']['rasters']['dem_folders'],
            out_slope_path=slope_out,
            out_aspect_path=aspect_out,
            out_dem_path=dem_out,
        )

    raw_ghi = config['paths']['raw']['rasters']['ghi']
    if not _esta_actualizado([raw_ghi], [ghi_utm_out]):
        reproject_raster_to_utm(raw_ghi, ghi_utm_out, epsg_code=32718)
    else:
        print(f"El raster GHI ya está actualizado: {ghi_utm_out}")

    dataset_out = results['dataset_ml']
    model_out = results.get('model_rf')

    entradas_dataset = [ruta_config, raw_ghi, ghi_utm_out, slope_out, dem_out]
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
            'ghi_32718': ghi_utm_out,
            'slope': slope_out,
            'dem_32718': dem_out
        }

        positivas, pool_negativos = generar_dataset_muestras(vectores, rutas_rasters, config['criterios'])
        entrenar_modelo_rf(
            positivas=positivas,
            pool_negativos=pool_negativos,
            ratio=config['ml_params']['ratio_negativos'],
            out_shp=dataset_out,
            random_state=config['ml_params']['random_state'],
            out_model_path=model_out,
            n_estimators=config['ml_params']['n_estimators']
        )

    print("Pipeline ejecutado correctamente.")


if __name__ == "__main__":
    main()
