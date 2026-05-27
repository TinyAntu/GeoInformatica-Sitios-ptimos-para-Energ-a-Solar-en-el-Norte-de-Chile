import os
import glob
import pyproj
import numpy as np
import rasterio
from rasterio.merge import merge
import surtgis
import geopandas as gpd

# Mantenemos la corrección de pyproj que tenías, es una buena práctica para evitar errores de BD
proj_data_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_LIB"] = proj_data_dir
os.environ["PROJ_DATA"] = proj_data_dir

def procesar_dem(carpetas_dem: list, out_slope_path: str, out_aspect_path: str) -> None:
    """
    Deduplica, une archivos DEM (.hgt) y calcula pendiente (slope) y orientación (aspect).
    Cumple con: Sin side-effects ocultos, rutas parametrizadas.
    """
    print("Deduplicando y uniendo archivos DEM...")
    archivos_hgt_unicos = {}
    
    # 1. Búsqueda y deduplicación basada en los parámetros (no variables globales)
    for carpeta in carpetas_dem:
        rutas_hgt = glob.glob(os.path.join(carpeta, '*.hgt'))
        for ruta in rutas_hgt:
            nombre_archivo = os.path.basename(ruta)
            if nombre_archivo not in archivos_hgt_unicos:
                archivos_hgt_unicos[nombre_archivo] = ruta

    rutas_finales_hgt = list(archivos_hgt_unicos.values())
    archivos_abiertos = [rasterio.open(fp) for fp in rutas_finales_hgt]

    if not archivos_abiertos:
        raise ValueError("No se encontraron archivos .hgt en las carpetas proporcionadas.")

    # 2. Merge de los DEM
    dem, transform = merge(archivos_abiertos)
    dem = dem[0].astype('float64')

    # Extraer metadata del primer archivo para usarla de base
    src = archivos_abiertos[0]
    meta = src.meta.copy()
    cell_size = src.res[0]
    nodata_value = src.nodata if src.nodata is not None else -9999.0

    # Cerrar archivos para liberar memoria
    for f in archivos_abiertos:
        f.close()

    # 3. Procesamiento (Cálculos matemáticos)
    if nodata_value is not None:
        dem = np.where(dem == nodata_value, np.nan, dem)

    slope_array = surtgis.slope(dem, cell_size=cell_size, units='degrees')
    aspect_array = surtgis.aspect_degrees(dem, cell_size=cell_size)

    # Actualizar metadata para el output (importante: actualizar el transform y dimensiones del mosaico)
    meta.update(
        dtype=rasterio.float32, 
        count=1, 
        nodata=nodata_value,
        transform=transform,
        height=dem.shape[0],
        width=dem.shape[1]
    )

    # 4. Guardar resultados
    with rasterio.open(out_slope_path, "w", **meta) as dst:
        slope_out = np.where(np.isnan(slope_array), nodata_value, slope_array).astype('float32')
        dst.write(slope_out, 1)
    
    with rasterio.open(out_aspect_path, "w", **meta) as dst:
        aspect_out = np.where(np.isnan(aspect_array), nodata_value, aspect_array).astype('float32')
        dst.write(aspect_out, 1)
        
    print(f"Archivos generados exitosamente:\n - {out_slope_path}\n - {out_aspect_path}")

def cargar_capas_vectoriales(diccionario_rutas: dict) -> dict:
    """
    Recibe el diccionario de rutas desde el config.yaml y devuelve un diccionario
    con los GeoDataFrames cargados.
    """
    capas = {}
    for nombre_capa, ruta in diccionario_rutas.items():
        print(f"Cargando {nombre_capa}...")
        capas[nombre_capa] = gpd.read_file(ruta)
    return capas