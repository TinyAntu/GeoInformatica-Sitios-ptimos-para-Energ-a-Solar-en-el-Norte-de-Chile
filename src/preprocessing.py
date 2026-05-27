import os
import glob
import pyproj
import numpy as np
import rasterio
from rasterio.merge import merge
import surtgis
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Mantenemos la corrección de pyproj que tenías, es una buena práctica para evitar errores de BD
proj_data_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_LIB"] = proj_data_dir
os.environ["PROJ_DATA"] = proj_data_dir

def _archivos_existentes(paths: list) -> list:
    return [path for path in paths if os.path.exists(path)]


def _esta_actualizado(paths_entrada: list, paths_salida: list) -> bool:
    outputs_existentes = _archivos_existentes(paths_salida)
    return len(outputs_existentes) == len(paths_salida)


def _save_dem_utm(dem_array: np.ndarray, transform, src_crs, out_path: str, nodata_value: float, epsg_code: int = 32718) -> None:
    print(f"Guardando DEM reproyectado a {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    zone = epsg_code % 100
    is_south = 32700 <= epsg_code < 32900
    proj4 = f"+proj=utm +zone={zone} {'+south' if is_south else ''} +datum=WGS84 +units=m +no_defs"
    dst_crs = rasterio.crs.CRS.from_proj4(proj4)
    height, width = dem_array.shape
    bounds = rasterio.transform.array_bounds(height, width, transform)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *bounds
    )

    dst_meta = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'count': 1,
        'nodata': nodata_value,
        'transform': dst_transform,
        'crs': dst_crs,
        'width': dst_width,
        'height': dst_height
    }

    with rasterio.open(out_path, 'w', **dst_meta) as dst:
        dst_arr = np.empty((dst_height, dst_width), dtype='float32')
        reproject(
            source=dem_array,
            src_transform=transform,
            src_crs=src_crs,
            destination=dst_arr,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata_value,
            dst_nodata=nodata_value
        )
        dst.write(dst_arr, 1)


def procesar_dem(carpetas_dem: list, out_slope_path: str, out_aspect_path: str, out_dem_path: str | None = None) -> None:
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
    print(f"Se encontraron {len(rutas_finales_hgt)} archivos DEM únicos")

    salidas_dem = [out_slope_path, out_aspect_path]
    if out_dem_path:
        salidas_dem.append(out_dem_path)

    if _esta_actualizado(rutas_finales_hgt, salidas_dem):
        print("Los productos DEM ya están actualizados. No se requiere reprocesar.")
        return

    archivos_abiertos = [rasterio.open(fp) for fp in rutas_finales_hgt]

    if not archivos_abiertos:
        raise ValueError("No se encontraron archivos .hgt en las carpetas proporcionadas.")

    # 2. Merge de los DEM
    print("Iniciando fusión de archivos DEM...")
    dem, transform = merge(archivos_abiertos)
    print(f"DEM fusionado con dimensiones: {dem.shape}")
    dem = dem[0].astype('float64')

    # Extraer metadata del primer archivo para usarla de base
    src = archivos_abiertos[0]
    cell_size = src.res[0]
    nodata_value = src.nodata if src.nodata is not None else -9999.0

    # Cerrar archivos para liberar memoria
    for f in archivos_abiertos:
        f.close()

    # 3. Procesamiento (Cálculos matemáticos)
    print("Procesando valores nodata...")
    if nodata_value is not None:
        dem = np.where(dem == nodata_value, np.nan, dem)

    print("Calculando slope...")
    slope_array = surtgis.slope(dem, cell_size=cell_size, units='degrees')
    print("Calculando aspect...")
    aspect_array = surtgis.aspect_degrees(dem, cell_size=cell_size)

    # Crear metadata limpia para el output (sin restricciones de formato SRTM)
    meta = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'count': 1,
        'nodata': nodata_value,
        'transform': transform,
        'crs': src.crs,
        'height': dem.shape[0],
        'width': dem.shape[1]
    }

    # 4. Guardar resultados
    for path in [out_slope_path, out_aspect_path]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    print(f"Guardando slope en {out_slope_path}...")
    with rasterio.open(out_slope_path, "w", **meta) as dst:
        slope_out = np.where(np.isnan(slope_array), nodata_value, slope_array).astype('float32')
        dst.write(slope_out, 1)
    
    print(f"Guardando aspect en {out_aspect_path}...")
    with rasterio.open(out_aspect_path, "w", **meta) as dst:
        aspect_out = np.where(np.isnan(aspect_array), nodata_value, aspect_array).astype('float32')
        dst.write(aspect_out, 1)

    if out_dem_path:
        _save_dem_utm(dem, transform, src.crs, out_dem_path, nodata_value)

    print(f"Archivos generados exitosamente:\n - {out_slope_path}\n - {out_aspect_path}" + (f"\n - {out_dem_path}" if out_dem_path else ""))

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

def reproject_raster_to_utm(src_path: str, dst_path: str, epsg_code: int = 32718) -> None:
    """Reproyecta un raster a un EPSG específico y lo guarda en disco."""
    print(f"Reproyectando {src_path} a EPSG:{epsg_code}...")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with rasterio.open(src_path) as src:
        # Usamos el string PROJ para evitar problemas de base de datos
        dst_crs = rasterio.crs.CRS.from_proj4(
            f"+proj=utm +zone={epsg_code % 100} +south +datum=WGS84 +units=m +no_defs"
        )
        
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update(crs=dst_crs, transform=transform, width=width, height=height)
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    rasterio.band(src, i),
                    rasterio.band(dst, i),
                    resampling=Resampling.bilinear
                )
    print(f"Reproyección exitosa: {dst_path}")