import os
import importlib.util
import glob
import pyproj
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import surtgis
import geopandas as gpd


def procesar_dem(carpetas_dem: list, out_slope_path: str, out_aspect_path: str, out_dem_path: str | None = None, resolucion_m: float | None = None) -> None:
    print("Deduplicando y uniendo archivos DEM...")
    archivos_hgt_unicos = {}
    for carpeta in carpetas_dem:
        for ruta in glob.glob(os.path.join(carpeta, '*.hgt')):
            nombre = os.path.basename(ruta)
            if nombre not in archivos_hgt_unicos:
                archivos_hgt_unicos[nombre] = ruta

    rutas_finales_hgt = list(archivos_hgt_unicos.values())
    if not rutas_finales_hgt:
        raise ValueError("No se encontraron archivos .hgt en las carpetas proporcionadas.")
    print(f"Se encontraron {len(rutas_finales_hgt)} archivos DEM únicos")

    salidas_derivados = [out_slope_path, out_aspect_path]
    salidas_dem = [out_dem_path] if out_dem_path else []

    # Nivel 1: todos los archivos ya existen → omitir
    if all(os.path.exists(p) for p in salidas_derivados + salidas_dem):
        print("Los productos DEM ya existen en procesados. No se requiere reprocesar.")
        return

    # Nivel 2: DEM reproyectado ya existe → cargar y generar solo slope/aspect
    dem_en_cache = out_dem_path and os.path.exists(out_dem_path)
    if dem_en_cache:
        print(f"DEM en caché ({out_dem_path}). Cargando sin re-fusionar los .hgt...")
        with rasterio.open(out_dem_path) as src:
            nodata_value = src.nodata if src.nodata is not None else -9999.0
            dst_transform = src.transform
            dst_crs = src.crs
            dem_utm = src.read(1).astype('float32')
        dem_utm[dem_utm == nodata_value] = np.nan  # in-place: conserva float32
    else:
        # Nivel 3: proceso completo
        # Paso 1: merge en EPSG:4326
        print("Iniciando fusión de archivos DEM...")
        archivos_abiertos = [rasterio.open(fp) for fp in rutas_finales_hgt]
        try:
            dem_geo, transform_geo = merge(archivos_abiertos)
            src_crs = archivos_abiertos[0].crs
            nodata_value = archivos_abiertos[0].nodata if archivos_abiertos[0].nodata is not None else -9999.0
        finally:
            for f in archivos_abiertos:
                f.close()
        print(f"DEM fusionado con dimensiones: {dem_geo.shape}")

        # float32 en vez de float64: la mitad de memoria. El DEM se guarda como float32 de
        # todos modos, así que la salida no cambia. Mantenemos el nodata como centinela
        # (reproject lo maneja con src_nodata) para evitar copias nan<->nodata innecesarias.
        dem_geo = dem_geo[0].astype('float32')

        # Paso 2: reproyectar a UTM EPSG:32719 en memoria
        print("Reproyectando DEM fusionado a UTM EPSG:32719...")
        dst_crs = rasterio.crs.CRS.from_epsg(32719)
        h, w = dem_geo.shape
        bounds = rasterio.transform.array_bounds(h, w, transform_geo)
        # resolucion_m fuerza la resolución de destino en metros; None = nativa (~30 m).
        # Subirla reduce el uso de memoria de forma cuadrática (área > 200.000 km²).
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, w, h, *bounds,
            resolution=resolucion_m if resolucion_m else None,
        )
        dem_utm = np.full((dst_height, dst_width), nodata_value, dtype='float32')
        reproject(
            source=dem_geo,
            src_transform=transform_geo,
            src_crs=src_crs,
            destination=dem_utm,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata_value,
            dst_nodata=nodata_value,
        )
        del dem_geo  # liberar el DEM geográfico antes de derivar slope/aspect
        dem_utm[dem_utm == nodata_value] = np.nan  # in-place: conserva float32 sin copiar

        # Paso 3: guardar DEM UTM para usarlo como caché en futuras ejecuciones
        if out_dem_path:
            parent = os.path.dirname(out_dem_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            print(f"Guardando DEM reproyectado en {out_dem_path}...")
            meta_dem = {
                'driver': 'GTiff', 'dtype': rasterio.float32, 'count': 1,
                'nodata': nodata_value, 'transform': dst_transform, 'crs': dst_crs,
                'width': dst_width, 'height': dst_height,
            }
            with rasterio.open(out_dem_path, 'w', **meta_dem) as dst:
                dst.write(np.where(np.isnan(dem_utm), nodata_value, dem_utm).astype('float32'), 1)

    # Paso 4 : calcular slope y aspect sobre el DEM UTM.
    # Patrón de memoria: los CÁLCULOS se hacen en float64 (surtgis, Rust/PyO3, exige un
    # arreglo float64 C-contiguo), pero cada arreglo se libera apenas deja de necesitarse
    # (calcular → guardar → del). El nodata se rellena in-place para no crear la copia
    # completa que generaba np.where. Con esto el pico de la fase pasa de ~4.5 arreglos
    # simultáneos a ~2 (a resolución nativa: ~28 GB → ~15 GB).
    cell_size = abs(dst_transform[0])
    print(f"Tamaño de celda UTM: {cell_size:.2f} m")
    dem_utm = np.ascontiguousarray(dem_utm, dtype=np.float64)  # libera el buffer float32 al reasignar

    meta_derivados = {
        'driver': 'GTiff', 'dtype': rasterio.float32, 'count': 1,
        'nodata': nodata_value, 'transform': dst_transform, 'crs': dst_crs,
        'width': dem_utm.shape[1], 'height': dem_utm.shape[0],
    }
    for path in salidas_derivados:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    print("Calculando slope...")
    slope_array = surtgis.slope(dem_utm, cell_size=cell_size, units='degrees')
    slope_array[np.isnan(slope_array)] = nodata_value
    print(f"Guardando slope en {out_slope_path}...")
    with rasterio.open(out_slope_path, 'w', **meta_derivados) as dst:
        dst.write(slope_array.astype('float32'), 1)
    del slope_array  # liberar antes de calcular aspect

    print("Calculando aspect...")
    aspect_array = surtgis.aspect_degrees(dem_utm, cell_size=cell_size)
    del dem_utm  # el DEM ya no se necesita
    aspect_array[np.isnan(aspect_array)] = nodata_value
    print(f"Guardando aspect en {out_aspect_path}...")
    with rasterio.open(out_aspect_path, 'w', **meta_derivados) as dst:
        dst.write(aspect_array.astype('float32'), 1)
    del aspect_array

    generados = [out_slope_path, out_aspect_path]
    if out_dem_path and not dem_en_cache:
        generados.append(out_dem_path)
    print("Archivos generados exitosamente:\n" + "\n".join(f" - {p}" for p in generados))


def cargar_capas_vectoriales(diccionario_rutas: dict) -> dict:
    """Carga el diccionario de rutas desde config.yaml y devuelve GeoDataFrames."""
    capas = {}
    for nombre_capa, ruta in diccionario_rutas.items():
        print(f"Cargando {nombre_capa}...")
        capas[nombre_capa] = gpd.read_file(ruta)
    return capas


def reproject_raster_to_utm(src_path: str, dst_path: str, epsg_code: int = 32719) -> None:
    """Reproyecta un raster al EPSG indicado. No hace nada si el archivo de salida ya existe."""
    if os.path.exists(dst_path):
        print(f"Raster ya existe en procesados, se omite: {dst_path}")
        return
    
    print(f"Reproyectando {src_path} a EPSG:{epsg_code}...")
    parent = os.path.dirname(dst_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with rasterio.open(src_path) as src:
        dst_crs = rasterio.crs.CRS.from_epsg(epsg_code)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(crs=dst_crs, transform=transform, width=width, height=height)

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    rasterio.band(src, i),
                    rasterio.band(dst, i),
                    resampling=Resampling.bilinear,
                )
    print(f"Reproyección exitosa: {dst_path}")
