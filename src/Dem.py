import os
import glob
import pyproj
import numpy as np
import surtgis
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Force Rasterio/PROJ to use the bundled PROJ data directory,
# instead of an incompatible PostGIS/PostgreSQL proj.db.
proj_data_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_LIB"] = proj_data_dir
os.environ["PROJ_DATA"] = proj_data_dir


"""
Tratamiendo de DEM
"""
# Carpetas de DEM
carpetas_dem = ['Norte_Region_II_DEM', 'Norte_Region_III_DEM']

print("Deduplicando y uniendo archivos DEM...")
archivos_hgt_unicos = {}

for carpeta in carpetas_dem:
    rutas_hgt = glob.glob(os.path.join(carpeta, '*.hgt'))
    for ruta in rutas_hgt:
        nombre_archivo = os.path.basename(ruta)
        if nombre_archivo not in archivos_hgt_unicos:
            archivos_hgt_unicos[nombre_archivo] = ruta

rutas_finales_hgt = list(archivos_hgt_unicos.values())
archivos_abiertos = [rasterio.open(fp) for fp in rutas_finales_hgt]

mosaic, out_trans = merge(archivos_abiertos)
out_meta = archivos_abiertos[0].meta.copy()

# El mosaico está en coordenadas geográficas (lat/lon)
# Reprojectamos a UTM EPSG:32718
src_crs = rasterio.crs.CRS.from_proj4("+proj=latlong +datum=WGS84 +no_defs")
dst_crs = rasterio.crs.CRS.from_proj4("+proj=utm +zone=18 +south +datum=WGS84 +units=m +no_defs")

# Guardar mosaic en archivo temporal en coordenadas geográficas
temp_geo_tif = "dem_norte_temp_geo.tif"
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "crs": src_crs
})

with rasterio.open(temp_geo_tif, "w", **out_meta) as dest:
    dest.write(mosaic)

# Reprojectamos de geográfico a UTM
with rasterio.open(temp_geo_tif) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "dtype": rasterio.int16
    })

    ruta_dem_unido = "dem_norte_32718.tif"
    with rasterio.open(ruta_dem_unido, "w", **out_meta) as dst:
        for i in range(1, src.count + 1):
            reproject(
                rasterio.band(src, i),
                rasterio.band(dst, i),
                resampling=Resampling.bilinear)

# Limpiar archivo temporal
os.remove(temp_geo_tif)

for f in archivos_abiertos:
    f.close()

print(f"Archivo DEM unido guardado como: {ruta_dem_unido}")


# --- PASO 2: PROCESAMIENTO TOPOGRÁFICO CON SURTGIS ---
print("Calculando Slope y Aspect con SurtGIS...")
with rasterio.open("dem_norte_32718.tif") as src:
    dem = src.read(1).astype('float64')
    meta = src.meta.copy()
    cell_size = src.res[0]
    nodata_value = src.nodata if src.nodata is not None else -9999.0

# Mask nodata values so SurtGIS does not compute invalid slopes/aspects
if nodata_value is not None:
    dem = np.where(dem == nodata_value, np.nan, dem)

# Compute slope and aspect arrays
slope_array = surtgis.slope(dem, cell_size=cell_size, units='degrees')
aspect_array = surtgis.aspect_degrees(dem, cell_size=cell_size)

# Prepare output metadata
meta.update(dtype=rasterio.float32, count=1, nodata=nodata_value)

# Save slope raster
with rasterio.open("slope_norte.tif", "w", **meta) as dst:
    slope_out = np.where(np.isnan(slope_array), nodata_value, slope_array).astype('float32')
    dst.write(slope_out, 1)

valid_slope = slope_out[slope_out != nodata_value]
if valid_slope.size:
    print('Slope output written:', 'min=', float(valid_slope.min()), 'max=', float(valid_slope.max()))
else:
    print('Slope output written but contains no valid values.')

# Save aspect raster
with rasterio.open("aspect_norte.tif", "w", **meta) as dst:
    aspect_out = np.where(np.isnan(aspect_array), nodata_value, aspect_array).astype('float32')
    dst.write(aspect_out, 1)

valid_aspect = aspect_out[aspect_out != nodata_value]
if valid_aspect.size:
    print('Aspect output written:', 'min=', float(valid_aspect.min()), 'max=', float(valid_aspect.max()))
else:
    print('Aspect output written but contains no valid values.')