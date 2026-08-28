import os
import sys
import yaml
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

# Ajuste de rutas para importar módulos locales. Va ANTES de los imports de src/ y scripts/:
# al invocar el script directamente (python scripts/profiles.py) sys.path[0] es scripts/, no
# la raíz del repo, así que sin esto los paquetes locales no resuelven.
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.explainability_spatial import zona_de_config
from src.ahp import pesos_perfil  # AHP formal con Ratio de Consistencia (hallazgo 3.3)
# Grilla de referencia = DEM (no GHI ~1 km), igual que generate_suitability_map.py
# (hallazgo 2.1, opción B): mismo criterio para que los mapas AHP sean comparables con el RF.
from scripts.generate_suitability_map import construir_grilla_referencia

# --- FUNCIONES GEOESPACIALES EN MEMORIA ---

def read_and_reproject_to_grid(src_path, dst_crs, dst_shape, dst_transform, nodata_val=np.nan,
                               resampling=Resampling.bilinear):
    # `resampling` debe ser Resampling.nearest para variables circulares como 'aspect'
    # (no se pueden interpolar linealmente a través de la discontinuidad 0°/360°).
    with rasterio.open(src_path) as src:
        destination = np.empty(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=destination,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=resampling, src_nodata=src.nodata, dst_nodata=nodata_val
        )
        return destination

def get_rasterized_mask(gdf, dst_shape, dst_transform, fill=0, default_value=1):
    if gdf is None or len(gdf) == 0:
        return np.full(dst_shape, fill, dtype=np.uint8)
    gdf_valid = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if len(gdf_valid) == 0:
        return np.full(dst_shape, fill, dtype=np.uint8)
    return rasterize(
        shapes=gdf_valid.geometry, out_shape=dst_shape,
        transform=dst_transform, fill=fill, default_value=default_value, all_touched=True
    )

def get_distance_raster_from_vector(gdf, dst_shape, dst_transform, pixel_size_meters):
    if gdf is None or len(gdf) == 0:
        return np.full(dst_shape, np.nan, dtype=np.float32)
    mask = get_rasterized_mask(gdf, dst_shape, dst_transform, fill=0, default_value=1)
    if not np.any(mask == 1):
        return np.full(dst_shape, np.nan, dtype=np.float32)
    inverted = (mask == 0).astype(np.uint8)
    return distance_transform_edt(inverted) * pixel_size_meters

def normalizar_in_memory(arr, valid_mask, inverse=False):
    """Normaliza un arreglo numpy a escala [0, 1] solo en los píxeles válidos."""
    out = np.zeros_like(arr, dtype=np.float32)
    valid_data = arr[valid_mask]
    if len(valid_data) == 0: return out
    
    vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
    if vmax - vmin == 0:
        out[valid_mask] = 0.0
    else:
        if inverse: out[valid_mask] = (vmax - valid_data) / (vmax - vmin) # Costo: menor es mejor
        else: out[valid_mask] = (valid_data - vmin) / (vmax - vmin)       # Beneficio: mayor es mejor
    return out

# --- LÓGICA PRINCIPAL DE PERFILES ---

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Perfiles de inversión AHP/WLC')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--zona', default=None,
                        help='Clave de config.yaml con otra zona (p. ej. "transferibilidad"). '
                             'Sin este flag procesa la zona de entrenamiento y escribe en '
                             'data/results, como siempre.')
    args = parser.parse_args()

    print("=== INICIANDO GENERACIÓN DE PERFILES DE INVERSIÓN (AHP) ===")

    # 1. Cargar configuración de vectores y pesos
    ruta_config = args.config if os.path.isabs(args.config) else os.path.join(directorio_raiz, args.config)
    with open(ruta_config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    paths_raw = config['paths']['raw']

    # Zona a procesar. Los pesos AHP no dependen de la región —son juicios de experto sobre
    # los criterios—, así que la única diferencia es sobre qué terreno se evalúan.
    
    zona = zona_de_config(config, args.zona or 'zona_estudio')
    bloque_zona = (config.get(args.zona) or {}) if args.zona else {}
    dir_salida = os.path.join(
        directorio_raiz, bloque_zona.get('dir_resultados', 'data/results'))
    os.makedirs(dir_salida, exist_ok=True)
    perfiles_config = config.get('perfiles_inversion', {})
    ahp_config = config.get('ahp_perfiles', {})  # matrices AHP por perfil (opcional)

    if not perfiles_config:
        print("ERROR: No se encontró la sección 'perfiles_inversion' en config.yaml.")
        return

    # 2. Definir rutas explícitas a los rasters procesados
    dir_processed = os.path.join(directorio_raiz, 'data', 'processed')
    ghi_path    = os.path.join(dir_processed, 'GHI_32719.tif')
    dem_path    = os.path.join(directorio_raiz, zona['dem'])
    slope_path  = os.path.join(directorio_raiz, zona['slope'])
    aspect_path = os.path.join(directorio_raiz, zona['aspect'])

    # Verificar existencia
    for path in [ghi_path, dem_path, slope_path, aspect_path]:
        if not os.path.exists(path):
            print(f"ERROR: No se encontró el raster {path}")
            return

    # 3. Grilla base = DEM (resolución fina), NO el GHI (~1 km). El GHI se reproyecta/alinea
    # a esta grilla más abajo, para que los mapas AHP de perfiles ya no queden ~10x menos
    # resueltos que el mapa RF con el que se comparan en src/consenso_perfiles.py.
    #
    # OJO: NO se usa la misma resolución que 'salida_mapa.resolucion_m' (100 m, la del mapa
    # RF). A esa resolución, este script mantiene ~15 arrays completos en memoria a la vez
    # (8 variables crudas + 7 normalizadas) SIN el manejo de memoria de
    # generate_suitability_map.py (que aplana a valid_mask de inmediato) — probó matar el
    # proceso por falta de RAM. 'perfiles_inversion.resolucion_m' (default 500 m, mismo
    # criterio que 'shap_espacial.resolucion_m' para el mismo tipo de costo) controla esto
    # de forma independiente; sigue siendo ~2x más fino que el GHI original.
    resolucion_m = bloque_zona.get('resolucion_m', perfiles_config.get('resolucion_m', 500))
    print(f"Cargando grilla base: {os.path.basename(dem_path)}")
    crs, transform, width, height = construir_grilla_referencia(dem_path, resolucion_m)
    grid_shape = (height, width)
    meta_base = {
        'driver': 'GTiff', 'dtype': rasterio.float32, 'count': 1,
        'crs': crs, 'transform': transform, 'width': width, 'height': height,
        'nodata': -9999.0,
    }
    ghi_data = read_and_reproject_to_grid(ghi_path, crs, grid_shape, transform)
    ghi_mask_valid = ~np.isnan(ghi_data)

    # 4. Preparar variables topográficas
    print("Alineando variables topográficas...")
    elev_data = read_and_reproject_to_grid(dem_path, crs, grid_shape, transform)
    slope_data = read_and_reproject_to_grid(slope_path, crs, grid_shape, transform)
    # 'aspect' es circular: NEAREST evita interpolar a través de 0°/360°.
    aspect_data = read_and_reproject_to_grid(aspect_path, crs, grid_shape, transform,
                                             resampling=Resampling.nearest)

    with np.errstate(invalid='ignore'):
        northness_data = np.cos(np.radians(aspect_data))

    # 5. Calcular distancias euclidianas a infraestructura
    print("Calculando distancias a infraestructura en memoria...")
    pixel_size = transform[0]
    
    lineas_path = os.path.join(directorio_raiz, paths_raw['vectores']['lineas'])
    lineas_gdf = gpd.read_file(lineas_path).to_crs(crs)
    dist_t_data = get_distance_raster_from_vector(lineas_gdf, grid_shape, transform, pixel_size)

    almacen_path = paths_raw['vectores'].get('almacenamiento')
    if almacen_path:
        almacen_path = os.path.join(directorio_raiz, almacen_path)
    almacen_gdf = gpd.read_file(almacen_path).to_crs(crs) if almacen_path and os.path.exists(almacen_path) else None
    dist_a_data = get_distance_raster_from_vector(almacen_gdf, grid_shape, transform, pixel_size)

    subs_path = paths_raw['vectores'].get('subestaciones')
    if subs_path:
        subs_path = os.path.join(directorio_raiz, subs_path)
    subs_gdf = gpd.read_file(subs_path).to_crs(crs) if subs_path and os.path.exists(subs_path) else None
    dist_s_data = get_distance_raster_from_vector(subs_gdf, grid_shape, transform, pixel_size)

    # 6. Máscara de validación (Regiones)
    regiones_path = os.path.join(directorio_raiz, paths_raw['vectores']['regiones'])
    regiones_gdf = gpd.read_file(regiones_path)
    regiones_norte = regiones_gdf[regiones_gdf['REGION'].isin(zona['regiones'])].to_crs(crs)
    region_mask = get_rasterized_mask(regiones_norte, grid_shape, transform)

    valid_mask = (
        (region_mask == 1) & ghi_mask_valid &
        (~np.isnan(elev_data)) & (~np.isnan(slope_data)) & (~np.isnan(northness_data)) &
        (~np.isnan(dist_t_data)) & (~np.isnan(dist_a_data)) & (~np.isnan(dist_s_data))
    )

    if np.sum(valid_mask) == 0:
        print("ERROR: No hay píxeles válidos para evaluar los perfiles.")
        return

    # 7. Normalización Matemática [0, 1]
    print("\nNormalizando variables para el cálculo AHP...")
    norm_ghi = normalizar_in_memory(ghi_data, valid_mask, inverse=False)
    norm_slope = normalizar_in_memory(slope_data, valid_mask, inverse=True)
    norm_elev = normalizar_in_memory(elev_data, valid_mask, inverse=True)
    norm_northness = normalizar_in_memory(northness_data, valid_mask, inverse=False)
    norm_dist_t = normalizar_in_memory(dist_t_data, valid_mask, inverse=True)
    norm_dist_a = normalizar_in_memory(dist_a_data, valid_mask, inverse=True)
    norm_dist_s = normalizar_in_memory(dist_s_data, valid_mask, inverse=True)

    # 8. Cálculo por Perfiles (Combinación Lineal Ponderada)
    meta_base.update(dtype=rasterio.float32, nodata=-9999.0, count=1)
    
    for nombre_perfil in ['conservador', 'agresivo']:
        # Pesos derivados por AHP (validados con Ratio de Consistencia) si hay matriz
        # para el perfil; si no, se usan los pesos directos de 'perfiles_inversion'.
        pesos = pesos_perfil(nombre_perfil, ahp_config) or perfiles_config.get(nombre_perfil)
        if not pesos: continue

        print(f"\nCalculando perfil: {nombre_perfil.capitalize()}...")
        mapa_wlc = np.full(grid_shape, -9999.0, dtype=np.float32)
        
        # Álgebra de mapas
        aptitud = (
            norm_ghi[valid_mask] * pesos.get('ghi', 0.0) +
            norm_slope[valid_mask] * pesos.get('slope', 0.0) +
            norm_elev[valid_mask] * pesos.get('elev', 0.0) +
            norm_northness[valid_mask] * pesos.get('northness', 0.0) +
            norm_dist_t[valid_mask] * pesos.get('dist_transmision', 0.0) +
            norm_dist_a[valid_mask] * pesos.get('dist_almacen', 0.0) +
            norm_dist_s[valid_mask] * pesos.get('dist_subestaciones', 0.0)
        )
        mapa_wlc[valid_mask] = aptitud

        # 9. Aplicar Zonas de Exclusión
        exclusion_paths = {
            'snap': paths_raw['vectores'].get('snap'),
            'lagos': paths_raw['vectores'].get('masas_lacustres'),
            'poblaciones': paths_raw['vectores'].get('areas_pobladas')
        }
        for key, path in exclusion_paths.items():
            if path:
                full_path = os.path.join(directorio_raiz, path)
                if os.path.exists(full_path):
                    gdf_excl = gpd.read_file(full_path).to_crs(crs)
                    mask_excl = get_rasterized_mask(gdf_excl, grid_shape, transform)
                    mapa_wlc[(mask_excl == 1) & (mapa_wlc != -9999.0)] = 0.0

        # Guardar en disco
        out_path = os.path.join(dir_salida, f'aptitud_{nombre_perfil}.tif')
        with rasterio.open(out_path, 'w', **meta_base) as dst:
            dst.write(mapa_wlc, 1)
        print(f" -> Guardado exitosamente: {out_path}")

    print("\n=== GENERACIÓN DE PERFILES COMPLETADA ===")

if __name__ == "__main__":
    main()