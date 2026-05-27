import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import rasterio
from rasterio.windows import Window

def _sample_raster_value(src, x, y):
    """Función auxiliar (privada) para extraer valor de raster con fallback local."""
    nodata = src.nodata if src.nodata is not None else -9999.0
    try:
        for val in src.sample([(x, y)]):
            v = val[0]
    except Exception:
        return np.nan
    if v == nodata or np.isnan(v):
        row, col = src.index(x, y)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return np.nan
        row0 = max(row - 1, 0)
        col0 = max(col - 1, 0)
        row1 = min(row + 2, src.height)
        col1 = min(col + 2, src.width)
        window = Window(col0, row0, col1 - col0, row1 - row0)
        arr = src.read(1, window=window)
        valid = arr[arr != nodata]
        if valid.size > 0:
            return float(np.nanmean(valid))
        return np.nan
    return float(v)


def extraer_valores_puntos(gdf, raster_path, col_name):
    """Extrae valores de un raster para un GeoDataFrame de puntos."""
    with rasterio.open(raster_path) as src:
        coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        valores = []
        for x, y in coords:
            valores.append(_sample_raster_value(src, x, y))
        gdf[col_name] = valores
    return gdf


def fill_missing_from_raster(gdf, raster_path, col_name):
    """Rellena valores NaN en gdf desde el raster usando ventana local."""
    with rasterio.open(raster_path) as src:
        for idx in gdf.index[gdf[col_name].isna()]:
            geom = gdf.loc[idx, 'geometry']
            if geom is None or geom.is_empty:
                continue
            x, y = geom.x, geom.y
            gdf.at[idx, col_name] = _sample_raster_value(src, x, y)
    return gdf


def generar_dataset_muestras(vectores: dict, rutas_rasters: dict, criterios: dict) -> tuple:
    """
    Sigue la lógica de Gap1.py: 
    1. Genera positivas desde plantas existentes
    2. Genera candidatos negativos aleatorios
    3. Limpia NaN en GHI y selecciona pool grande (Hard Negative Mining)
    4. Muestrea todos los rasters en positivas + pool
    5. Rellena NaN en positivas y filtra negativos por criterios AHP
    Retorna (gdf_positivas, gdf_negativas_pool).
    """
    print("Iniciando muestreo de datos espaciales...")
    
    # 1. FILTRADO POR REGIÓN
    codigos_norte = [2, 3, '2', '3', '02', '03']
    regiones_norte = vectores['regiones'][vectores['regiones']['REGION'].isin(['Antofagasta', 'Atacama'])].to_crs(epsg=32718)
    fotos_norte = vectores['fotovoltaicas'][vectores['fotovoltaicas']['REGION'].isin(codigos_norte)].to_crs(epsg=32718)
    lineas_norte = vectores['lineas'][vectores['lineas']['REGION'].isin(codigos_norte)].to_crs(epsg=32718)
    
    # 2. POSITIVAS: centroides de plantas existentes
    positivas = gpd.GeoDataFrame(
        {'id_muestra': range(len(fotos_norte)), 'clase': 1}, 
        geometry=fotos_norte.geometry.centroid, 
        crs="EPSG:32718"
    )

    # 3. NEGATIVAS ALEATORIAS dentro de las regiones
    bounds = regiones_norte.total_bounds
    puntos_random = []
    n_neg_deseados = len(positivas) * 2 
    
    while len(puntos_random) < n_neg_deseados * 5:
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])
        pto = Point(x, y)
        if regiones_norte.contains(pto).any():
            puntos_random.append(pto)
            
    candidatos_neg = gpd.GeoDataFrame(geometry=puntos_random, crs="EPSG:32718")
    
    # 4. EXCLUSIONES BÁSICAS
    buffer_plantas = positivas.geometry.buffer(5000).union_all()
    candidatos_neg = candidatos_neg[~candidatos_neg.geometry.intersects(buffer_plantas)]
    
    # 5. HARD NEGATIVE MINING: Primero extrae GHI de candidatos
    print(f"  Extrayendo GHI de {len(candidatos_neg)} candidatos negativos...")
    candidatos_neg = extraer_valores_puntos(candidatos_neg, rutas_rasters['ghi_32718'], 'ghi')
    
    # Limpia NaN en GHI ANTES de seleccionar pool
    candidatos_neg = candidatos_neg.dropna(subset=['ghi'])
    print(f"  Candidatos tras limpiar GHI NaN: {len(candidatos_neg)}")
    
    # Selecciona pool grande: top por GHI (Hard Negative Mining)
    if len(candidatos_neg) > 0:
        n_pool = min(len(candidatos_neg), max(len(positivas) * 10 * 3, 1000))
        candidatos_top = candidatos_neg.sort_values(by='ghi', ascending=False).head(n_pool).copy()
        candidatos_top['clase'] = 0
        print(f"  Pool seleccionado (top GHI): {len(candidatos_top)}")
    else:
        raise ValueError("No hay candidatos negativos con GHI válido. Revisa los rasters.")
    
    # 6. MUESTREO CONJUNTO: Positivas + candidatos_top
    print(f"  Muestreando rasters para {len(positivas)} positivas + {len(candidatos_top)} candidatos...")
    muestras_pre = pd.concat(
        [positivas[['clase', 'geometry']], candidatos_top[['clase', 'geometry']]], 
        ignore_index=True
    )
    muestras_pre = gpd.GeoDataFrame(muestras_pre, geometry='geometry', crs='EPSG:32718')
    
    # Muestrea todos los rasters en conjunto
    muestras_pre = extraer_valores_puntos(muestras_pre, rutas_rasters['ghi_32718'], 'ghi')
    muestras_pre = extraer_valores_puntos(muestras_pre, rutas_rasters['slope'], 'slope')
    muestras_pre = extraer_valores_puntos(muestras_pre, rutas_rasters['dem_32718'], 'elev')
    
    # Distancia a transmisión
    if len(lineas_norte) > 0:
        muestras_pre['dist_transmision'] = muestras_pre.geometry.apply(
            lambda g: lineas_norte.geometry.distance(g).min()
        )
    else:
        muestras_pre['dist_transmision'] = 1e6
    
    # 7. SEPARAR POSITIVAS Y NEGATIVOS
    positivas = muestras_pre[muestras_pre['clase'] == 1].copy()
    negativos_pre = muestras_pre[muestras_pre['clase'] == 0].copy()
    
    # 8. RELLENAR NaN EN POSITIVAS (estrategia: ventana local + media global)
    print(f"  Rellenando NaN en positivas...")
    positivas = fill_missing_from_raster(positivas, rutas_rasters['ghi_32718'], 'ghi')
    positivas = fill_missing_from_raster(positivas, rutas_rasters['slope'], 'slope')
    positivas = fill_missing_from_raster(positivas, rutas_rasters['dem_32718'], 'elev')
    
    # Si aún faltan valores, imputar con media global del raster
    for col, raster_path in [('ghi', rutas_rasters['ghi_32718']), 
                              ('slope', rutas_rasters['slope']), 
                              ('elev', rutas_rasters['dem_32718'])]:
        if positivas[col].isna().any():
            with rasterio.open(raster_path) as src:
                nodata = src.nodata if src.nodata is not None else -9999.0
                band = src.read(1)
                mean_val = float(np.nanmean(band[band != nodata]))
            positivas[col] = positivas[col].fillna(mean_val)
    
    # 9. LIMPIAR NEGATIVOS: eliminar NaN
    negativos_pre = negativos_pre.dropna(subset=['ghi', 'slope', 'elev'])
    print(f"  Negativos tras limpiar NaN: {len(negativos_pre)}")
    
    # 10. APLICAR CRITERIOS AHP
    pool_negativos = negativos_pre[
        (negativos_pre['ghi'] >= criterios['ghi_min']) &
        (negativos_pre['slope'] <= criterios['slope_max']) &
        (negativos_pre['elev'] <= criterios['elev_max'])
    ].copy()
    pool_negativos['clase'] = 0
    
    print(f"\n=== RESUMEN MUESTREO ===")
    print(f"Muestras positivas generadas: {len(positivas)}")
    print(f"Pool de negativos elegibles: {len(pool_negativos)}")
    
    if len(pool_negativos) > 0:
        print(f"  GHI negat: min={pool_negativos['ghi'].min():.1f}, max={pool_negativos['ghi'].max():.1f}, mean={pool_negativos['ghi'].mean():.1f}")
        print(f"  Slope negat: min={pool_negativos['slope'].min():.1f}, max={pool_negativos['slope'].max():.1f}, mean={pool_negativos['slope'].mean():.1f}")
        print(f"  Elev negat: min={pool_negativos['elev'].min():.0f}, max={pool_negativos['elev'].max():.0f}, mean={pool_negativos['elev'].mean():.0f}")
    
    return positivas, pool_negativos