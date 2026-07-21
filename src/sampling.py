import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import rasterio
from rasterio.windows import Window


def _sample_raster_value(src, x, y):
    """Extrae valor de raster en (x, y) con fallback a ventana 3x3 si hay nodata."""
    nodata = src.nodata if src.nodata is not None else -9999.0
    try:
        for val in src.sample([(x, y)]):
            v = val[0]
    except Exception:
        return np.nan
    if v == nodata or np.isnan(float(v)):
        row, col = src.index(x, y)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return np.nan
        row0, col0 = max(row - 1, 0), max(col - 1, 0)
        row1, col1 = min(row + 2, src.height), min(col + 2, src.width)
        window = Window(col0, row0, col1 - col0, row1 - row0)
        arr = src.read(1, window=window)
        valid = arr[arr != nodata]
        return float(np.nanmean(valid)) if valid.size > 0 else np.nan
    return float(v)


def extraer_valores_puntos(gdf: gpd.GeoDataFrame, raster_path: str, col_name: str) -> gpd.GeoDataFrame:
    """Extrae valores de un raster para todos los puntos de un GeoDataFrame."""
    with rasterio.open(raster_path) as src:
        coords = list(zip(gdf.geometry.x, gdf.geometry.y))
        gdf[col_name] = [_sample_raster_value(src, x, y) for x, y in coords]
    return gdf


def fill_missing_from_raster(gdf: gpd.GeoDataFrame, raster_path: str, col_name: str) -> gpd.GeoDataFrame:
    """Rellena valores NaN en gdf desde el raster usando ventana local."""
    with rasterio.open(raster_path) as src:
        for idx in gdf.index[gdf[col_name].isna()]:
            geom = gdf.loc[idx, 'geometry']
            if geom is None or geom.is_empty:
                continue
            gdf.at[idx, col_name] = _sample_raster_value(src, geom.x, geom.y)
    return gdf


def generar_dataset_muestras(
    vectores: dict,
    rutas_rasters: dict,
    criterios: dict,
    ratio: int = 3,
    random_state: int = 42,
) -> tuple:
    """
    Genera muestras positivas (plantas existentes) y un pool de negativos geográficamente 
    representativo incluyendo distancias euclidianas continuas a la red de transmisión.
    """
    print("Iniciando muestreo de datos espaciales (Con Variable de Distancia)...")
    rng = np.random.default_rng(random_state)

    # 1. Filtrado por región
    codigos_norte = [2, 3, '2', '3', '02', '03', 'II', 'III', 'Antofagasta', 'Atacama', 'ANTOFAGASTA', 'ATACAMA']
    
    regiones_norte = vectores['regiones'][
        vectores['regiones']['REGION'].isin(['Antofagasta', 'Atacama'])
    ].to_crs(epsg=32719)
    
    fotos_norte = vectores['fotovoltaicas'][
        vectores['fotovoltaicas']['REGION'].astype(str).str.upper().str.strip().isin(
            [str(c).upper().strip() for c in codigos_norte]
        )
    ].to_crs(epsg=32719)
    
    if 'REGION' in vectores['lineas'].columns:
        lineas_norte = vectores['lineas'][
            vectores['lineas']['REGION'].astype(str).str.upper().str.strip().isin(
                [str(c).upper().strip() for c in codigos_norte]
            )
        ].to_crs(epsg=32719)
    else:
        print("\n[ALERTA] La capa de líneas NO tiene una columna 'REGION'. Recortando espacialmente...")
        lineas_norte = gpd.clip(vectores['lineas'], regiones_norte).to_crs(epsg=32719)

    def _clip_spatially(layer, nombre):
        if layer is None or len(layer) == 0:
            return gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719')
        layer_utm = layer.to_crs(epsg=32719)
        if 'REGION' in layer_utm.columns:
            try:
                return layer_utm[
                    layer_utm['REGION'].astype(str).str.upper().str.strip().isin(
                        [str(c).upper().strip() for c in codigos_norte]
                    )
                ]
            except Exception:
                pass
        try:
            return gpd.clip(layer_utm, regiones_norte)
        except Exception as e:
            print(f"  [AVISO] No se pudo recortar '{nombre}' espacialmente: {e}. Usando capa completa.")
            return layer_utm

    almacen_norte = _clip_spatially(vectores.get('almacenamiento', gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719')), 'almacenamiento')
    subestaciones_norte = _clip_spatially(vectores.get('subestaciones', gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719')), 'subestaciones')

    # Limpieza estructural de geometrías nulas o vacías
    regiones_norte = regiones_norte[regiones_norte.geometry.notna() & ~regiones_norte.geometry.is_empty]
    lineas_norte = lineas_norte[lineas_norte.geometry.notna() & ~lineas_norte.geometry.is_empty]
    almacen_norte = almacen_norte[almacen_norte.geometry.notna() & ~almacen_norte.geometry.is_empty]
    subestaciones_norte = subestaciones_norte[subestaciones_norte.geometry.notna() & ~subestaciones_norte.geometry.is_empty]

    fotos_norte = fotos_norte.copy()
    fotos_norte['geometry'] = fotos_norte.geometry.make_valid()
    fotos_norte = fotos_norte[fotos_norte.geometry.notna() & ~fotos_norte.geometry.is_empty]
    fotos_norte = fotos_norte.reset_index(drop=True) # Corrección crucial para evitar desalineación (NaNs)

    print(f"Regiones al norte de Chile: {len(regiones_norte)}")
    print(f"Plantas fotovoltaicas en el norte: {len(fotos_norte)}")
    print(f"Líneas de transmisión en el norte: {len(lineas_norte)}")

    # 2. Positivas: centroides validados
    positivas = gpd.GeoDataFrame(
        {'id_muestra': range(len(fotos_norte)), 'clase': 1},
        geometry=fotos_norte.geometry.centroid,
        crs='EPSG:32719',
    )
    # Filtro de seguridad extremo: asegurar que no existan coordenadas NaN en los centroides
    positivas = positivas[positivas.geometry.apply(lambda g: not (np.isnan(g.x) or np.isnan(g.y)) if g is not None else False)]

    # 3. Candidatos negativos usando criterio de distancia dinámica del config.yaml
    bounds = regiones_norte.total_bounds
    n_neg_deseados = len(positivas) * ratio
    puntos_random = []

    # Los candidatos se generan en toda la región de estudio (sin restringir a un buffer
    # alrededor de las líneas: ese filtro introducía sesgo y quedó descartado). El criterio
    # de distancia máxima a transmisión se aplica más abajo, en el filtro AHP.
    print("  Generando candidatos aleatorios dentro de las regiones de estudio...")
    intentos = 0
    while len(puntos_random) < n_neg_deseados * 6 and intentos < n_neg_deseados * 150:
        intentos += 1
        x = rng.uniform(bounds[0], bounds[2])
        y = rng.uniform(bounds[1], bounds[3])
        pto = Point(x, y)

        if regiones_norte.contains(pto).any():
            puntos_random.append(pto)

    candidatos_neg = gpd.GeoDataFrame(geometry=puntos_random, crs='EPSG:32719')
    print(f"  Se generaron {len(candidatos_neg)} candidatos iniciales.")

    # 4. Exclusión: buffer alrededor de plantas existentes
    buffer_plantas = positivas.geometry.buffer(5000).union_all()
    candidatos_neg = candidatos_neg[~candidatos_neg.geometry.intersects(buffer_plantas)]

    # 4b. Exclusión de restricciones territoriales (poblados/lagos/SNAP)
    claves_exclusion = ['areas_pobladas', 'masas_lacustres', 'snap']
    mascaras = []
    for clave in claves_exclusion:
        if clave not in vectores or len(vectores[clave]) == 0:
            continue
        capa = vectores[clave].to_crs(epsg=32719)
        try:
            capa = gpd.clip(capa, regiones_norte)
        except Exception as e:
            print(f"  [AVISO] No se pudo recortar '{clave}': {e}. Usando capa completa.")
        if len(capa) > 0:
            mascaras.append(capa.geometry.union_all())
      
    if mascaras:
        mascara_total = mascaras[0]
        for m in mascaras[1:]:
            mascara_total = mascara_total.union(m)
        antes = len(candidatos_neg)
        candidatos_neg = candidatos_neg[~candidatos_neg.geometry.intersects(mascara_total)]
        print(f"  Candidatos eliminados por zonas de exclusión: {antes - len(candidatos_neg)}")

    # 5. Muestreo de control de GHI (Sin Hard Mining Sesgado)
    print(f"  Extrayendo GHI de {len(candidatos_neg)} candidatos negativos...")
    candidatos_neg = extraer_valores_puntos(candidatos_neg, rutas_rasters['ghi_32719'], 'ghi')
    candidatos_neg = candidatos_neg.dropna(subset=['ghi'])
    print(f"  Candidatos tras limpiar GHI NaN: {len(candidatos_neg)}")

    if len(candidatos_neg) == 0:
        raise ValueError("No hay candidatos negativos con GHI válido. Revisa los rasters.")
    
    # El pool ahora no extrae los valores top sino que mantiene la distribución original de GHI para evitar sesgos. Se limita a un pool representativo.
    n_pool = min(len(candidatos_neg), max(n_neg_deseados * 4, 12000))
    if len(candidatos_neg) > n_pool:
        candidatos_top = candidatos_neg.sample(n=n_pool, random_state=random_state).copy()
    else:
        candidatos_top = candidatos_neg.copy()
    candidatos_top['clase'] = 0
    print(f"  Pool seleccionado (muestreo aleatorio representativo): {len(candidatos_top)}")

    # 6. Muestreo conjunto de capas raster
    print(f"  Muestreando rasters para {len(positivas)} positivas + {len(candidatos_top)} candidatos...")
    muestras_pre = gpd.GeoDataFrame(
        pd.concat(
            [positivas[['clase', 'geometry']], candidatos_top[['clase', 'ghi', 'geometry']]],
            ignore_index=True,
        ),
        geometry='geometry',
        crs='EPSG:32719',
    )

    muestras_pre = extraer_valores_puntos(muestras_pre, rutas_rasters['slope'], 'slope')
    muestras_pre = extraer_valores_puntos(muestras_pre, rutas_rasters['aspect'], 'aspect')
    muestras_pre = extraer_valores_puntos(muestras_pre, rutas_rasters['dem_32719'], 'elev')

    # Convertir variable circular 'aspect' a 'northness' (Coseno en Radianes)
    print("  Transformando variable circular 'aspect' a 'northness'...")
    muestras_pre['northness'] = np.cos(np.radians(muestras_pre['aspect']))

    # Calcular distancia euclidiana continua para el modelo
    print("  Calculando distancias euclidianas exactas a la infraestructura para el modelo...")
    if len(lineas_norte) > 0:
        muestras_pre['dist_transmision'] = muestras_pre.geometry.apply(
            lambda g: lineas_norte.geometry.distance(g).min() if g is not None and not g.is_empty else np.nan
        )
    else:
        muestras_pre['dist_transmision'] = 0.0

    if len(almacen_norte) > 0:
        muestras_pre['dist_almacen'] = muestras_pre.geometry.apply(
            lambda g: almacen_norte.geometry.distance(g).min() if g is not None and not g.is_empty else np.nan
        )
    else:
        muestras_pre['dist_almacen'] = 0.0

    if len(subestaciones_norte) > 0:
        muestras_pre['dist_subestaciones'] = muestras_pre.geometry.apply(
            lambda g: subestaciones_norte.geometry.distance(g).min() if g is not None and not g.is_empty else np.nan
        )
    else:
        muestras_pre['dist_subestaciones'] = 0.0

    # 7. Separar clases
    positivas = muestras_pre[muestras_pre['clase'] == 1].copy()
    print(f"  Positivas tras muestreo y extracción de rasters: {len(positivas)}")
    negativos_pre = muestras_pre[muestras_pre['clase'] == 0].copy()

    # 8. Rellenar NaN en positivas
    print("  Rellenando NaN en positivas...")
    for col, raster_path in [
        ('ghi',    rutas_rasters['ghi_32719']),
        ('slope',  rutas_rasters['slope']),
        ('aspect', rutas_rasters['aspect']),
        ('elev',   rutas_rasters['dem_32719']),
    ]:
        positivas = fill_missing_from_raster(positivas, raster_path, col)
        if positivas[col].isna().any():
            with rasterio.open(raster_path) as src:
                nodata = src.nodata if src.nodata is not None else -9999.0
                band = src.read(1)
                mean_val = float(np.nanmean(band[band != nodata]))
            positivas[col] = positivas[col].fillna(mean_val)
    
    # Recalcular northness para positivas tras corregir NaN en aspect
    positivas['northness'] = np.cos(np.radians(positivas['aspect']))

    # 9. Limpiar negativos y aplicar criterios AHP dinámicos (Agregando dist_transmision a dropna)
    negativos_pre = negativos_pre.dropna(subset=['ghi', 'slope', 'aspect', 'elev', 'northness', 'dist_transmision', 'dist_almacen', 'dist_subestaciones'])
    print(f"  Negativos tras limpiar NaN: {len(negativos_pre)}")

    filtro_ahp = (
        (negativos_pre['ghi']   >= criterios['ghi_min']) &
        (negativos_pre['slope'] <= criterios['slope_max']) &
        (negativos_pre['elev']  <= criterios['elev_max'])
    )

    # Si hay líneas, aplicar también el criterio de distancia máxima a transmisión
    if len(lineas_norte) > 0:
        filtro_ahp &= negativos_pre['dist_transmision'] <= criterios.get('dist_max', 20000)

    # Crear pool de negativos finales con clase 0
    pool_negativos = negativos_pre[filtro_ahp].copy()
    pool_negativos['clase'] = 0

    print(f"\n=== RESUMEN MUESTREO ===")
    print(f"Muestras positivas generadas: {len(positivas)}")
    print(f"Pool de negativos elegibles:  {len(pool_negativos)}")

    if len(pool_negativos) > 0:
        print(f"  GHI negat:   min={pool_negativos['ghi'].min():.1f}, max={pool_negativos['ghi'].max():.1f}")
        print(f"  Slope negat: min={pool_negativos['slope'].min():.1f}, max={pool_negativos['slope'].max():.1f}")
        print(f"  Elev negat:  min={pool_negativos['elev'].min():.0f}, max={pool_negativos['elev'].max():.0f}")
        print(f"  Dist negat:  min={pool_negativos['dist_transmision'].min():.1f}, max={pool_negativos['dist_transmision'].max():.1f}")

    # Retorna los GeoDataFrames listos conteniendo la columna 'dist_transmision'
    return positivas, pool_negativos