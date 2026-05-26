import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.merge import merge
import glob
import os
import surtgis
from shapely.geometry import Point
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

"""
Carga de Archivos de estudio
"""

ruta_ghi = 'GHI_32718.tif'  # Usar versión reproyectada a EPSG:32718

# Cargamos todas las regiones de Chile
Regiones = gpd.read_file('REGIONES/REGIONES_v1.shp')

# Cargamos las plantas fotovoltaicas
Fotovoltaicas = gpd.read_file('Paneles.gdb')

# Cargamos las lienas de transmisión
Lineas = gpd.read_file('Linea_de_Transmision/Línea_de_Transmisión.shp')

# Cargamos los almacenamientos de energía
Almacenamientos = gpd.read_file('Almacenamiento_de_Energia/Almacenamiento_de_Energía.shp')

# Cargamos las subestaciones eléctricas
Subestaciones = gpd.read_file('Subestaciones/Subestaciones.shp')

"""
Filtrado para regiones de interes
"""

#Filtramos las regiones de Antofagasta, Atacama
Regiones_filtradas = Regiones[Regiones['REGION'].isin(['Antofagasta', 'Atacama'])]

# Filtramos las plantas fotovoltaicas
codigos_norte = [2, 3, '2', '3', '02', '03']

# Filtramos usando los códigos en lugar de los nombres
Fotovoltaicas_filtradas = Fotovoltaicas[Fotovoltaicas['REGION'].isin(codigos_norte)]

# Filtramos las líneas de transmisión
Lineas_filtradas = Lineas[Lineas['REGION'].isin(codigos_norte)]

# Filtramos los almacenamientos de energía
Almacenamientos_filtrados = Almacenamientos[Almacenamientos['REGION'].isin(codigos_norte)]

# Filtramos las subestaciones eléctricas
Subestaciones_filtradas = Subestaciones[Subestaciones['REGION'].isin(codigos_norte)]


# --- PASO 3: REPROYECCIÓN DE VECTORES ---
# Todo debe estar estrictamente en EPSG:32718
Fotovoltaicas_filtradas = Fotovoltaicas_filtradas.to_crs(epsg=32718)
Lineas_filtradas = Lineas_filtradas.to_crs(epsg=32718)
Regiones_filtradas = Regiones_filtradas.to_crs(epsg=32718)

# --- PASO 4: GENERACIÓN DE MUESTRAS (POSITIVAS Y HARD NEGATIVES) ---
print("Generando muestras de entrenamiento...")

# 4.1 Positivas: Centroides de plantas existentes
positivas = gpd.GeoDataFrame({
    'id_muestra': range(len(Fotovoltaicas_filtradas)),
    'clase': 1
}, geometry=Fotovoltaicas_filtradas.geometry.centroid, crs="EPSG:32718")

# 4.2 Negativas (Hard Negative Mining): Alta radiación, pero sin plantas
# Generamos puntos aleatorios dentro de las regiones, extraemos su GHI y filtramos los mejores
bounds = Regiones_filtradas.total_bounds
puntos_random = []
# Generamos el triple de puntos para poder filtrar y quedarnos con los "Hard Negatives"
n_negativos_deseados = len(positivas) * 2 

while len(puntos_random) < n_negativos_deseados * 5:
    x = np.random.uniform(bounds[0], bounds[2])
    y = np.random.uniform(bounds[1], bounds[3])
    pto = Point(x, y)
    if Regiones_filtradas.contains(pto).any():
        puntos_random.append(pto)

candidatos_negativos = gpd.GeoDataFrame(geometry=puntos_random, crs="EPSG:32718")

# Excluimos un buffer de 5km alrededor de plantas existentes para no solapar
buffer_plantas = positivas.geometry.buffer(5000).union_all()
candidatos_negativos = candidatos_negativos[~candidatos_negativos.geometry.intersects(buffer_plantas)]

# Extraemos el GHI para los candidatos usando una función auxiliar de rasterio
# (Usamos rasterio.sample para puntos masivos en memoria, ya que surtgis.zonal_stats 
# está pensado para polígonos/zonas o requiere guardar un SHP intermedio)
def extraer_valores_puntos(gdf, raster_path, col_name):
    with rasterio.open(raster_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        # Filtrar valores muestreados, reemplazando nodata con NaN para limpieza posterior
        valores = []
        for val in src.sample(coords):
            v = val[0]
            valores.append(np.nan if (v == nodata or np.isnan(v)) else v)
        gdf[col_name] = valores
    return gdf

candidatos_negativos = extraer_valores_puntos(candidatos_negativos, ruta_ghi, 'ghi')

# HARD NEGATIVE MINING: Nos quedamos con los que tienen el GHI más alto (los "falsos atractivos")
# Limpiamos NaN que hayan quedado del muestreo
candidatos_negativos = candidatos_negativos.dropna(subset=['ghi'])
print(f"  Candidatos negativos después de limpiar GHI NaN: {len(candidatos_negativos)}")
if len(candidatos_negativos) > 0:
    candidatos_negativos = candidatos_negativos.sort_values(by='ghi', ascending=False).head(n_negativos_deseados)
candidatos_negativos['clase'] = 0
negativas = candidatos_negativos[['clase', 'geometry']]

# Dataset final unificado
muestras_ml = pd.concat([positivas[['clase', 'geometry']], negativas[['clase', 'geometry']]], ignore_index=True)
muestras_ml = gpd.GeoDataFrame(muestras_ml, geometry='geometry', crs="EPSG:32718")

# --- PASO 5: EXTRACCIÓN DE VARIABLES EXPLICATIVAS (FEATURES) ---
print("Extrayendo features para el modelo ML...")

# 5.1 Distancia a la línea de transmisión más cercana (en metros)
# Hacemos un spatial join o nearest neighbor
lineas_union = Lineas_filtradas.geometry.union_all()

# 5.2 Valores de Rasters (GHI, Slope, Aspect)
print(f"  Muestreando GHI...")
muestras_ml = extraer_valores_puntos(muestras_ml, ruta_ghi, 'ghi')
ghi_valid = muestras_ml['ghi'].notna().sum()
print(f"    Valores GHI válidos: {ghi_valid}/{len(muestras_ml)}")

print(f"  Muestreando Slope...")
muestras_ml = extraer_valores_puntos(muestras_ml, 'slope_norte.tif', 'slope')
slope_valid = muestras_ml['slope'].notna().sum()
print(f"    Valores Slope válidos: {slope_valid}/{len(muestras_ml)}")

print(f"  Muestreando Aspect...")
muestras_ml = extraer_valores_puntos(muestras_ml, 'aspect_norte.tif', 'aspect')
aspect_valid = muestras_ml['aspect'].notna().sum()
print(f"    Valores Aspect válidos: {aspect_valid}/{len(muestras_ml)}")

# Limpiamos posibles valores NoData (ej. puntos fuera del raster)
print(f"  Muestras antes de limpiar NaN: {len(muestras_ml)}")
muestras_ml = muestras_ml.dropna(subset=['ghi', 'slope', 'aspect'])
print(f"  Muestras después de limpiar NaN: {len(muestras_ml)}")
muestras_ml = muestras_ml[(muestras_ml['slope'] >= 0) & (muestras_ml['ghi'] >= 0)]
print(f"  Muestras después de filtro slope/ghi: {len(muestras_ml)}")

# AHORA calcular distancia a líneas DESPUÉS de haber filtrado por rasters
if len(muestras_ml) > 0:
    muestras_ml['dist_transmision'] = muestras_ml.geometry.distance(lineas_union)
else:
    print("\n¡ERROR! No hay muestras válidas después del filtrado de rasters.")
    print("Revisa:")
    print("  - ¿El GHI.tif existe y está en EPSG:32718?")
    print("  - ¿Los puntos caen dentro del raster?")
    print("  - ¿Los valores de slope/aspect son válidos?")
    exit(1) 

# --- PASO 6: ENTRENAMIENTO DEL RANDOM FOREST ---
print("Entrenando Random Forest y extrayendo pesos empíricos...")
features = ['ghi', 'slope', 'aspect', 'dist_transmision']
X = muestras_ml[features]
y = muestras_ml['clase']

rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
rf.fit(X, y)

# Extraer y formatear las importancias (pesos)
pesos_empiricos = pd.DataFrame({
    'Criterio': features,
    'Peso_ML': rf.feature_importances_
}).sort_values(by='Peso_ML', ascending=False)

# Normalizar para que sumen 1 (o 100%)
pesos_empiricos['Peso_ML'] = pesos_empiricos['Peso_ML'] / pesos_empiricos['Peso_ML'].sum()

print("\n--- RESULTADOS BRECHA 1: PESOS DATA-DRIVEN ---")
print(pesos_empiricos.to_string(index=False))

# Opcional: Guardar el dataset para usarlo luego con SHAP (Brecha 8)
muestras_ml.to_file("dataset_entrenamiento_rf.shp")