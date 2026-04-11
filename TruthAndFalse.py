import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# ==========================================
# 1. CARGAR Y FILTRAR MUESTRAS POSITIVAS
# ==========================================
print("Procesando muestras positivas para el Desierto de Atacama...")

# Leer el CSV con el separador correcto
df = pd.read_csv('data_export_2026-04-11.csv', sep='|', encoding='latin-1')

# XV: Arica y parinacota, I: Tarapacá, II: Antofagasta, III: Atacama Hay mas regiones en los datos pero el foco esta al norte
regiones_norte = ['XV', 'I', 'II', 'III']
df_atacama = df[df['REGION'].isin(regiones_norte)].copy()

print(f"Plantas filtradas en el norte: {len(df_atacama)} de {len(df)} totales.")

# Crear la geometría a partir de las coordenadas
geometria = [Point(xy) for xy in zip(df_atacama['NUEVO_X'], df_atacama['NUEVO_Y'])]


gdf_positivos = gpd.GeoDataFrame(df_atacama, geometry=geometria, crs="EPSG:4326")

# Reproyectar a coordenadas proyectadas (UTM 19S) como pide la hoja de ruta
gdf_positivos = gdf_positivos.to_crs(epsg=32719)

# Para random forest necesitamos una columna de clase (1 para positivos)
gdf_positivos['clase'] = 1
gdf_positivos = gdf_positivos[['NOMBRE_PROYECTO', 'REGION', 'clase', 'geometry']]


# ==========================================
# 2. GENERAR MUESTRAS NEGATIVAS EN EL DESIERTO
# ==========================================
print("Generando muestras negativas...")

# Obtener los límites extremos (Norte, Sur, Este, Oeste) solo de las plantas del norte
minx, miny, maxx, maxy = gdf_positivos.total_bounds

# Margen menor para no salirnos del territorio desértico (aprox 10 km)
margen = 10000 

# Buffer de 2 km alrededor de plantas reales para evitar falsos negativos
buffer_positivos = gdf_positivos.geometry.buffer(2000).union_all()

# Igualamos la cantidad de muestras negativas a las positivas para balancear el modelo (Toca ver si esto es mejor asi o de otra manera)
cantidad_negativos = len(gdf_positivos)
puntos_negativos = []

while len(puntos_negativos) < cantidad_negativos:
    # Generar coordenadas aleatorias dentro del recuadro del Desierto de Atacama
    random_x = np.random.uniform(minx - margen, maxx + margen)
    random_y = np.random.uniform(miny - margen, maxy + margen)
    punto_candidato = Point(random_x, random_y)
    
    # Los puntos tiene que estar fuera del buffer las plantas que si existen
    if not punto_candidato.within(buffer_positivos):
        puntos_negativos.append(punto_candidato)

# Crear GeoDataFrame de negativos
gdf_negativos = gpd.GeoDataFrame(geometry=puntos_negativos, crs="EPSG:32719")
gdf_negativos['clase'] = 0
gdf_negativos['NOMBRE_PROYECTO'] = 'Punto Aleatorio (Desierto)'
gdf_negativos['REGION'] = 'Aleatoria'


# ==========================================
# 3. COMBINAR Y EXPORTAR
# ==========================================
print("Uniendo y exportando dataset final...")

dataset_final = pd.concat([gdf_positivos, gdf_negativos], ignore_index=True)

archivo_salida = "dataset_ml_atacama_epsg32719.gpkg"
dataset_final.to_file(archivo_salida, driver="GPKG")

print(f"¡Listo! Dataset guardado como '{archivo_salida}'.")