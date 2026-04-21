import ee
import osmnx as ox
import geopandas as gpd
import surtgis
import os
from dotenv import load_dotenv
import unicodedata
import re

# Autenticación e Inicialización de Google Earth Engine (GEE)
load_dotenv()
ID_PROYECTO = os.getenv("ID_PROYECTO")
print("Iniciando conexión con Google Earth Engine...")

try: 
    ee.Initialize(project=ID_PROYECTO)
    print("GEE Inicializado correctamente.")

except Exception as e:
    print("Se requiere autenticación manual...")
    ee.Authenticate() 
    ee.Initialize(project=ID_PROYECTO)
    print("Autenticación e inicialización exitosas.")
    print("GEE Inicializado correctamente.")


def limpiar_nombre_region(nombre):
    #Limpia el texto quitando tildes, 'ñ', caracteres especiales y limitando el largo, para evitar problemas con nombres de archivos en Drive
    nombre_limpio = unicodedata.normalize('NFKD', nombre).encode('ASCII', 'ignore').decode('utf-8')
    nombre_limpio = re.sub(r'[^a-zA-Z0-9_]', '_', nombre_limpio)
    nombre_limpio = re.sub(r'_+', '_', nombre_limpio)
    return nombre_limpio[:80].strip('_')

def obtener_DEM_regiones():
    dataset = ee.Image('USGS/SRTMGL1_003')
    elevation = dataset.select('elevation')

    regiones_chile = ee.FeatureCollection("FAO/GAUL/2015/level1") \
        .filter(ee.Filter.eq('ADM0_NAME', 'Chile'))

    print("Obteniendo lista de regiones...")
    nombres_regiones = regiones_chile.aggregate_array('ADM1_NAME').getInfo()
    
    print(f"Se encontraron {len(nombres_regiones)} regiones. Iniciando envío de tareas...")

    for nombre in nombres_regiones:
        # Aplicamos la nueva limpieza extrema de nombres
        nombre_limpio = limpiar_nombre_region(nombre)
        
        region_geom = regiones_chile.filter(ee.Filter.eq('ADM1_NAME', nombre)).geometry()
        
        print(f"-> Enviando a Drive: {nombre_limpio}")
        
        task = ee.batch.Export.image.toDrive(
            image=elevation.clip(region_geom),
            description=f'DEM_20m_{nombre_limpio}',
            folder='DEM_Chile_Regiones_20m',
            fileNamePrefix=f'dem_{nombre_limpio}_20m',
            region=region_geom,
            scale=20,
            crs='EPSG:32719',
            maxPixels=1e13
        )
        
        task.start()

    print("\n¡Todo listo!")
    print("Las tareas están en cola. Monitorea el progreso aquí: https://code.earthengine.google.com/tasks")
#obtener_DEM_regiones()

# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE?hl=es-419#description
def descargar_ghi_promedio_2025():
    print("Configurando dataset de Radiación Solar (TerraClimate)...")
    
    # 2. Cargar colección de imágenes de TerraClimate para el año 2025
    # La banda 'srad' es la radiación de onda corta descendente (GHI)
    dataset_2025 = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
                    .filter(ee.Filter.date('2025-01-01', '2025-12-31')) \
                    .select('srad')
    
    # 3. Calcular el promedio anual en la nube
    # Multiplicamos por 0.1 porque es el factor de escala oficial de TerraClimate 
    # para obtener W/m^2 reales.
    ghi_promedio = dataset_2025.mean().multiply(0.1)

    # 4. Obtener polígonos de las regiones de Chile
    regiones_chile = ee.FeatureCollection("FAO/GAUL/2015/level1") \
        .filter(ee.Filter.eq('ADM0_NAME', 'Chile'))

    print("Obteniendo lista de regiones...")
    nombres_regiones = regiones_chile.aggregate_array('ADM1_NAME').getInfo()
    
    print(f"Se enviarán {len(nombres_regiones)} tareas a la nube.")

    # 5. Ciclo de exportación por región
    for nombre in nombres_regiones:
        nombre_limpio = limpiar_nombre_region(nombre)
        region_geom = regiones_chile.filter(ee.Filter.eq('ADM1_NAME', nombre)).geometry()
        
        print(f"-> Preparando exportación: {nombre_limpio}")
        
        task = ee.batch.Export.image.toDrive(
            image=ghi_promedio.clip(region_geom),
            description=f'GHI_Promedio_2025_{nombre_limpio}',
            folder='Radiacion_Solar_Chile_2025', 
            fileNamePrefix=f'ghi_2025_{nombre_limpio}',
            region=region_geom,
            scale=4638,          # Resolución nativa de TerraClimate (~4.6 km)
            crs='EPSG:32719',    # UTM 19S
            maxPixels=1e13
        )
        
        task.start()

    print("\n¡Proceso finalizado!")
    print("Todas las regiones del 2025 están siendo procesadas en Google Drive.")
    print("Puedes ver el avance aquí: https://code.earthengine.google.com/tasks")
    
descargar_ghi_promedio_2025()