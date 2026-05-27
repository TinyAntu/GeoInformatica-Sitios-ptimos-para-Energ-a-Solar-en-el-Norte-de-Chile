import yaml as config
from src.preprocessing import procesar_dem, cargar_capas_vectoriales

# ... (código para leer el config.yaml) ...

# 1. Cargar TODOS tus vectores limpiamente
vectores = cargar_capas_vectoriales(config['paths']['raw']['vectores'])
# Para usar áreas pobladas después, solo llamas a: vectores['areas_pobladas']

# 2. Procesar el DEM
procesar_dem(
    carpetas_dem=config['paths']['raw']['rasters']['dem_folders'],
    out_slope_path=config['paths']['processed']['slope'],
    out_aspect_path=config['paths']['processed']['aspect']
)