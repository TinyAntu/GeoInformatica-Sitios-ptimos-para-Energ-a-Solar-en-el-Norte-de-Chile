import os
import sys
import yaml
import joblib
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

# 1. Ajuste de rutas para importar módulos locales
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

def read_and_reproject_to_grid(src_path, dst_crs, dst_shape, dst_transform, nodata_val=np.nan):
    """Reproyecta un raster en memoria para que coincida con la grilla base."""
    print(f"  Reproyectando {os.path.basename(src_path)} a la grilla base...")
    with rasterio.open(src_path) as src:
        destination = np.empty(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=nodata_val
        )
        return destination

def get_rasterized_mask(gdf, dst_shape, dst_transform, fill=0, default_value=1):
    """Rasteriza geometrías vectoriales sobre la grilla base."""
    if gdf is None or len(gdf) == 0:
        return np.full(dst_shape, fill, dtype=np.uint8)
    
    gdf_valid = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if len(gdf_valid) == 0:
        return np.full(dst_shape, fill, dtype=np.uint8)
    
    mask = rasterize(
        shapes=gdf_valid.geometry,
        out_shape=dst_shape,
        transform=dst_transform,
        fill=fill,
        default_value=default_value,
        all_touched=True
    )
    return mask

def generar_grafico_precision_recall(paths_results, directorio_raiz, model):
    """
    Carga el dataset de entrenamiento, extrae el set de validación (test)
    y genera/guarda la curva Precision-Recall para evaluar la calibración del modelo.
    """
    try:
        print("\n--- Generando gráfica de curva Precision-Recall para validación ---")
        # Obtener ruta desde la configuración con un fallback seguro
        dataset_ml_path = paths_results.get('dataset_ml', 'data/results/dataset_entrenamiento_rf.shp')
        
        if not os.path.exists(dataset_ml_path):
            print(f"  [AVISO] No se encontró el SHP en {dataset_ml_path}. Se omite el gráfico.")
            return

        df_eval = gpd.read_file(dataset_ml_path)
        
        # Ajustar nombre de columna si ESRI truncó el archivo físico a 10 caracteres (.shp)
        if 'dist_trans' in df_eval.columns:
            df_eval = df_eval.rename(columns={'dist_trans': 'dist_transmision'})
            
        # Matriz de características en el orden estricto de entrenamiento
        X_eval = df_eval[['slope', 'ghi', 'elev', 'northness', 'dist_transmision']]
        y_eval = df_eval['clase']
        
        # Separar el 30% de validación usando la misma semilla aleatoria (random_state=42)
        _, X_test_ev, _, y_test_ev = train_test_split(
            X_eval, y_eval, test_size=0.3, stratify=y_eval, random_state=42
        )
        
        # Calcular probabilidades de la clase positiva (1)
        y_probs_ev = model.predict_proba(X_test_ev)[:, 1]
        
        # Construir la curva y calcular el Average Precision (AP)
        precision, recall, _ = precision_recall_curve(y_test_ev, y_probs_ev)
        ap_score = average_precision_score(y_test_ev, y_probs_ev)
        
        # Diseño y construcción de la figura con Matplotlib
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                 label=f'Curva PR (Average Precision = {ap_score:.4f})')
        plt.xlabel('Recall (Sensibilidad)')
        plt.ylabel('Precision (Exactitud de Predicción)')
        plt.title('Curva Precision-Recall - Modelo de Aptitud Solar')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Guardar la imagen en alta resolución
        plot_output_path = os.path.join(directorio_raiz, 'data/results/curva_precision_recall.png')
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica PR guardada en: {plot_output_path}\n")
            
    except Exception as e:
        print(f"  [AVISO] No se pudo generar la gráfica PR automáticamente: {e}")

def main():
    print("=== INICIANDO GENERACIÓN DE MAPA DE APTITUD SOLAR ===")

    # Cargar configuraciones
    ruta_config = os.path.join(directorio_raiz, 'config.yaml')
    with open(ruta_config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    paths_raw = config['paths']['raw']
    paths_processed = config['paths']['processed']
    paths_results = config['paths']['results']

    model_path = paths_results.get('model_rf', 'data/results/model_rf.pkl')
    output_tif_path = os.path.join(directorio_raiz, 'data/results/mapa_probabilidad_aptitud.tif')
    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)

    # Verificar que el modelo exista
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo entrenado en {model_path}. Por favor ejecuta run_pipeline primero.")
        return

    print(f"Cargando modelo Random Forest desde: {model_path}")
    model = joblib.load(model_path)

    # 2. Abrir el raster GHI como base de referencia
    ghi_path = paths_processed['ghi_32718']
    print(f"Usando como grilla de referencia: {ghi_path}")
    
    with rasterio.open(ghi_path) as ghi_src:
        meta_base = ghi_src.meta.copy()
        transform = ghi_src.transform
        width = ghi_src.width
        height = ghi_src.height
        crs = ghi_src.crs
        
        # Leer radiación cruda
        ghi_data = ghi_src.read(1)
        # Nodata en el GHI base
        nodata_ghi = ghi_src.nodata if ghi_src.nodata is not None else -9999.0
        ghi_mask_valid = (ghi_data != nodata_ghi) & (~np.isnan(ghi_data))

    grid_shape = (height, width)
    print(f"Dimensiones de la grilla de trabajo: {grid_shape}")

    # 3. Cargar y reproyectar variables raster
    dem_path = paths_processed['dem_32718']
    slope_path = paths_processed['slope']
    aspect_path = paths_processed['aspect']

    elev_data = read_and_reproject_to_grid(dem_path, crs, grid_shape, transform, nodata_val=np.nan)
    slope_data = read_and_reproject_to_grid(slope_path, crs, grid_shape, transform, nodata_val=np.nan)
    aspect_data = read_and_reproject_to_grid(aspect_path, crs, grid_shape, transform, nodata_val=np.nan)

    # CORRECCIÓN 1: Transformación lineal de variable circular 'aspect' a 'northness' en las matrices
    print("  Transformando variable matricial 'aspect' a 'northness'...")
    with np.errstate(invalid='ignore'):
        northness_data = np.cos(np.radians(aspect_data))

    # 4. Calcular distancia a líneas de transmisión
    lineas_path = paths_raw['vectores']['lineas']
    print(f"Cargando líneas de transmisión desde: {lineas_path}")
    lineas_gdf = gpd.read_file(lineas_path).to_crs(crs)
    
    print("  Rasterizando líneas de transmisión y calculando distancia euclidiana...")
    lineas_mask = get_rasterized_mask(lineas_gdf, grid_shape, transform, fill=0, default_value=1)
    
    # distance_transform_edt calcula la distancia a los píxeles con valor 0. Invertimos el mask.
    lines_inverted = (lineas_mask == 0).astype(np.uint8)
    dist_pixels = distance_transform_edt(lines_inverted)
    
    # Convertir distancias de píxeles a metros (el transform[0] nos da el ancho del píxel en metros)
    pixel_size_meters = transform[0]
    dist_transmision_data = dist_pixels * pixel_size_meters
    print(f"  Distancia máxima calculada a líneas de transmisión: {dist_transmision_data.max():.2f} metros")

    # 5. Generar Máscara de Regiones (Antofagasta y Atacama)
    regiones_path = paths_raw['vectores']['regiones']
    print(f"Cargando regiones desde: {regiones_path}")
    regiones_gdf = gpd.read_file(regiones_path)
    regiones_norte = regiones_gdf[regiones_gdf['REGION'].isin(['Antofagasta', 'Atacama'])].to_crs(crs)
    region_mask = get_rasterized_mask(regiones_norte, grid_shape, transform, fill=0, default_value=1)

    # 6. Preparar máscara de píxeles válidos para predecir
    # Un píxel es válido si está dentro de la región y no es NaN en ninguna variable de entrada
    valid_mask = (
        (region_mask == 1) &
        ghi_mask_valid &
        (~np.isnan(elev_data)) &
        (~np.isnan(slope_data)) &
        (~np.isnan(northness_data)) &  # Cambiado aspect_data por northness_data
        (~np.isnan(dist_transmision_data))
    )

    n_valid_pixels = np.sum(valid_mask)
    print(f"Cantidad de píxeles válidos a predecir: {n_valid_pixels} de {width * height}")

    if n_valid_pixels == 0:
        print("ERROR: No hay píxeles válidos para predecir. Revisa la superposición de tus rasters.")
        return

    # Extraer valores y aplanar para alimentar al modelo
    slope_flat = slope_data[valid_mask]
    ghi_flat = ghi_data[valid_mask]
    elev_flat = elev_data[valid_mask]
    northness_flat = northness_data[valid_mask]
    dist_flat = dist_transmision_data[valid_mask]

    # CORRECCIÓN 2: Firma de orden de entrada estricta del modelo
    # features = ['slope', 'ghi', 'elev', 'northness', 'dist_transmision']
    X_pred = np.column_stack((slope_flat, ghi_flat, elev_flat, northness_flat, dist_flat))

    # 7. Ejecutar predicciones de probabilidad
    print("Prediciendo probabilidades de aptitud solar con Random Forest...")
    # predict_proba retorna [probabilidad_clase_0, probabilidad_clase_1]
    # Queremos la probabilidad de la clase 1 (sitio óptimo)
    prob_predictions = model.predict_proba(X_pred)[:, 1]

    # Reconstruir la grilla bidimensional
    suitability_grid = np.full(grid_shape, -9999.0, dtype=np.float32)
    suitability_grid[valid_mask] = prob_predictions

    # 8. Cargar y aplicar zonas de exclusión absoluta (SNAP, lagos, áreas pobladas)
    # Estas zonas tendrán idoneidad/probabilidad de 0.0 de forma obligatoria
    print("Aplicando zonas de exclusión territorial (SNAP, masas de agua, zonas pobladas)...")
    
    exclusion_paths = {
        'snap': paths_raw['vectores'].get('snap'),
        'lagos': paths_raw['vectores'].get('masas_lacustres'),
        'poblaciones': paths_raw['vectores'].get('areas_pobladas')
    }
    
    for key, path in exclusion_paths.items():
        if path and os.path.exists(path):
            print(f"  Aplicando exclusión: {key} ({os.path.basename(path)})...")
            gdf_excl = gpd.read_file(path).to_crs(crs)
            mask_excl = get_rasterized_mask(gdf_excl, grid_shape, transform, fill=0, default_value=1)
            # Todo píxel en zona de exclusión y que sea válido se fuerza a 0.0
            suitability_grid[(mask_excl == 1) & (suitability_grid != -9999.0)] = 0.0

    # 9. Grafico de validación adicional: Curva Precision-Recall para evaluar el desempeño del modelo en el dataset de entrenamiento
    generar_grafico_precision_recall(paths_results, directorio_raiz, model)

    # 10. Guardar mapa en formato GeoTIFF
    meta_base.update(
        dtype=rasterio.float32,
        nodata=-9999.0,
        count=1
    )
    
    print(f"Guardando raster de aptitud en: {output_tif_path}")
    with rasterio.open(output_tif_path, 'w', **meta_base) as dst:
        dst.write(suitability_grid, 1)

    print("=== MAPA GENERADO CON ÉXITO ===")
    print("Puedes abrir este archivo directamente en QGIS o ArcGIS para su análisis cartográfico.")

if __name__ == "__main__":
    main()