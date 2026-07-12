import os
import sys
import yaml
import joblib
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo: solo guardamos figuras, evita el crash de Tkinter en hilos
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

# 1. Ajuste de rutas para importar módulos locales
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.features import FEATURES  # orden canónico de features (ver src/features.py)

def read_and_reproject_to_grid(src_path, dst_crs, dst_shape, dst_transform, nodata_val=np.nan,
                               resampling=Resampling.bilinear):
    """Reproyecta un raster en memoria para que coincida con la grilla base.

    `resampling` es bilinear por defecto (variables continuas), pero debe ser
    Resampling.nearest para variables que NO se pueden interpolar linealmente
    (p. ej. 'aspect', que es circular: promediar 359° y 1° daría ~180°).
    """
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
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=nodata_val
        )
        return destination

def construir_grilla_referencia(dem_path, resolucion_m=None):
    """Grilla de inferencia del mapa de aptitud (hallazgo 2.1, opción B).

    Se usa el DEM como base (resolución fina, ~30 m) en vez del GHI (~1 km), para que
    las distancias euclidianas calculadas en inferencia se acerquen a las distancias
    exactas por vector usadas en el entrenamiento (src/sampling.py).

    Si `resolucion_m` es None se usa la grilla nativa del DEM; si se especifica, se
    reconstruye la grilla a esa resolución sobre la misma extensión (permite bajar la
    resolución si falta memoria). Devuelve (crs, transform, width, height).
    """
    with rasterio.open(dem_path) as src:
        crs = src.crs
        if resolucion_m is None:
            return crs, src.transform, src.width, src.height
        bounds = src.bounds
    width = max(1, int(round((bounds.right - bounds.left) / resolucion_m)))
    height = max(1, int(round((bounds.top - bounds.bottom) / resolucion_m)))
    transform = from_origin(bounds.left, bounds.top, resolucion_m, resolucion_m)
    return crs, transform, width, height


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

def calcular_distancia_a_capa(path, crs, grid_shape, transform, pixel_size_meters, nombre=""):
    """Carga una capa vectorial, la rasteriza sobre la grilla base y devuelve la distancia
    euclidiana (en metros) de cada píxel a la geometría más cercana de esa capa."""
    etiqueta = nombre or os.path.basename(path)
    print(f"  Cargando y calculando distancia euclidiana a {etiqueta}...")
    gdf = gpd.read_file(path).to_crs(crs)
    mask = get_rasterized_mask(gdf, grid_shape, transform, fill=0, default_value=1)

    if mask.max() == 0:
        # La capa no aportó geometrías válidas dentro de la grilla.
        print(f"  [AVISO] '{etiqueta}' no rasterizó ninguna geometría; se usa distancia = 0.0.")
        return np.zeros(grid_shape, dtype=np.float32)

    # distance_transform_edt mide la distancia a los píxeles con valor 0, por eso invertimos el mask.
    inverted = (mask == 0).astype(np.uint8)
    dist_pixels = distance_transform_edt(inverted)
    dist_metros = (dist_pixels * pixel_size_meters).astype(np.float32)
    print(f"  Distancia máxima a {etiqueta}: {dist_metros.max():.2f} metros")
    return dist_metros

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
        
        # Ajustar nombres de columna si ESRI truncó el archivo físico a 10 caracteres (.shp)
        rename_map = {
            'dist_trans': 'dist_transmision',
            'dist_almac': 'dist_almacen',
            'dist_subs':  'dist_subestaciones',
        }
        df_eval = df_eval.rename(columns={k: v for k, v in rename_map.items() if k in df_eval.columns})

        # Matriz de características en el orden estricto de entrenamiento
        X_eval = df_eval[FEATURES]
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

    # 2. Grilla de referencia = DEM (resolución fina), NO el GHI (~1 km).
    #    Así las distancias euclidianas de inferencia dejan de estar cuantizadas a ~1 km y
    #    se acercan a las distancias exactas del entrenamiento (hallazgo 2.1, opción B).
    dem_path = paths_processed['dem_32719']
    slope_path = paths_processed['slope']
    aspect_path = paths_processed['aspect']
    ghi_path = paths_processed['ghi_32719']

    resolucion_m = config.get('salida_mapa', {}).get('resolucion_m')
    crs, transform, width, height = construir_grilla_referencia(dem_path, resolucion_m)
    grid_shape = (height, width)
    n_pix = width * height
    print(f"Grilla de inferencia (base DEM): {width}x{height} = {n_pix:,} píxeles a ~{transform[0]:.0f} m")
    if n_pix > 60_000_000:
        print("  [AVISO] Grilla muy grande: si te quedas sin memoria, sube 'salida_mapa.resolucion_m' "
              "en config.yaml (p.ej. 100).")

    # Metadatos del GeoTIFF de salida, construidos desde la grilla del DEM.
    meta_base = {
        'driver': 'GTiff', 'dtype': rasterio.float32, 'count': 1,
        'crs': crs, 'transform': transform, 'width': width, 'height': height,
        'nodata': -9999.0,
    }

    # 3. Cargar variables raster sobre la grilla fina.
    # GHI (~1 km) se reproyecta/upsamplea a la grilla del DEM: solo alinea, no agrega
    # información nueva (el dato sigue siendo de 1 km), pero permite predecir en la grilla fina.
    ghi_data = read_and_reproject_to_grid(ghi_path, crs, grid_shape, transform, nodata_val=np.nan)
    ghi_mask_valid = ~np.isnan(ghi_data)

    elev_data = read_and_reproject_to_grid(dem_path, crs, grid_shape, transform, nodata_val=np.nan)
    slope_data = read_and_reproject_to_grid(slope_path, crs, grid_shape, transform, nodata_val=np.nan)
    # 'aspect' es circular (0°/360°): se reproyecta con NEAREST para no interpolar a través
    # de la discontinuidad angular. Además queda consistente con el muestreo por vecino más
    # cercano usado en el entrenamiento (src/sampling.py).
    aspect_data = read_and_reproject_to_grid(aspect_path, crs, grid_shape, transform, nodata_val=np.nan,
                                             resampling=Resampling.nearest)

    # Transformación de la variable circular 'aspect' a 'northness' en las matrices
    print("  Transformando variable matricial 'aspect' a 'northness'...")
    with np.errstate(invalid='ignore'):
        northness_data = np.cos(np.radians(aspect_data))

    # 4. Calcular distancias euclidianas a la infraestructura (mismas 3 que usa el modelo)
    # El transform[0] da el ancho del píxel en metros para convertir distancias a unidades reales.
    pixel_size_meters = transform[0]
    print("Calculando distancias a infraestructura de red...")

    dist_transmision_data = calcular_distancia_a_capa(
        paths_raw['vectores']['lineas'], crs, grid_shape, transform,
        pixel_size_meters, nombre="líneas de transmisión")
    
    dist_almacen_data = calcular_distancia_a_capa(
        paths_raw['vectores']['almacenamiento'], crs, grid_shape, transform,
        pixel_size_meters, nombre="almacenamiento de energía")
    
    dist_subestaciones_data = calcular_distancia_a_capa(
        paths_raw['vectores']['subestaciones'], crs, grid_shape, transform,
        pixel_size_meters, nombre="subestaciones")

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
        (~np.isnan(northness_data)) &
        (~np.isnan(dist_transmision_data)) &
        (~np.isnan(dist_almacen_data)) &
        (~np.isnan(dist_subestaciones_data))
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
    dist_trans_flat = dist_transmision_data[valid_mask]
    dist_almacen_flat = dist_almacen_data[valid_mask]
    dist_subestaciones_flat = dist_subestaciones_data[valid_mask]

    # Firma de orden de entrada estricta del modelo. Debe coincidir con FEATURES
    # (src/features.py); el assert falla en voz alta si alguien reordena las features.
    assert FEATURES == ['slope', 'ghi', 'elev', 'northness',
                        'dist_transmision', 'dist_almacen', 'dist_subestaciones'], (
        "El orden de FEATURES cambió: actualiza el column_stack de X_pred acorde.")
    X_pred = np.column_stack((
        slope_flat, ghi_flat, elev_flat, northness_flat,
        dist_trans_flat, dist_almacen_flat,
        dist_subestaciones_flat,
    ))

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