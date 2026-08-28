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
        if src.crs is None:
            raise ValueError(
                f"'{src_path}' no tiene CRS definido. No se puede reproyectar de forma "
                "confiable: revisa la etapa de preprocesamiento que generó este raster."
            )
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
        if crs is None:
            raise ValueError(
                f"'{dem_path}' no tiene CRS definido. Este raster define la grilla de "
                "referencia de todo el mapa de aptitud: revisa la etapa de preprocesamiento."
            )
        if crs.to_epsg() != 32719:
            raise ValueError(
                f"'{dem_path}' está en {crs} (EPSG:{crs.to_epsg()}), pero el proyecto exige "
                "EPSG:32719 (UTM 19S, ver AGENTS.md). Reproyéctalo antes de generar el mapa."
            )
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

# Debe coincidir con la lista usada en src/sampling.py: mismo criterio de filtrado regional
# en entrenamiento e inferencia, para no introducir sesgo train/inferencia (train-serving skew).
CODIGOS_NORTE = ['2', '3', '02', '03', 'II', 'III', 'Antofagasta', 'Atacama', 'ANTOFAGASTA', 'ATACAMA']


def _filtrar_a_norte(gdf, regiones_norte, nombre="", codigos=None, modo='exacto'):
    """Recorta una capa vectorial a la zona de estudio con el mismo criterio que el
    entrenamiento (src/sampling.py): filtra por columna REGION si existe; si no, recorta
    espacialmente contra las regiones. Sin este filtro, la inferencia calcularía distancias
    contra infraestructura fuera de la zona de estudio que el modelo nunca vio al entrenar.

    `codigos` permite evaluar el modelo sobre una región distinta de la de entrenamiento; por
    defecto usa CODIGOS_NORTE, de modo que las llamadas existentes no cambian.

    `modo` controla cómo se compara la columna REGION:
      - 'exacto'    coincidencia exacta. Es lo que usa el entrenamiento y por tanto lo que
                    hay que replicar para no introducir sesgo train/inferencia.
      - 'compuesto' compara por tokens separados por ';'. La capa de transmisión codifica las
                    líneas que cruzan fronteras como 'ATACAMA;COQUIMBO', y esas cadenas no
                    coinciden con nada en modo exacto: el norte pierde así el 15,6 % de la red
                    y Coquimbo el 47,1 %. Se ofrece para medir ese sesgo, no como default:
                    cambiarlo por defecto desalinearía la inferencia del modelo entrenado.
    """
    if 'REGION' in gdf.columns:
        objetivo = {str(c).upper().strip() for c in (codigos or CODIGOS_NORTE)}
        col = gdf['REGION'].astype(str).str.upper().str.strip()
        if modo == 'compuesto':
            # fillna('') antes de map: la columna REGION trae nulos en algunas capas y, a
            # diferencia de isin() —que los descarta sin ruido—, map() se los pasa al lambda.
            pertenece = col.fillna('').map(
                lambda v: bool(objetivo & {t.strip() for t in str(v).split(';')})
            )
            return gdf[pertenece.astype(bool)]
        return gdf[col.isin(objetivo)]
    try:
        return gpd.clip(gdf, regiones_norte)
    except Exception as e:
        print(f"  [AVISO] No se pudo recortar '{nombre}' espacialmente: {e}. Usando capa completa.")
        return gdf


def calcular_distancia_a_capa(path, crs, grid_shape, transform, pixel_size_meters, regiones_norte, nombre="", codigos=None, modo='exacto'):
    """Carga una capa vectorial, la recorta a la zona de estudio (igual que en entrenamiento,
    ver src/sampling.py), la rasteriza sobre la grilla base y devuelve la distancia euclidiana
    (en metros) de cada píxel a la geometría más cercana de esa capa.

    Nota metodológica: la distancia se calcula sobre una MÁSCARA RASTERIZADA (mismo enfoque
    que "Euclidean Distance" de ArcGIS o "Proximity" de QGIS), no como distancia vectorial
    exacta a la geometría real (que sí se usa en src/sampling.py para las muestras de
    entrenamiento, vía `geometry.distance()`). Esto introduce un sesgo de cuantización
    acotado por ~1 píxel de la grilla (con `salida_mapa.resolucion_m=100` en config.yaml,
    hasta ±100 m), pequeño frente al rango típico de estas distancias (hasta 20 km, ver
    `criterios.dist_max`). Se documenta como limitación conocida y aceptada: reemplazarlo
    por una distancia vectorial exacta punto a punto sobre millones de píxeles de la grilla
    completa sería sustancialmente más costoso computacionalmente.
    """
    etiqueta = nombre or os.path.basename(path)
    print(f"  Cargando y calculando distancia euclidiana a {etiqueta}...")
    gdf = gpd.read_file(path).to_crs(crs)
    gdf = _filtrar_a_norte(gdf, regiones_norte, nombre=etiqueta, codigos=codigos, modo=modo)
    mask = get_rasterized_mask(gdf, grid_shape, transform, fill=0, default_value=1)

    if mask.max() == 0:
        # La capa no aportó geometrías válidas dentro de la grilla. Antes esto retornaba
        # distancia = 0.0 en TODA la grilla, invirtiendo la semántica de la variable ("no hay
        # infraestructura cerca" pasaba a leerse como "estás encima de ella en todas partes").
        # Con infraestructura real (líneas/subestaciones/almacenamiento) en Antofagasta y
        # Atacama, esta rama solo debería activarse por un error de datos o de configuración
        # (ruta mal apuntada, capa vacía tras el filtro regional): se falla explícitamente en
        # vez de seguir con un valor fabricado que contaminaría el resto del pipeline en silencio.
        raise ValueError(
            f"'{etiqueta}' ({path}) no rasterizó ninguna geometría dentro de la grilla de "
            "estudio (Antofagasta/Atacama). Revisa la ruta en config.yaml y que la capa "
            "efectivamente tenga datos en la región — no se genera el mapa con una distancia "
            "fabricada."
        )

    # distance_transform_edt mide la distancia a los píxeles con valor 0, por eso invertimos el mask.
    inverted = (mask == 0).astype(np.uint8)
    # `sampling` usa el espaciado real de cada eje (alto, ancho) en vez de escalar por un único
    # valor: no asume píxel cuadrado. `pixel_size_meters` (transform[0]) es el ancho de píxel
    # (eje X); `transform[4]` es el alto de píxel (eje Y, negativo por la convención
    # norte-arriba de rasterio). Con salida_mapa.resolucion_m fijo ambos coinciden, pero con
    # resolucion_m=null (grilla nativa del DEM) no había garantía de que fueran iguales.
    pixel_size_y = abs(transform[4])
    dist_metros = distance_transform_edt(
        inverted, sampling=(pixel_size_y, pixel_size_meters)
    ).astype(np.float32)
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

    # 4. Generar Máscara de Regiones (Antofagasta y Atacama). Se calcula antes que las
    # distancias porque `calcular_distancia_a_capa` necesita `regiones_norte` para recortar
    # la infraestructura a la zona de estudio (mismo criterio que el entrenamiento).
    regiones_path = paths_raw['vectores']['regiones']
    print(f"Cargando regiones desde: {regiones_path}")
    regiones_gdf = gpd.read_file(regiones_path)
    regiones_norte = regiones_gdf[regiones_gdf['REGION'].isin(['Antofagasta', 'Atacama'])].to_crs(crs)
    region_mask = get_rasterized_mask(regiones_norte, grid_shape, transform, fill=0, default_value=1)

    # 5. Calcular distancias euclidianas a la infraestructura (mismas 3 que usa el modelo)
    # El transform[0] da el ancho del píxel en metros para convertir distancias a unidades reales.
    pixel_size_meters = transform[0]
    print("Calculando distancias a infraestructura de red...")

    dist_transmision_data = calcular_distancia_a_capa(
        paths_raw['vectores']['lineas'], crs, grid_shape, transform,
        pixel_size_meters, regiones_norte, nombre="líneas de transmisión")

    dist_almacen_data = calcular_distancia_a_capa(
        paths_raw['vectores']['almacenamiento'], crs, grid_shape, transform,
        pixel_size_meters, regiones_norte, nombre="almacenamiento de energía")

    dist_subestaciones_data = calcular_distancia_a_capa(
        paths_raw['vectores']['subestaciones'], crs, grid_shape, transform,
        pixel_size_meters, regiones_norte, nombre="subestaciones")

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
        if not path:
            print(f"  [AVISO] Exclusión '{key}' no tiene ruta configurada en config.yaml "
                  "(paths.raw.vectores); el mapa se genera SIN aplicar esta restricción.")
            continue
        if not os.path.exists(path):
            print(f"  [AVISO] Exclusión '{key}' apunta a '{path}', que no existe; el mapa se "
                  "genera SIN aplicar esta restricción.")
            continue
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