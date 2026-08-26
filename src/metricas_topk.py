"""Métricas Recall@K% y Precisión@K% del mapa de aptitud contra las plantas reales.

Cierra los dos umbrales que PEP1 (§6.5) declaró antes de entrenar

Recall@K3 >= 0.85 y Precisión@K1 >= 0.15.

A diferencia del AUC/Brier, que se calculan sobre los 420 puntos del dataset, estas
métricas se calculan sobre el RÁSTER completo: miden el producto cartográfico, no el
clasificador. La pregunta que responden es "si tomo el mejor K% del territorio según el
mapa, ¿qué fracción de la flota solar real capturé y qué fracción de esa área está
realmente ocupada por plantas?".

`K` se interpreta como PORCENTAJE DEL ÁREA de estudio (K1 = top 1%, K3 = top 3%), que es
la única lectura coherente con la justificación de PEP1 (huella de plantas contra los
~180.000 km² de área neta). Con K = "3 sitios" sería imposible capturar el 85% de 105
plantas.
"""

import os

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURES

# Barrido de K por defecto. Incluye obligatoriamente 1 y 3 —los dos que PEP1 comprometió y
# que busca contrastar_con_umbrales_pep1 por nombre de clave— y añade puntos intermedios
# para poder dibujar recall/precisión como curva en función de K, no como dos puntos
# sueltos. El coste por K adicional es marginal: la grilla y la puntuación del modelo se
# calculan una sola vez (ver _CACHE_GRILLA).
KS_DEFECTO = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)

# Los dos K sobre los que PEP1 §6.5 fijó umbrales.
K_RECALL_PEP1 = '3'
K_PRECISION_PEP1 = '1'

# Umbrales declarados en PEP1 §6.5, para contrastar automáticamente.
UMBRAL_RECALL_K3 = 0.85
UMBRAL_PRECISION_K1 = 0.15

REGIONES_ESTUDIO = ['Antofagasta', 'Atacama']


def _ruta(directorio_raiz, ruta):
    """Resuelve una ruta del config contra la raíz del proyecto (patrón del repo)."""
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def cargar_regiones(config, directorio_raiz, crs, regiones_estudio=None):
    """Carga las regiones de la zona de estudio, reproyectadas al CRS de la grilla.

    `regiones_estudio` permite apuntar a otra zona (p. ej. Coquimbo, para medir la
    generalización del modelo). Si es None se toma del config y, en su defecto, de la
    constante REGIONES_ESTUDIO, que conserva el comportamiento previo.
    """
    ruta = _ruta(directorio_raiz, config['paths']['raw']['vectores']['regiones'])
    if regiones_estudio is None:
        regiones_estudio = (config.get('zona_estudio') or {}).get('regiones', REGIONES_ESTUDIO)
    regiones = gpd.read_file(ruta)
    regiones = regiones[regiones['REGION'].isin(regiones_estudio)]
    if len(regiones) == 0:
        raise ValueError(f"Ninguna de las regiones {regiones_estudio} existe en {ruta}.")
    return regiones.to_crs(crs)


def cargar_plantas(config, directorio_raiz, crs, ha_por_mw=None, regiones_estudio=None):
    """Carga las plantas fotovoltaicas recortadas a la zona de estudio.

    `Paneles.gdb` es una capa de PUNTOS (716 registros, área 0): no existe la huella real
    de las plantas en ningún dato del proyecto. Con `ha_por_mw` se reconstruye una huella
    aproximada bufferizando cada punto a un círculo de área proporcional a su potencia
    instalada (columna POTENCIAMW), que sí está completa.

    Rango publicado de ocupación de suelo para FV utility-scale, en ha/MWac:
      1,45  LBNL 2022 (Bolinger et al., 736 plantas 2007-2019), montaje fijo
      2,25  LBNL 2022, seguidor de un eje  <- el montaje relevante en Atacama (ver T3)
      2,95  NREL 2013 (Ong et al.), área directa
      3,60  NREL 2013, área total (incluye caminos, subestación y servidumbres)
    El valor por defecto del proyecto (2,0) queda dentro de ese rango, cerca de la cifra
    LBNL de seguidor. NREL 2013 se cita como cota superior: se basa en módulos de 2012 y la
    densidad de potencia subió 43-52 % entre 2011 y 2019, de ahí la diferencia con LBNL.

    POTENCIAMW es la potencia declarada al SEN, es decir AC: homogénea con las cifras
    /MWac de arriba.

    El supuesto debe declararse siempre en el informe: la precisión de área es
    directamente proporcional a él.
    """
    ruta = _ruta(directorio_raiz, config['paths']['raw']['vectores']['fotovoltaicas'])
    plantas = gpd.read_file(ruta).to_crs(crs)
    plantas['geometry'] = plantas.geometry.make_valid()
    plantas = plantas[plantas.geometry.notna() & ~plantas.geometry.is_empty]

    regiones = cargar_regiones(config, directorio_raiz, crs, regiones_estudio)
    plantas = gpd.clip(plantas, regiones)
    plantas = plantas[plantas.geometry.notna() & ~plantas.geometry.is_empty]
    plantas = plantas.reset_index(drop=True)

    if ha_por_mw:
        potencia = plantas['POTENCIAMW'].astype(float).fillna(0.0).clip(lower=0.0)
        area_m2 = potencia * ha_por_mw * 10_000.0
        radio = np.sqrt(area_m2 / np.pi)
        plantas = plantas.assign(geometry=plantas.geometry.buffer(radio))
        plantas = plantas[plantas.geometry.notna() & ~plantas.geometry.is_empty]
        plantas = plantas.reset_index(drop=True)

    return plantas


def rasterizar_plantas(plantas, grid_shape, transform):
    """Rasteriza la huella de las plantas sobre la grilla.

    IMPORTANTE: `all_touched=False` (criterio por defecto: el centro de la celda debe caer
    dentro del polígono). Con `all_touched=True` —como usa get_rasterized_mask para las
    máscaras de región— una planta pequeña marcaría píxeles completos e inflaría
    artificialmente la precisión, que es justo lo que esta métrica debe medir sin sesgo.
    """
    if len(plantas) == 0:
        return np.zeros(grid_shape, dtype=bool)
    mask = rasterize(
        shapes=plantas.geometry,
        out_shape=grid_shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype='uint8',
    )
    return mask.astype(bool)


def _plantas_a_filas_columnas(plantas, transform, grid_shape):
    """Convierte el centroide de cada planta a (fila, columna) dentro de la grilla.

    El recall se mide por centroide —no por huella— para que una planta cuente una sola
    vez sin importar su tamaño: si no, las plantas grandes pesarían más que las chicas.
    Es además el mismo criterio con que src/sampling.py construyó las muestras positivas.
    """
    alto, ancho = grid_shape
    inverso = ~transform
    filas_columnas = []
    for punto in plantas.geometry.centroid:
        col, fila = inverso * (punto.x, punto.y)
        fila, col = int(fila), int(col)
        if 0 <= fila < alto and 0 <= col < ancho:
            filas_columnas.append((fila, col))
    return filas_columnas


def _precision_por_zonas(top, plantas_en_zona, n_plantas, n_valido, area_px_m2):
    """Precisión a nivel de ZONA CANDIDATA: fracción de zonas que contienen >=1 planta real.

    Por qué no se mide como solapamiento de área: `Paneles.gdb` es una capa de PUNTOS (716
    registros, área total 0), no de polígonos. Sin huella real, una "precisión de área"
    degenera en `recall * n_plantas / n_pixeles_top` —redundante con el recall— y su máximo
    alcanzable (105 / n_pixeles_top ~ 1,3% en el top 1%) queda muy por debajo del umbral de
    0,15 de PEP1, que asumía huellas de ~20 km² inexistentes en los datos.

    A nivel de zona la pregunta recupera sentido operacional: "de las zonas que el mapa
    propone como candidatas, ¿cuántas contienen efectivamente una planta real?".

    La línea base al azar no es una constante: para una zona de área a_i sobre un área total
    A con N plantas repartidas uniformemente, P(la zona contenga >=1) = 1 - (1 - a_i/A)^N.
    Se promedia sobre las zonas, de modo que el enriquecimiento compara contra un azar que
    ya considera el tamaño real de cada zona (una zona grande acierta más por puro tamaño).
    """
    from scipy.ndimage import label

    # 8-conectividad, igual criterio que src/postgis_validation.py al poligonizar.
    etiquetas, n_zonas = label(top, structure=np.ones((3, 3), dtype=int))
    if n_zonas == 0:
        return None

    tamanos = np.bincount(etiquetas.ravel())[1:]  # índice 0 = fondo
    zonas_con_planta = {etiquetas[f, c] for f, c in plantas_en_zona
                        if etiquetas[f, c] > 0}
    precision = len(zonas_con_planta) / n_zonas

    fraccion_area = tamanos / n_valido
    p_azar_por_zona = 1.0 - np.power(1.0 - fraccion_area, n_plantas)
    precision_azar = float(p_azar_por_zona.mean())

    return {
        'n_zonas_candidatas': int(n_zonas),
        'n_zonas_con_planta': len(zonas_con_planta),
        'precision': round(precision, 4),
        'precision_base_azar': round(precision_azar, 6),
        'factor_enriquecimiento': (round(precision / precision_azar, 1)
                                   if precision_azar > 0 else None),
        'area_zona_mediana_km2': round(float(np.median(tamanos)) * area_px_m2 / 1e6, 2),
        'area_zona_maxima_km2': round(float(tamanos.max()) * area_px_m2 / 1e6, 2),
    }


def _precision_por_area(top, mask_buffer, valido, n_valido, n_top, area_px_m2):
    """Precisión como solapamiento de área contra la huella reconstruida de las plantas.

    Es la definición que PEP1 §6.5 tenía en mente, recuperable solo tras bufferizar los
    puntos por potencia instalada (ver cargar_plantas). Se reporta junto al TECHO teórico
    —la precisión que lograría un modelo perfecto que metiera toda la huella dentro del
    top-K%— porque ese techo demuestra si el umbral comprometido era alcanzable.

    Con los 7.195,7 MW del catastro sobre ~203.000 km² de área de estudio, el techo en el
    top 1 % va de 5,13 % (1,45 ha/MW, LBNL fijo) a 12,78 % (3,60 ha/MW, NREL área total):
    NINGUNA cifra publicada alcanza el 0,15 exigido por PEP1 §6.5. Haría falta 4,23 ha/MW,
    por encima de todo benchmark existente, y además con captura perfecta. La conclusión
    es por tanto invariante al supuesto, no un artefacto del valor elegido.
    """
    px_huella = int((mask_buffer & valido).sum())
    if n_top == 0 or px_huella == 0:
        return None

    px_acertados = int((top & mask_buffer).sum())
    precision = px_acertados / n_top
    base_azar = px_huella / n_valido
    techo = px_huella / n_top  # modelo perfecto: toda la huella dentro del top-K%

    return {
        'precision': round(precision, 6),
        'precision_base_azar': round(base_azar, 6),
        'precision_techo_modelo_perfecto': round(min(techo, 1.0), 6),
        'pct_del_techo_alcanzado': round(100.0 * precision / techo, 1) if techo else None,
        'factor_enriquecimiento': round(precision / base_azar, 1) if base_azar else None,
        'area_huella_km2': round(px_huella * area_px_m2 / 1e6, 2),
    }


def calcular_metricas_topk(prob, valido, mask_plantas, plantas_rc, area_px_m2,
                           ks=KS_DEFECTO, mask_buffer=None):
    """Calcula recall y precisión por zona candidata para cada K% sobre una grilla.

    `prob` y `valido` son arrays 2D de la misma forma. `plantas_rc` es la lista de
    (fila, columna) de los centroides de plantas. `mask_plantas` se conserva solo para
    reportar cuántos píxeles marca la capa de plantas (control de resolución).
    """
    prob_valido = prob[valido]
    n_valido = int(prob_valido.size)
    if n_valido == 0:
        raise ValueError("No hay píxeles válidos en la grilla: no se puede calcular top-K.")

    plantas_en_zona = [(f, c) for f, c in plantas_rc if valido[f, c]]
    n_plantas = len(plantas_en_zona)
    if n_plantas == 0:
        raise ValueError("Ninguna planta cae dentro de los píxeles válidos de la grilla.")

    px_plantas = int((mask_plantas & valido).sum())

    resultados = {
        'n_pixeles_validos': n_valido,
        'n_plantas_evaluadas': n_plantas,
        'n_pixeles_marcados_por_plantas': px_plantas,
        'por_k': {},
    }

    for k in ks:
        umbral = float(np.percentile(prob_valido, 100.0 - k))
        top = valido & (prob >= umbral)
        n_top = int(top.sum())

        aciertos = sum(1 for f, c in plantas_en_zona if top[f, c])
        recall = aciertos / n_plantas

        resultados['por_k'][f"{k:g}"] = {
            'umbral_probabilidad': round(umbral, 6),
            # El % de área realmente seleccionado puede exceder K: las probabilidades del RF
            # son discretas y muchos píxeles empatan justo en el umbral. Se reporta para que
            # la lectura del recall sea honesta.
            'pct_area_seleccionada': round(100.0 * n_top / n_valido, 3),
            'n_pixeles_top': n_top,
            'recall': round(recall, 4),
            'plantas_capturadas': aciertos,
            'precision_zonas': _precision_por_zonas(top, plantas_en_zona, n_plantas,
                                                    n_valido, area_px_m2),
        }

        if mask_buffer is not None:
            resultados['por_k'][f"{k:g}"]['precision_area'] = _precision_por_area(
                top, mask_buffer, valido, n_valido, n_top, area_px_m2)

    return resultados


# La grilla de features no depende del modelo: construirla implica reproyectar 4 rásters y
# 3 transformadas de distancia, lo más caro de este script. Se cachea por resolución para
# reutilizarla entre las variantes A-500 y los dos folds LORO (5 usos, 1 construcción).
_CACHE_GRILLA = {}


def _grilla_features(config, resolucion_m):
    if resolucion_m not in _CACHE_GRILLA:
        from src.explainability_spatial import _construir_features
        _CACHE_GRILLA[resolucion_m] = _construir_features(config, resolucion_m)
    return _CACHE_GRILLA[resolucion_m]


def _grilla_desde_modelo(config, directorio_raiz, modelo, resolucion_m):
    """Puntúa un modelo sobre la grilla de inferencia y devuelve (prob, valido, ...).

    Reutiliza _construir_features de src/explainability_spatial.py, que ya arma las 7
    features en el orden canónico de FEATURES y restringe la máscara válida a
    Antofagasta+Atacama.
    """
    crs, transform, width, height, valido, X = _grilla_features(config, resolucion_m)
    prob = np.full((height, width), np.nan, dtype=np.float32)
    if X.shape[0]:
        prob[valido] = modelo.predict_proba(X)[:, 1]
    return crs, transform, (height, width), valido, prob


def _mask_region(regiones, nombre_region, grid_shape, transform):
    """Máscara booleana de una región concreta sobre la grilla."""
    una = regiones[regiones['REGION'] == nombre_region]
    if len(una) == 0:
        return np.zeros(grid_shape, dtype=bool)
    mask = rasterize(shapes=una.geometry, out_shape=grid_shape, transform=transform,
                     fill=0, default_value=1, all_touched=True, dtype='uint8')
    return mask.astype(bool)


def evaluar_mapa_entregado(config, directorio_raiz, ks=KS_DEFECTO, ha_por_mw=None):
    """(A-100) Métricas sobre el mapa que efectivamente se entrega, a su resolución nativa.

    Es la cifra del producto final, pero es IN-SAMPLE: las 105 plantas entrenaron el modelo
    que generó este mapa (PEP1 §6.3 reentrena al 100% para la etapa cartográfica).
    """
    ruta_mapa = _ruta(directorio_raiz, 'data/results/mapa_probabilidad_aptitud.tif')
    if not os.path.exists(ruta_mapa):
        raise FileNotFoundError(f"No existe el mapa de aptitud: {ruta_mapa}")

    with rasterio.open(ruta_mapa) as src:
        prob = src.read(1, masked=True)
        transform, crs = src.transform, src.crs
        area_px_m2 = abs(src.transform[0] * src.transform[4])
        grid_shape = (src.height, src.width)

    valido = ~np.ma.getmaskarray(prob) & np.isfinite(prob.filled(np.nan))
    prob = prob.filled(np.nan)

    plantas = cargar_plantas(config, directorio_raiz, crs)
    mask_plantas = rasterizar_plantas(plantas, grid_shape, transform)
    plantas_rc = _plantas_a_filas_columnas(plantas, transform, grid_shape)

    mask_buffer = None
    if ha_por_mw:
        mask_buffer = rasterizar_plantas(
            cargar_plantas(config, directorio_raiz, crs, ha_por_mw), grid_shape, transform)

    res = calcular_metricas_topk(prob, valido, mask_plantas, plantas_rc, area_px_m2, ks,
                                 mask_buffer=mask_buffer)
    res['resolucion_m'] = round(float(np.sqrt(area_px_m2)), 1)
    return res


def evaluar_por_region(config, directorio_raiz, muestras, params_rf, resolucion_m,
                       random_state=42, ks=KS_DEFECTO, modelo_fijo=None, ha_por_mw=None):
    """Evalúa por región, tomando el top-K% DENTRO de cada una.

    Con `modelo_fijo=None` (por defecto) hace LORO: entrena un modelo por región usando las
    muestras de la *otra* y lo evalúa en la excluida. Es la cifra defendible — las plantas
    de Atacama nunca vieron el modelo que las evalúa. Se usa LORO, y no el split 70/30 de
    `src/modeling.py`, porque ese split es ALEATORIO y está inflado por autocorrelación
    espacial, como advierte el propio código (comentario sobre Roberts et al. 2017 /
    Ploton et al. 2020).

    Con `modelo_fijo` se reutiliza el mismo modelo entrenado al 100% en ambas regiones.
    Eso produce el ancla in-sample con la MISMA estructura de zonas que LORO: sin ella, el
    top-K% global (sobre las dos regiones juntas) fragmenta distinto que el top-K% por
    región y la comparación confundiría generalización con geometría de las zonas.
    """
    features = list(FEATURES)
    resultados_por_region = {}
    total_plantas = 0
    aciertos_totales = {f"{k:g}": 0 for k in ks}
    # La precisión por zonas se agrega sumando zonas de ambas regiones (no promediando
    # porcentajes): las regiones aportan cantidades distintas de zonas candidatas.
    zonas_totales = {f"{k:g}": 0 for k in ks}
    zonas_con_planta = {f"{k:g}": 0 for k in ks}
    azar_ponderado = {f"{k:g}": 0.0 for k in ks}
    px_top_totales = {f"{k:g}": 0 for k in ks}
    px_acertados = {f"{k:g}": 0.0 for k in ks}
    px_huella_total = {f"{k:g}": 0.0 for k in ks}

    for region in REGIONES_ESTUDIO:
        if modelo_fijo is not None:
            print(f"  Evaluando modelo completo (in-sample) en '{region}'...")
            modelo = modelo_fijo
            n_entrenamiento = None
        else:
            entrenamiento = muestras[muestras['REGION'] != region].dropna(subset=features)
            if entrenamiento['clase'].nunique() < 2:
                print(f"  [AVISO] Sin ambas clases para entrenar excluyendo {region}; se omite.")
                continue
            print(f"  Entrenando sin '{region}' ({len(entrenamiento)} muestras) "
                  f"y evaluando en '{region}'...")
            modelo = RandomForestClassifier(n_jobs=-1, random_state=random_state, **params_rf)
            modelo.fit(entrenamiento[features], entrenamiento['clase'])
            n_entrenamiento = int(len(entrenamiento))

        crs, transform, grid_shape, valido, prob = _grilla_desde_modelo(
            config, directorio_raiz, modelo, resolucion_m)
        area_px_m2 = abs(transform[0] * transform[4])

        regiones = cargar_regiones(config, directorio_raiz, crs)
        valido_region = valido & _mask_region(regiones, region, grid_shape, transform)

        plantas = cargar_plantas(config, directorio_raiz, crs)
        una_region = regiones[regiones['REGION'] == region]
        plantas_region = gpd.clip(plantas, una_region)
        plantas_region = plantas_region[plantas_region.geometry.notna()
                                        & ~plantas_region.geometry.is_empty]

        mask_plantas = rasterizar_plantas(plantas_region, grid_shape, transform)
        plantas_rc = _plantas_a_filas_columnas(plantas_region, transform, grid_shape)

        mask_buffer = None
        if ha_por_mw:
            buffer_region = gpd.clip(
                cargar_plantas(config, directorio_raiz, crs, ha_por_mw), una_region)
            buffer_region = buffer_region[buffer_region.geometry.notna()
                                          & ~buffer_region.geometry.is_empty]
            mask_buffer = rasterizar_plantas(buffer_region, grid_shape, transform)

        res = calcular_metricas_topk(prob, valido_region, mask_plantas, plantas_rc,
                                     area_px_m2, ks, mask_buffer=mask_buffer)
        res['n_muestras_entrenamiento'] = n_entrenamiento
        resultados_por_region[region] = res

        # Agregación: recall sobre el total de plantas; precisión sumando zonas de ambas
        # regiones (una región con más zonas candidatas debe pesar más en el total).
        total_plantas += res['n_plantas_evaluadas']
        for clave, valores in res['por_k'].items():
            aciertos_totales[clave] += valores['plantas_capturadas']
            zonas = valores.get('precision_zonas')
            if zonas:
                zonas_totales[clave] += zonas['n_zonas_candidatas']
                zonas_con_planta[clave] += zonas['n_zonas_con_planta']
                azar_ponderado[clave] += (zonas['precision_base_azar']
                                          * zonas['n_zonas_candidatas'])
            # La precisión de área se agrega sumando píxeles, no promediando porcentajes:
            # precision = acertados/n_top, así que se reconstruyen ambos numeradores.
            area = valores.get('precision_area')
            if area:
                n_top = valores['n_pixeles_top']
                px_top_totales[clave] += n_top
                px_acertados[clave] += area['precision'] * n_top
                px_huella_total[clave] += area['precision_techo_modelo_perfecto'] * n_top

    if not resultados_por_region:
        raise ValueError("LORO no produjo ningún fold evaluable.")

    combinado = {}
    for clave in aciertos_totales:
        n_zonas = zonas_totales[clave]
        precision = zonas_con_planta[clave] / n_zonas if n_zonas else None
        azar = azar_ponderado[clave] / n_zonas if n_zonas else None
        combinado[clave] = {
            'recall': round(aciertos_totales[clave] / total_plantas, 4) if total_plantas else None,
            'plantas_capturadas': aciertos_totales[clave],
            'precision_zonas': round(precision, 4) if precision is not None else None,
            'precision_base_azar': round(azar, 6) if azar is not None else None,
            # El enriquecimiento es el número comparable entre variantes: la precisión cruda
            # depende de cuánto se fragmenta el top-K% (una zona de 500 km² con una planta
            # cuenta igual que una de 1 km²), y la base al azar ya pondera por ese tamaño.
            'factor_enriquecimiento': (round(precision / azar, 1)
                                       if precision is not None and azar else None),
            'n_zonas_candidatas': n_zonas,
            'n_zonas_con_planta': zonas_con_planta[clave],
        }

        n_top_total = px_top_totales[clave]
        if n_top_total:
            precision_area = px_acertados[clave] / n_top_total
            techo_area = px_huella_total[clave] / n_top_total
            combinado[clave]['precision_area'] = {
                'precision': round(precision_area, 6),
                'precision_techo_modelo_perfecto': round(min(techo_area, 1.0), 6),
                'pct_del_techo_alcanzado': (round(100.0 * precision_area / techo_area, 1)
                                            if techo_area else None),
            }

    return {
        'resolucion_m': resolucion_m,
        'estrategia': ('in_sample_por_region' if modelo_fijo is not None
                       else 'leave_one_region_out'),
        'n_plantas_evaluadas': total_plantas,
        'por_region': resultados_por_region,
        'combinado_por_k': combinado,
    }


def contrastar_con_umbrales_pep1(loro_combinado):
    """Contrasta la cifra defendible (LORO) contra los umbrales declarados en PEP1 §6.5.

    La precisión se contrasta con la definición de ÁREA, que es la que PEP1 tenía en mente.
    Se incluye el techo teórico porque determina si el umbral era siquiera alcanzable: con
    7.196 MW instalados y 2 ha/MW la huella total es ~144 km² sobre ~202.000 km², de modo
    que ni un modelo perfecto pasaría de ~7,1% en el top 1%. Un 'NO CUMPLE' contra un
    umbral inalcanzable no dice nada sobre la calidad del modelo: por eso se reporta además
    qué porcentaje del máximo alcanzable se logró.
    """
    recall_k3 = loro_combinado.get(K_RECALL_PEP1, {}).get('recall')
    area_k1 = loro_combinado.get(K_PRECISION_PEP1, {}).get('precision_area') or {}
    precision_k1 = area_k1.get('precision')
    techo_k1 = area_k1.get('precision_techo_modelo_perfecto')

    return {
        'recall_k3': {
            'valor': recall_k3,
            'umbral_pep1': UMBRAL_RECALL_K3,
            'cumple': (recall_k3 is not None and recall_k3 >= UMBRAL_RECALL_K3),
            'umbral_alcanzable': True,
        },
        'precision_k1': {
            'valor': precision_k1,
            'definicion': 'solapamiento de área con la huella reconstruida por potencia',
            'umbral_pep1': UMBRAL_PRECISION_K1,
            'cumple': (precision_k1 is not None and precision_k1 >= UMBRAL_PRECISION_K1),
            'techo_modelo_perfecto': techo_k1,
            'umbral_alcanzable': (techo_k1 is not None and techo_k1 >= UMBRAL_PRECISION_K1),
            'pct_del_techo_alcanzado': area_k1.get('pct_del_techo_alcanzado'),
        },
        'precision_k1_por_zonas': {
            'valor': loro_combinado.get(K_PRECISION_PEP1, {}).get('precision_zonas'),
            'definicion': 'fracción de zonas candidatas conexas que contienen >=1 planta',
        },
    }
