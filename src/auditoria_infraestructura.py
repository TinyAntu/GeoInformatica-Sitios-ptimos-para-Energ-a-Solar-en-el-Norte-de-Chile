"""Auditoría de fuga temporal en las distancias a infraestructura (revisión de implementación, C1).

Problema que mide: las capas de subestaciones, líneas de transmisión y almacenamiento del
CEN son una fotografía de 2026. Incluyen infraestructura DEDICADA que se construyó PARA las
propias plantas fotovoltaicas (su subestación elevadora, su línea de conexión y su BESS), a
veces del mismo dueño y en la misma fecha o después. Si el modelo ve esa infraestructura, la distancia de una planta
a la red es casi cero porque la planta existe, no porque el sitio fuera atractivo antes de
construirla (causalidad inversa). Eso puede inflar el AUC, el recall y la dominancia de las
variables de "acceso a red" en SHAP.

La auditoría reentrena el MISMO modelo (mismos 420 puntos, mismas otras variables, mismos
hiperparámetros, `random_state=42`) cambiando solo cómo se mide cada distancia auditada:

  - 'todas'          la capa completa: reproduce la línea base del pipeline.
  - 'sin_dedicadas'  excluye `TIPO == 'DEDICADO'`: regla atemporal, aplicable también al mapa.
  - 'previas'        para cada muestra, solo la infraestructura que operaba al menos
                     `margen_meses` antes de su fecha de referencia. Es la variante causal:
                     "¿estaba la red ahí cuando se decidió construir?".

Las tres distancias se auditan por separado y también juntas (tabla EXPERIMENTOS): al quitar
la fuga de una sola capa, la importancia puede migrar a otra que arrastra la misma fuga.

Fecha de referencia de cada muestra: las positivas usan la fecha de operación de su planta.
Los negativos no tienen fecha (son puntos sintéticos), así que reciben una fecha sorteada de
la distribución empírica de fechas de las plantas. Es el análogo temporal del fondo de grupo
objetivo: compara cada negativo con la red que había en un momento en que efectivamente se
construían plantas, en vez de con la red de hoy.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.features import FEATURES
from src.spatial_validation import (
    asignar_bloques_espaciales,
    spatial_block_cv,
    leave_one_region_out_cv,
    make_objective_spatial,
)

VARIANTES = ('todas', 'sin_dedicadas', 'previas')

# Feature auditada -> clave de su capa en `config.paths.raw.vectores`.
CAPA_POR_FEATURE = {
    'dist_subestaciones': 'subestaciones',
    'dist_transmision': 'lineas',
    'dist_almacen': 'almacenamiento',
}

# Cada experimento declara qué variante usa cada feature auditada; la que no aparece queda
# en 'todas'. Se declaran como datos para que agregar un experimento sea agregar una fila.
_TRES_FEATURES = tuple(CAPA_POR_FEATURE)
EXPERIMENTOS = {
    'linea_base':          {},
    'sub_sin_dedicadas':   {'dist_subestaciones': 'sin_dedicadas'},
    'sub_previas':         {'dist_subestaciones': 'previas'},
    'lin_sin_dedicadas':   {'dist_transmision': 'sin_dedicadas'},
    'lin_previas':         {'dist_transmision': 'previas'},
    'alm_sin_dedicadas':   {'dist_almacen': 'sin_dedicadas'},
    'alm_previas':         {'dist_almacen': 'previas'},
    'ambas_sin_dedicadas': {'dist_subestaciones': 'sin_dedicadas', 'dist_transmision': 'sin_dedicadas'},
    'ambas_previas':       {'dist_subestaciones': 'previas', 'dist_transmision': 'previas'},
    'tres_sin_dedicadas':  {f: 'sin_dedicadas' for f in _TRES_FEATURES},
    'tres_previas':        {f: 'previas' for f in _TRES_FEATURES},
}

# Variables que la revisión agrupa como "acceso a red" por su colinealidad (Spearman ~0,69).
# El almacenamiento no es red de transmisión, pero también es infraestructura con fuga: se
# reporta aparte y dentro del total de infraestructura.
FAMILIA_ACCESO_RED = ('dist_transmision', 'dist_subestaciones')
FAMILIA_INFRAESTRUCTURA = tuple(CAPA_POR_FEATURE)


def variantes_del_experimento(experimento: str) -> dict:
    """Variante de cada feature auditada en un experimento (las no declaradas: 'todas')."""
    declaradas = EXPERIMENTOS[experimento]
    return {feature: declaradas.get(feature, 'todas') for feature in CAPA_POR_FEATURE}


def tiene_mapa(experimento: str) -> bool:
    """Un experimento admite mapa si ninguna feature usa 'previas'.

    'previas' depende de la fecha de cada planta, así que no existe una sola grilla de
    distancias "previa" para todo el territorio.
    """
    return 'previas' not in variantes_del_experimento(experimento).values()


def asignar_tipo_por_subestacion(almacenamiento: gpd.GeoDataFrame,
                                 subestaciones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Agrega a la capa de almacenamiento el `TIPO` de la subestación a la que se conecta.

    A diferencia de subestaciones y líneas, la capa de almacenamiento no trae `TIPO`
    (DEDICADO/ZONAL/NACIONAL). Sí trae `NOMBRE_SE`, el nombre de su subestación de conexión.
    Un BESS conectado a una subestación DEDICADA es, en la práctica, almacenamiento de la
    propia central: se marca DEDICADO para que la regla 'sin_dedicadas' sea la misma en las
    tres capas. Si el nombre no calza con ninguna subestación, queda 'S/I' (no se descarta:
    no hay evidencia de que sea dedicado).
    """
    normalizar = lambda s: s.astype(str).str.upper().str.strip()
    tipo_por_nombre = (subestaciones.assign(_nombre=normalizar(subestaciones['NOMBRE']))
                       .drop_duplicates('_nombre').set_index('_nombre')['TIPO'])
    resultado = almacenamiento.copy()
    resultado['TIPO'] = normalizar(resultado['NOMBRE_SE']).map(tipo_por_nombre).fillna('S/I')
    return resultado


def preparar_infraestructura(capa: gpd.GeoDataFrame, codigos: list) -> gpd.GeoDataFrame:
    """Recorta una capa de infraestructura a la zona de estudio, como el muestreo.

    `src/sampling.py` filtra subestaciones y líneas por la columna REGION contra los códigos
    de la zona (coincidencia exacta): repetirlo aquí garantiza que la variante 'todas'
    reproduzca exactamente las variables del dataset. Agrega `f_op` (fecha de operación como
    datetime, NaT si falta).
    """
    objetivo = {str(c).upper().strip() for c in codigos}
    region = capa['REGION'].astype(str).str.upper().str.strip()
    recorte = capa[region.isin(objetivo)].to_crs(epsg=32719).copy()
    recorte = recorte[recorte.geometry.notna() & ~recorte.geometry.is_empty]
    recorte['f_op'] = _a_fecha_sin_zona(recorte['F_OPERACIO'])
    return recorte.reset_index(drop=True)


def _a_fecha_sin_zona(serie: pd.Series) -> pd.Series:
    """Convierte a datetime sin zona horaria: las capas mezclan fechas con y sin UTC."""
    fechas = pd.to_datetime(serie, errors='coerce', utc=True)
    return fechas.dt.tz_localize(None)


def asignar_fechas_referencia(muestras: gpd.GeoDataFrame, plantas: gpd.GeoDataFrame,
                              random_state: int = 42) -> pd.DataFrame:
    """Fecha de referencia de cada muestra y los datos de la planta asociada (si la hay).

    Las positivas se emparejan con su planta por vecino más cercano: el muestreo tomó el
    centroide de cada planta, así que la distancia de emparejamiento debe ser ~0 m (se
    devuelve para verificarlo). Las positivas sin fecha y todos los negativos reciben una
    fecha sorteada de las fechas reales de las plantas emparejadas.

    Devuelve un DataFrame alineado al índice de `muestras` con las columnas
    `fecha_ref`, `fecha_imputada`, `propiedad_planta` y `dist_emparejamiento_m`.
    """
    plantas = plantas.to_crs(muestras.crs)[['PROPIEDAD', 'F_OPERACIO', 'geometry']].copy()
    plantas['f_planta'] = _a_fecha_sin_zona(plantas['F_OPERACIO'])

    resultado = pd.DataFrame(index=muestras.index)
    resultado['fecha_ref'] = pd.NaT
    resultado['propiedad_planta'] = None
    resultado['dist_emparejamiento_m'] = np.nan

    positivas = muestras[muestras['clase'] == 1][['geometry']]
    if len(positivas):
        pareadas = gpd.sjoin_nearest(positivas, plantas, how='left',
                                     distance_col='dist_emparejamiento_m')
        # Un empate de distancia duplicaría filas: se conserva la primera planta.
        pareadas = pareadas[~pareadas.index.duplicated(keep='first')]
        resultado.loc[pareadas.index, 'fecha_ref'] = pareadas['f_planta']
        resultado.loc[pareadas.index, 'propiedad_planta'] = pareadas['PROPIEDAD']
        resultado.loc[pareadas.index, 'dist_emparejamiento_m'] = pareadas['dist_emparejamiento_m']

    resultado['fecha_ref'] = pd.to_datetime(resultado['fecha_ref'])
    fechas_reales = resultado.loc[muestras['clase'] == 1, 'fecha_ref'].dropna().values
    if len(fechas_reales) == 0:
        raise ValueError("Ninguna planta emparejada tiene F_OPERACIO: no hay fechas que sortear.")

    sin_fecha = resultado['fecha_ref'].isna()
    rng = np.random.default_rng(random_state)
    resultado.loc[sin_fecha, 'fecha_ref'] = rng.choice(fechas_reales, size=int(sin_fecha.sum()))
    resultado['fecha_imputada'] = sin_fecha
    return resultado


def filtrar_infraestructura(capa: gpd.GeoDataFrame, variante: str,
                            fecha_corte=None, margen_meses: int = 12) -> gpd.GeoDataFrame:
    """Subconjunto de la capa que "ve" una muestra según la variante.

    En 'previas' un elemento sin fecha se EXCLUYE: no se puede afirmar que existía antes
    de la planta, y aceptarlo reintroduciría justo la fuga que se quiere medir.
    """
    if variante == 'todas':
        return capa
    if variante == 'sin_dedicadas':
        tipo = capa['TIPO'].astype(str).str.upper().str.strip()
        return capa[tipo != 'DEDICADO']
    if variante == 'previas':
        if fecha_corte is None:
            raise ValueError("La variante 'previas' necesita la fecha de corte de la muestra.")
        limite = pd.Timestamp(fecha_corte) - pd.DateOffset(months=margen_meses)
        return capa[capa['f_op'].notna() & (capa['f_op'] <= limite)]
    raise ValueError(f"Variante desconocida: '{variante}'. Opciones: {VARIANTES}.")


def distancia_a_infraestructura(muestras: gpd.GeoDataFrame, capa: gpd.GeoDataFrame,
                                variante: str, fechas_ref: pd.Series | None = None,
                                margen_meses: int = 12) -> np.ndarray:
    """Distancia euclidiana exacta (vectorial, en metros) de cada muestra a su red visible.

    Mismo cálculo que `src/sampling.py` (`geometry.distance().min()`), válido para puntos
    (subestaciones) y líneas: la única diferencia entre variantes es QUÉ elementos se miran.
    """
    if variante != 'previas':
        visibles = filtrar_infraestructura(capa, variante)
        if len(visibles) == 0:
            raise ValueError(f"La variante '{variante}' dejó la capa vacía.")
        return np.array([visibles.geometry.distance(g).min() for g in muestras.geometry])

    if fechas_ref is None:
        raise ValueError("La variante 'previas' necesita `fechas_ref`.")
    distancias = []
    for idx, geometria in zip(muestras.index, muestras.geometry):
        visibles = filtrar_infraestructura(capa, 'previas', fechas_ref.loc[idx], margen_meses)
        distancias.append(visibles.geometry.distance(geometria).min() if len(visibles) else np.nan)
    return np.array(distancias, dtype=float)


def diagnosticar_fuga(muestras: gpd.GeoDataFrame, capa: gpd.GeoDataFrame,
                      fechas: pd.DataFrame, margen_meses: int = 12) -> dict:
    """Describe, para cada planta con fecha real, cuál es el elemento más cercano de la capa.

    Son las tres huellas de causalidad inversa: que el más cercano sea DEDICADO, que haya
    entrado en operación junto con la planta o después, y que tenga el mismo dueño.
    """
    es_positiva = (muestras['clase'] == 1) & ~fechas['fecha_imputada']
    tipos, contemporaneos, mismo_dueno = [], [], []
    for idx in muestras.index[es_positiva]:
        geometria = muestras.geometry.loc[idx]
        distancias = capa.geometry.distance(geometria)
        cercano = capa.loc[distancias.idxmin()]
        tipos.append(str(cercano['TIPO']).upper().strip())
        limite = fechas.loc[idx, 'fecha_ref'] - pd.DateOffset(months=margen_meses)
        contemporaneos.append(pd.isna(cercano['f_op']) or cercano['f_op'] > limite)
        mismo_dueno.append(str(cercano['PROPIEDAD']).upper().strip()
                           == str(fechas.loc[idx, 'propiedad_planta']).upper().strip())

    n = len(tipos)
    if n == 0:
        return {'n_plantas_con_fecha': 0}
    return {
        'n_plantas_con_fecha': n,
        'margen_meses': margen_meses,
        'pct_mas_cercana_dedicada': round(100.0 * tipos.count('DEDICADO') / n, 1),
        'pct_mas_cercana_no_previa': round(100.0 * sum(contemporaneos) / n, 1),
        'pct_mas_cercana_mismo_dueno': round(100.0 * sum(mismo_dueno) / n, 1),
        'tipos_mas_cercana': pd.Series(tipos).value_counts().to_dict(),
        'nota': ("'no previa' = entró en operación después de (fecha planta - margen) o no "
                 "tiene fecha: no se puede afirmar que existía al decidir la planta."),
    }


def auc_univariado(distancia: np.ndarray, clase: np.ndarray) -> float:
    """AUC de una distancia por sí sola (más cerca = más apto).

    Es el indicador más directo de fuga: si una sola distancia separa las clases casi a la
    perfección, el modelo no necesita "aprender" nada más.
    """
    validas = np.isfinite(distancia)
    return float(roc_auc_score(clase[validas], -distancia[validas]))


def _importancia_shap(muestras: gpd.GeoDataFrame, params_rf: dict, random_state: int) -> dict:
    """Importancia SHAP global (% de media |valor|) del modelo ajustado con las 420 muestras.

    Mismo criterio que `src/explainability.py`: TreeExplainer sobre la clase "apto".
    """
    import shap

    X = muestras[FEATURES]
    modelo = RandomForestClassifier(**params_rf, random_state=random_state, n_jobs=-1)
    modelo.fit(X, muestras['clase'].astype(int))
    valores = shap.TreeExplainer(modelo).shap_values(X)
    valores = valores[:, :, 1] if valores.ndim == 3 else valores
    importancia = np.abs(valores).mean(axis=0)
    porcentaje = 100.0 * importancia / importancia.sum()
    return {f: round(float(p), 2) for f, p in zip(FEATURES, porcentaje)}


def optimizar_hiperparametros(muestras: gpd.GeoDataFrame, optuna_config: dict,
                              tamano_bloque_km: float = 30,
                              random_state: int = 42) -> tuple[dict, float]:
    """Optimiza los hiperparámetros del RF para un experimento, igual que el pipeline.

    Replica `src/modeling.py::optimizar_hiperparametros_optuna` (mismo objetivo: Brier medio
    de SBCV con `make_objective_spatial`; mismo sampler TPE con la misma semilla; mismo
    espacio de búsqueda de `optuna_params`), pero sobre las muestras ya armadas del
    experimento: así lo único que cambia entre experimentos son las distancias auditadas.
    `muestras` debe traer REGION asignada, como en el pipeline.

    Devuelve (params_rf con `class_weight='balanced'`, mejor Brier de la búsqueda).
    """
    import optuna

    con_bloques = asignar_bloques_espaciales(muestras, tamano_bloque_m=tamano_bloque_km * 1000)
    objetivo = make_objective_spatial(con_bloques, list(FEATURES), optuna_config, random_state)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=random_state))
    estudio.optimize(objetivo, n_trials=optuna_config.get('n_trials', 30),
                     show_progress_bar=False)

    params_rf = {**estudio.best_params, 'class_weight': 'balanced'}
    return params_rf, float(estudio.best_value)


def evaluar_variante(muestras: gpd.GeoDataFrame, params_rf: dict,
                     tamano_bloque_km: float = 30, random_state: int = 42) -> dict:
    """Desempeño e importancias de un experimento. `muestras` debe traer REGION asignada.

    Los hiperparámetros NO se reoptimizan: el objetivo es aislar el efecto de las variables,
    y reoptimizar por experimento mezclaría ese efecto con el de la búsqueda de Optuna.
    """
    con_bloques = asignar_bloques_espaciales(muestras, tamano_bloque_m=tamano_bloque_km * 1000)
    sbcv = spatial_block_cv(con_bloques, list(FEATURES), params_rf, n_splits=5,
                            random_state=random_state, tamano_bloque_km=tamano_bloque_km)
    loro = leave_one_region_out_cv(con_bloques, list(FEATURES), params_rf,
                                   random_state=random_state)
    shap_pct = _importancia_shap(muestras, params_rf, random_state)
    ranking_shap = sorted(shap_pct, key=shap_pct.get, reverse=True)
    clase = muestras['clase'].astype(int).values

    return {
        'auc_sbcv_media': round(sbcv['auc_mean'], 4),
        'auc_sbcv_std': round(sbcv['auc_std'], 4),
        'brier_sbcv_media': round(sbcv['brier_mean'], 4),
        'auc_loro_por_region': {r: round(v['auc'], 4) for r, v in loro['folds'].items()},
        'auc_loro_media': round(loro['auc_mean'], 4) if loro['auc_mean'] is not None else None,
        'auc_univariado': {f: round(auc_univariado(muestras[f].values, clase), 4)
                           for f in CAPA_POR_FEATURE},
        'importancia_mdi_sbcv': {d['feature']: round(d['importance_mean'], 4)
                                 for d in sbcv['importancias']},
        'importancia_shap_pct': shap_pct,
        'shap_pct_acceso_red': round(sum(shap_pct[f] for f in FAMILIA_ACCESO_RED), 2),
        'shap_pct_infraestructura': round(sum(shap_pct[f] for f in FAMILIA_INFRAESTRUCTURA), 2),
        'rango_shap': {f: ranking_shap.index(f) + 1 for f in CAPA_POR_FEATURE},
        'variable_mas_importante_shap': ranking_shap[0],
    }


def resumir_distancias(muestras: gpd.GeoDataFrame, feature: str) -> dict:
    """Mediana de una distancia por clase (m): cuánto separa la variable a las clases."""
    clase = muestras['clase'].astype(int)
    d = muestras[feature]
    return {
        'mediana_positivas_m': round(float(d[clase == 1].median()), 1),
        'mediana_negativas_m': round(float(d[clase == 0].median()), 1),
        'n_sin_infraestructura_visible': int(d.isna().sum()),
    }
