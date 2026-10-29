"""Control de fuga en las capas de infraestructura (subestaciones, líneas y almacenamiento).

Las capas del CEN son una fotografía de 2026 e incluyen infraestructura DEDICADA: la
subestación elevadora, la línea de conexión y el BESS construidos PARA una central
específica. Si el modelo las ve, la distancia de una planta a la red es casi cero porque la
planta existe, no porque el sitio fuera atractivo antes de construirla (causalidad
inversa). La auditoría (data/results/analisis/auditoria_infraestructura/README.md) midió que esa
fuga infla el AUC en ~6 puntos y el recall cartográfico en ~20.

Con `control_fuga.excluir_dedicadas: true` el pipeline completo trabaja con la red sin
infraestructura dedicada. Este módulo es el único lugar que decide qué red "ve" el pipeline:
`rutas_vectores_efectivas(config)` devuelve las rutas de las capas que deben usar el
muestreo, el mapa, los perfiles AHP, SHAP espacial, las métricas y la transferencia.

Las capas filtradas se generan una sola vez (fase 1, `generar_capas_sin_dedicadas`) sobre las
capas NACIONALES completas, conservando REGION, para que la transferencia a otra región
(Coquimbo) use el mismo criterio que el entrenamiento.
"""

import os

import geopandas as gpd

from src.utils import _esta_actualizado, _archivos_capa_vectorial

# Capas sujetas al control de fuga -> nombre del archivo filtrado.
CAPAS_CON_FUGA = ('subestaciones', 'lineas', 'almacenamiento')


def asignar_tipo_por_subestacion(almacenamiento: gpd.GeoDataFrame,
                                 subestaciones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Agrega a la capa de almacenamiento el `TIPO` de la subestación a la que se conecta.

    A diferencia de subestaciones y líneas, la capa de almacenamiento no trae `TIPO`
    (DEDICADO/ZONAL/NACIONAL). Sí trae `NOMBRE_SE`, el nombre de su subestación de conexión.
    Un BESS conectado a una subestación DEDICADA es, en la práctica, almacenamiento de la
    propia central: se marca DEDICADO para que la regla sea la misma en las tres capas. Si el
    nombre no calza con ninguna subestación, queda 'S/I' (no se descarta: no hay evidencia
    de que sea dedicado).
    """
    normalizar = lambda s: s.astype(str).str.upper().str.strip()
    tipo_por_nombre = (subestaciones.assign(_nombre=normalizar(subestaciones['NOMBRE']))
                       .drop_duplicates('_nombre').set_index('_nombre')['TIPO'])
    resultado = almacenamiento.copy()
    resultado['TIPO'] = normalizar(resultado['NOMBRE_SE']).map(tipo_por_nombre).fillna('S/I')
    return resultado


def excluir_dedicadas(capa: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Quita la infraestructura con `TIPO == 'DEDICADO'`; conserva el resto de columnas."""
    tipo = capa['TIPO'].astype(str).str.upper().str.strip()
    return capa[tipo != 'DEDICADO']


def control_fuga_activo(config: dict) -> bool:
    return bool((config.get('control_fuga') or {}).get('excluir_dedicadas', False))


def rutas_capas_sin_dedicadas(config: dict) -> dict:
    """Ruta del .gpkg filtrado de cada capa, en el mismo estilo (relativa/absoluta) que el config."""
    directorio = config['paths']['processed']['infraestructura']
    return {nombre: os.path.join(directorio, f'{nombre}.gpkg') for nombre in CAPAS_CON_FUGA}


def rutas_vectores_efectivas(config: dict) -> dict:
    """Rutas de las capas vectoriales que debe usar el pipeline.

    Es una copia de `paths.raw.vectores`; con el control de fuga activo, subestaciones, líneas
    y almacenamiento apuntan a sus versiones sin infraestructura dedicada. El resto de las
    capas (regiones, plantas, exclusiones) no cambia.
    """
    vectores = dict(config['paths']['raw']['vectores'])
    if control_fuga_activo(config):
        vectores.update(rutas_capas_sin_dedicadas(config))
    return vectores


def generar_capas_sin_dedicadas(config: dict) -> dict:
    """Escribe las capas sin infraestructura dedicada (EPSG:32719, GeoPackage).

    Se omite si los archivos ya existen y son más nuevos que las capas originales (chequeo
    incremental del proyecto). Devuelve {capa: ruta}. `config` debe traer las rutas resueltas.
    """
    vectores = config['paths']['raw']['vectores']
    salidas = rutas_capas_sin_dedicadas(config)
    entradas = [archivo for nombre in CAPAS_CON_FUGA
                for archivo in _archivos_capa_vectorial(vectores[nombre])]
    if _esta_actualizado(entradas, list(salidas.values())):
        print("  Capas sin infraestructura dedicada ya actualizadas; se omite.")
        return salidas

    os.makedirs(os.path.dirname(next(iter(salidas.values()))), exist_ok=True)
    crudas = {nombre: gpd.read_file(vectores[nombre]) for nombre in CAPAS_CON_FUGA}
    # El TIPO del almacenamiento se hereda de la capa NACIONAL de subestaciones, antes de
    # filtrar nada: su subestación de conexión puede estar fuera de la zona de estudio.
    crudas['almacenamiento'] = asignar_tipo_por_subestacion(crudas['almacenamiento'],
                                                            crudas['subestaciones'])
    for nombre, capa in crudas.items():
        filtrada = excluir_dedicadas(capa).to_crs(epsg=32719)
        filtrada.to_file(salidas[nombre], driver='GPKG')
        print(f"  {nombre}: {len(filtrada)} de {len(capa)} elementos no dedicados -> "
              f"{os.path.basename(salidas[nombre])}")
    return salidas
