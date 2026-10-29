"""Dónde va cada resultado y cada figura del proyecto (único punto que decide la estructura).

Las raíces salen de `config.yaml > paths.results` (y, para otra zona, de su `dir_resultados` /
`dir_figuras`); este módulo fija las subcarpetas por tema debajo de ellas. Los nombres de
archivo no se tocan: solo cambia la carpeta, para que las referencias del informe y del
manuscrito se sigan reconociendo.

    tema              resultados (data/results/<tema>/)                  figuras (figures/<tema>/)
    ----------------  -------------------------------------------------  ---------------------------------
    modelo            model_rf.pkl y métricas, dataset de entrenamiento,  importancias, ROC, precisión-recall,
                      validación espacial, métricas top-K                 folds SBCV, métricas de validación
    aptitud           mapa RF, perfiles AHP, consenso                     mapas de aptitud y de consenso
    rendimiento       rendimiento del motor (fijo 23°, 0°, seguidor),     mapas de rendimiento, fijo vs. seguidor
                      comparación de montajes
    cruce             aptitud × rendimiento, rendimiento en aptas         mapa del cruce
    explicabilidad    SHAP global y espacial                              resumen y waterfalls SHAP, dominante
    postgis           sitios validados, reporte, script SQL               —
    transferibilidad  solo zonas: métricas, AOA y DI                      mapas de AOA y DI

Una zona (p. ej. Coquimbo) repite los mismos temas dentro de su propia carpeta. Los estudios
que corren fuera del pipeline (auditoría, desacople, ...) van a `analisis/<nombre>/`.
"""

import os

from src.utils import directorio_raiz

TEMAS = ('modelo', 'aptitud', 'rendimiento', 'cruce', 'explicabilidad', 'postgis',
         'transferibilidad')


def _abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _raiz_zona(config: dict, zona, clave: str, defecto: str) -> str:
    """Raíz de la zona de estudio (zona=None) o del bloque `zona` de config.yaml."""
    if not zona:
        return config['paths']['results'][clave]
    bloque = config.get(zona) or {}
    if not bloque:
        raise KeyError(f"config.yaml no tiene el bloque de zona '{zona}'.")
    clave_zona = 'dir_resultados' if clave == 'base' else 'dir_figuras'
    return bloque.get(clave_zona, defecto.format(zona=zona))


def _en_tema(raiz: str, tema: str, nombre: str) -> str:
    if tema not in TEMAS:
        raise ValueError(f"Tema desconocido '{tema}'. Temas válidos: {', '.join(TEMAS)}.")
    carpeta = os.path.join(_abs(raiz), tema)
    os.makedirs(carpeta, exist_ok=True)
    return os.path.join(carpeta, nombre)


def ruta_resultado(config: dict, tema: str, nombre: str, zona: str = None) -> str:
    """Ruta absoluta de un resultado: `<raíz>/<tema>/<nombre>` (crea la carpeta del tema)."""
    return _en_tema(_raiz_zona(config, zona, 'base', 'data/results/{zona}'), tema, nombre)


def ruta_figura(config: dict, tema: str, nombre: str, zona: str = None) -> str:
    """Ruta absoluta de una figura: `<figuras>/<tema>/<nombre>` (crea la carpeta del tema)."""
    return _en_tema(_raiz_zona(config, zona, 'figuras', 'figures/{zona}'), tema, nombre)


def prefijo_rendimiento(config: dict, zona: str = None) -> str:
    """Prefijo de salida del motor: `<raíz>/rendimiento/rendimiento`.

    El motor escribe `{prefijo}{sufijo}_{poa,ac,specific_yield}.tif`, con sufijo `_fijo`,
    `_fijo_tilt0` o `_seguidor` según el montaje.
    """
    return ruta_resultado(config, 'rendimiento', 'rendimiento', zona)


def dir_resultados(config: dict, tema: str, zona: str = None) -> str:
    """Carpeta de resultados de un tema, para las funciones que reciben un directorio."""
    return os.path.dirname(ruta_resultado(config, tema, '_', zona))


def dir_figuras(config: dict, tema: str, zona: str = None) -> str:
    """Carpeta de figuras de un tema, para las funciones que reciben un directorio."""
    return os.path.dirname(ruta_figura(config, tema, '_', zona))


def dir_analisis(config: dict, nombre: str) -> str:
    """Carpeta de un estudio fuera del pipeline: `data/results/analisis/<nombre>/`."""
    carpeta = os.path.join(_abs(config['paths']['results']['analisis']), nombre)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def dir_figuras_analisis(config: dict, nombre: str) -> str:
    """Carpeta de figuras de un estudio fuera del pipeline: `figures/analisis/<nombre>/`."""
    carpeta = os.path.join(_abs(config['paths']['results']['figuras_analisis']), nombre)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta
