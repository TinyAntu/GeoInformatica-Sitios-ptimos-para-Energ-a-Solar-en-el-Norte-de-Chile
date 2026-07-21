"""Definición canónica de las features del modelo.

El ORDEN es un invariante del proyecto: debe coincidir exactamente con el orden
en que se apilan las columnas al predecir el mapa de aptitud
(scripts/generate_suitability_map.py) y con las columnas muestreadas en
src/sampling.py. Centralizar la lista aquí evita que una edición en un archivo
desalinee silenciosamente el `column_stack` de otro.
"""

FEATURES = [
    'slope',
    'ghi',
    'elev',
    'northness',
    'dist_transmision',
    'dist_almacen',
    'dist_subestaciones',
]
