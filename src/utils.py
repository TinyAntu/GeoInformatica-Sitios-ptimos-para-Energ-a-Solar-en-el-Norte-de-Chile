import os

def _esta_actualizado(paths_entrada: list, paths_salida: list) -> bool:
    """Retorna True solo si todos los outputs existen Y son más recientes que todos los inputs."""
    if not paths_salida or any(not os.path.exists(p) for p in paths_salida):
        return False
    if not paths_entrada or any(not os.path.exists(p) for p in paths_entrada):
        return False
    tiempo_entrada_max = max(os.path.getmtime(p) for p in paths_entrada)
    tiempo_salida_min = min(os.path.getmtime(p) for p in paths_salida)
    return tiempo_salida_min > tiempo_entrada_max
