import os
import sys

def _asegurar_proj_lib():
    """Garantiza que la variable de entorno PROJ_LIB apunte a la carpeta proj_data de rasterio si está disponible."""
    if 'PROJ_LIB' not in os.environ:
        try:
            import rasterio
            proj_dir = os.path.join(os.path.dirname(rasterio.__file__), 'proj_data')
            if os.path.isdir(proj_dir):
                os.environ['PROJ_LIB'] = proj_dir
        except Exception:
            pass

_asegurar_proj_lib()
