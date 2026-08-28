import os
import site

def _asegurar_proj_lib_test():
    if 'PROJ_LIB' not in os.environ:
        try:
            import rasterio
            proj_dir = os.path.join(os.path.dirname(rasterio.__file__), 'proj_data')
            if os.path.isdir(proj_dir):
                os.environ['PROJ_LIB'] = proj_dir
        except Exception:
            pass

_asegurar_proj_lib_test()
