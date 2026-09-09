import os
import sys
import subprocess

try:
    import resource  # solo Unix; en Windows se omite el reporte de memoria
except ImportError:
    resource = None

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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


def _ruta_abs(ruta: str) -> str:
    """Resuelve una ruta del config contra la raíz del proyecto."""
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def _resolver_rutas(obj, base_dir: str = directorio_raiz):
    """Convierte recursivamente todas las rutas relativas del config a absolutas."""
    if isinstance(obj, str):
        return os.path.join(base_dir, obj) if not os.path.isabs(obj) else obj
    if isinstance(obj, dict):
        return {k: _resolver_rutas(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolver_rutas(item, base_dir) for item in obj]
    return obj


def _shapefile_paths(shp_path: str) -> list:
    """Las 5 rutas hermanas de un shapefile: ESRI reparte un .shp en varios archivos."""
    base, _ = os.path.splitext(shp_path)
    return [base + ext for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']]


def _pico_memoria_hijos_kb() -> int:
    """Marca de agua de memoria del subproceso más grande lanzado hasta ahora (KiB).

    `ru_maxrss` de RUSAGE_CHILDREN es acumulado: no da el pico de UN hijo, sino el máximo
    histórico entre todos. Por eso quien lo consume solo debe reportar los AUMENTOS, que sí
    identifican a la etapa que acaba de establecer un récord.
    """
    if resource is None:
        return 0
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def _correr_etapa(nombre_etapa: str, script: str, args_extra: list | None = None,
                  motor_best_effort: bool = False) -> bool:
    """Corre `scripts/<script>` como subproceso independiente.

    Cada etapa se ejecuta en su propio proceso (no como import + llamada a `main()` en el
    mismo proceso del orquestador, para evitar problemas de memoria y dependencias
    compartidas).

    Devuelve True si la etapa corrió con éxito (código 0), False si se omitió en modo
    best-effort. Si no es best-effort y falla, aborta el proceso completo.

    Vive aquí, y no en run_pipeline.py, porque scripts/run_zona.py necesita exactamente la
    misma semántica de aborto/best-effort para encadenar la cadena de una zona.
    """
    _asegurar_proj_lib()
    env = os.environ.copy()
    ruta_script = os.path.join(directorio_raiz, 'scripts', script)
    cmd = [sys.executable, ruta_script] + (args_extra or [])
    print(f"\n--- {nombre_etapa} ---")
    pico_antes = _pico_memoria_hijos_kb()
    resultado = subprocess.run(cmd, cwd=directorio_raiz, env=env)
    # Correr cada etapa en su propio proceso es lo que hace que su memoria se devuelva al
    # sistema al terminar (un hilo no lo haría: comparten heap). Si esta etapa estableció un
    # nuevo máximo, se reporta: es el dato que permite ubicar un OOM sin adivinar.
    pico_despues = _pico_memoria_hijos_kb()
    if pico_despues > pico_antes:
        print(f"  [MEM] nuevo pico de memoria en un subproceso: {pico_despues / 1024 / 1024:.2f} GB")
    if resultado.returncode != 0:
        if motor_best_effort:
            print(f"  [AVISO] '{nombre_etapa}' terminó con código {resultado.returncode} "
                  "(probablemente el motor Rust solarpv-rs no está compilado, ver AGENTS.md "
                  "sección 6). Se omite y el proceso continúa con el resto de las etapas.")
            return False
        print(f"ERROR: '{nombre_etapa}' falló con código {resultado.returncode}.")
        sys.exit(resultado.returncode)
    return True


def _esta_actualizado(paths_entrada: list, paths_salida: list) -> bool:
    """Retorna True solo si todos los outputs existen Y son más recientes que todos los inputs."""
    if not paths_salida or any(not os.path.exists(p) for p in paths_salida):
        return False
    if not paths_entrada or any(not os.path.exists(p) for p in paths_entrada):
        return False
    tiempo_entrada_max = max(os.path.getmtime(p) for p in paths_entrada)
    tiempo_salida_min = min(os.path.getmtime(p) for p in paths_salida)
    return tiempo_salida_min > tiempo_entrada_max
