import os
import sys
import subprocess

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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
    ruta_script = os.path.join(directorio_raiz, 'scripts', script)
    cmd = [sys.executable, ruta_script] + (args_extra or [])
    print(f"\n--- {nombre_etapa} ---")
    resultado = subprocess.run(cmd, cwd=directorio_raiz)
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
