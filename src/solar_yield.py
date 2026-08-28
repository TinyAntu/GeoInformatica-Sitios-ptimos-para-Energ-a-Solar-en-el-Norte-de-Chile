"""Integración con el motor Rust `solarpv-rs` (mapa de rendimiento fotovoltaico).

Nuestro modelo (RF + AHP) responde *dónde* es apto instalar una planta solar; este
módulo agrega la segunda capa que pide la PEP2: *cuánto produce* cada sitio, en unidades
físicas reales (kWh/kWp/año). No reimplementamos la física PV en Python: invocamos el
binario `solarpv-cli` (validado contra pvlib a ≤0.2 %) vía subprocess y consumimos sus
GeoTIFF de salida.

El motor emite en EPSG:32719 (UTM 19S) sobre el mismo DEM del proyecto, por lo que su
salida `*_specific_yield.tif` es directamente comparable, celda a celda, con nuestro
`data/results/mapa_probabilidad_aptitud.tif`. Antes de devolver la ruta validamos que la
salida efectivamente esté en ese CRS y dentro de la extensión de Antofagasta/Atacama
(ver AGENTS.md, sección 6).
"""

import os
import shutil
import subprocess


class MotorNoDisponibleError(RuntimeError):
    """El binario `solarpv-cli` no se encontró o no es ejecutable."""


class SalidaMotorInvalidaError(RuntimeError):
    """La salida del motor no cumple el CRS o la extensión esperados del proyecto."""


def _resolver_binario(binario: str) -> str:
    """Devuelve la ruta absoluta al ejecutable del motor o lanza un error accionable.

    Acepta tanto una ruta a un archivo (p. ej. 'solarpv-rs/target/release/solarpv-cli')
    como un comando en el PATH (p. ej. 'solarpv-cli').
    """
    # Caso 1: ruta directa a un archivo ejecutable existente.
    if os.path.isfile(binario) and os.access(binario, os.X_OK):
        return os.path.abspath(binario)

    # Caso 2: comando disponible en el PATH.
    encontrado = shutil.which(binario)
    if encontrado:
        return encontrado

    raise MotorNoDisponibleError(
        f"No se encontró el binario del motor solarpv-rs en '{binario}'.\n"
        "Compílalo con:\n"
        "    git clone https://github.com/franciscoparrao/solarpv-rs.git\n"
        "    cd solarpv-rs && cargo build --release -p solarpv-cli --features terrain\n"
        "y ajusta 'solarpv.binario' en config.yaml a la ruta del ejecutable "
        "(por defecto solarpv-rs/target/release/solarpv-cli)."
    )


def construir_comando(
    binario: str,
    dem_path: str,
    out_prefix: str,
    lat: float,
    lon: float,
    date: str,
    mount: str = "tilt",
    tilt=None,
    surface_azimuth=None,
    gcr=None,
    per_cell_lat: bool = True,
    svf: bool = True,
) -> list:
    """Arma la lista de argumentos para `solarpv-cli` (montaje fijo o seguidor).

    Se separa de la ejecución para poder testearla sin correr el binario real y para
    poder imprimirla en modo --dry-run. Los flags de terreno `--per-cell-lat` y `--svf`
    (sombreado por horizonte) van por defecto: son los que hacen al motor consciente del
    relieve, que es justamente el valor que aporta sobre el GHI plano.
    """
    cmd = [
        binario,
        "--dem", dem_path,
        "--lat", str(lat),
        "--lon", str(lon),
        "--date", date,
        "--annual",
        "--out-prefix", out_prefix,
    ]
    if per_cell_lat:
        cmd.append("--per-cell-lat")
    if svf:
        cmd.append("--svf")

    if mount == "tilt":
        cmd += ["--mount", "tilt"]
        if tilt is not None:
            cmd += ["--tilt", str(tilt)]
        if surface_azimuth is not None:
            cmd += ["--surface-azimuth", str(surface_azimuth)]
    elif mount == "tracker":
        cmd += ["--mount", "tracker"]
        if gcr is not None:
            cmd += ["--gcr", str(gcr)]
    else:
        raise ValueError(f"Montaje no soportado: '{mount}'. Usa 'tilt' o 'tracker'.")

    return cmd


def _remuestrear_dem(dem_path: str, resolucion_m: float) -> str:
    """Remuestrea el DEM a `resolucion_m` (bilinear) en un archivo temporal y lo devuelve.

    El DEM nativo (~90 m, 86 M celdas) hace que el motor tarde ~1 h; remuestrear a una
    resolución más gruesa controla runtime y memoria, a costa de detalle de terreno.
    Mismo patrón que `salida_mapa.resolucion_m` del pipeline de aptitud.
    """
    import tempfile
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    with rasterio.open(dem_path) as src:
        b = src.bounds
        width = max(1, int(round((b.right - b.left) / resolucion_m)))
        height = max(1, int(round((b.top - b.bottom) / resolucion_m)))
        transform = from_origin(b.left, b.top, resolucion_m, resolucion_m)
        data = src.read(1, out_shape=(height, width), resampling=Resampling.bilinear)
        meta = src.meta.copy()
        meta.update(width=width, height=height, transform=transform)

    out = os.path.join(tempfile.gettempdir(), f"dem_solarpv_{int(resolucion_m)}m.tif")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(data, 1)
    print(f"  DEM remuestreado a {resolucion_m:.0f} m: {width}x{height} = {width*height:,} celdas -> {out}")
    return out


def _reestampar_crs(path: str, target_srid: int) -> bool:
    """Asigna EPSG:target_srid al raster si perdió el código EPSG (devuelve True si actuó).

    El motor solarpv-rs preserva la grilla del DEM (shape/transform idénticos) pero
    serializa el CRS como `LOCAL_CS` sin autoridad EPSG. Como el DEM de entrada del
    proyecto siempre está en EPSG:target_srid (ver AGENTS.md, invariante de CRS) y la
    grilla no cambia, re-estampar la autoridad es seguro y no reproyecta ningún píxel.
    """
    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(path) as src:
        if src.crs:
            epsg = src.crs.to_epsg()
            if epsg == target_srid or (src.crs.is_projected and ("19S" in str(src.crs) or str(target_srid) in str(src.crs))):
                return False

    try:
        new_crs = CRS.from_epsg(target_srid)
    except Exception:
        if target_srid == 32719:
            new_crs = CRS.from_dict({'proj': 'utm', 'zone': 19, 'south': True, 'datum': 'WGS84', 'units': 'm'})
        else:
            new_crs = CRS.from_string(f"EPSG:{target_srid}")

    with rasterio.open(path, "r+") as dst:
        dst.crs = new_crs
    return True


def _validar_raster_salida(path: str, target_srid: int, bounds_utm: dict) -> None:
    """Verifica que el .tif del motor esté en el CRS y la extensión del proyecto.

    - CRS: debe ser EPSG:target_srid (32719 por defecto).
    - Extensión: los límites del raster deben solaparse con la caja UTM de
      Antofagasta/Atacama declarada en config.yaml (postgis_validation.bounds_utm).
    Si algo no calza, lanza SalidaMotorInvalidaError con el detalle: preferimos fallar en
    voz alta antes de cruzar por error un raster mal georreferenciado con la aptitud.
    """
    import rasterio  # import perezoso: permite testear construir_comando sin GDAL/rasterio

    with rasterio.open(path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        es_valido = (
            epsg == target_srid
            or (src.crs and src.crs.is_projected and (str(target_srid) in str(src.crs) or "19S" in str(src.crs)))
        )
        if not es_valido:
            raise SalidaMotorInvalidaError(
                f"El raster '{path}' está en EPSG:{epsg}, se esperaba EPSG:{target_srid}. "
                "El motor debe emitir en UTM 19S para cruzarse celda a celda con la aptitud."
            )

        b = src.bounds
        # Sin solape en X o en Y => el raster cae fuera de la zona de estudio.
        sin_solape = (
            b.right < bounds_utm["min_x"] or b.left > bounds_utm["max_x"] or
            b.top < bounds_utm["min_y"] or b.bottom > bounds_utm["max_y"]
        )
        if sin_solape:
            raise SalidaMotorInvalidaError(
                f"La extensión del raster '{path}' ({tuple(round(v, 1) for v in b)}) "
                "no solapa con la zona de estudio Antofagasta/Atacama "
                f"({bounds_utm}). Revisa el DEM de entrada."
            )


def generar_mapa_rendimiento(
    dem_path: str,
    out_prefix: str,
    lat: float,
    lon: float,
    date: str,
    binario: str = "solarpv-rs/target/release/solarpv-cli",
    mount: str = "tilt",
    tilt=None,
    surface_azimuth=None,
    gcr=None,
    per_cell_lat: bool = True,
    svf: bool = True,
    target_srid: int = 32719,
    bounds_utm: dict = None,
    resolucion_m: float = None,
    dry_run: bool = False,
) -> str:
    """Corre `solarpv-cli` sobre `dem_path` y devuelve la ruta al `*_specific_yield.tif`.

    Escribe `{out_prefix}_{poa,ac,specific_yield}.tif`; nos interesa el specific_yield
    (kWh/kWp/año por celda). En modo `dry_run` solo imprime el comando y no ejecuta nada
    (útil cuando el motor aún no está compilado en este entorno).

    Lanza MotorNoDisponibleError si el binario no existe, o SalidaMotorInvalidaError si la
    salida no está en EPSG:target_srid dentro de la zona de estudio.
    """
    if not dry_run and not os.path.isfile(dem_path):
        raise FileNotFoundError(
            f"No se encontró el DEM en '{dem_path}'. Ejecuta primero el pipeline para "
            "generar data/processed/dem_norte_32719.tif (o descarga los datos, ver README)."
        )

    salida_esperada = f"{out_prefix}_specific_yield.tif"
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    if dry_run:
        # En dry-run no remuestreamos (sin efectos secundarios): mostramos el comando base.
        cmd = construir_comando(
            binario, dem_path, out_prefix, lat, lon, date,
            mount=mount, tilt=tilt, surface_azimuth=surface_azimuth, gcr=gcr,
            per_cell_lat=per_cell_lat, svf=svf,
        )
        if resolucion_m:
            print(f"[dry-run] (se remuestrearía el DEM a {resolucion_m:.0f} m antes de correr)")
        print("[dry-run] Comando que se ejecutaría:\n    " + " ".join(cmd))
        return salida_esperada

    # Remuestreo opcional del DEM para controlar runtime/memoria (ver _remuestrear_dem).
    dem_efectivo = _remuestrear_dem(dem_path, resolucion_m) if resolucion_m else dem_path

    binario_abs = _resolver_binario(binario)
    cmd = construir_comando(
        binario_abs, dem_efectivo, out_prefix, lat, lon, date,
        mount=mount, tilt=tilt, surface_azimuth=surface_azimuth, gcr=gcr,
        per_cell_lat=per_cell_lat, svf=svf,
    )

    print(f"Ejecutando solarpv-cli (montaje '{mount}')...\n    " + " ".join(cmd))
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"El motor solarpv-cli falló (código {resultado.returncode}).\n"
            f"--- stdout ---\n{resultado.stdout}\n--- stderr ---\n{resultado.stderr}"
        )

    if not os.path.isfile(salida_esperada):
        raise SalidaMotorInvalidaError(
            f"El motor terminó sin error pero no se encontró la salida esperada "
            f"'{salida_esperada}'. Revisa el --out-prefix y la versión del motor."
        )

    # El motor pierde el código EPSG (emite LOCAL_CS): re-estampamos el CRS conocido.
    if _reestampar_crs(salida_esperada, target_srid):
        print(f"  [CRS] El motor emitió un CRS local sin EPSG; re-estampado a "
              f"EPSG:{target_srid} (misma grilla, sin reproyectar).")

    if bounds_utm is not None:
        _validar_raster_salida(salida_esperada, target_srid, bounds_utm)

    print(f"  [OK] Mapa de rendimiento generado y validado: {salida_esperada}")
    return salida_esperada
