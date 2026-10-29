"""Integración con el motor Rust `solarpv-rs` (mapa de rendimiento fotovoltaico).

Nuestro modelo (RF + AHP) responde *dónde* es apto instalar una planta solar; este
módulo agrega la segunda capa que pide la PEP2: *cuánto produce* cada sitio, en unidades
físicas reales (kWh/kWp/año). No reimplementamos la física PV en Python: invocamos el
binario `solarpv-cli` (validado contra pvlib a ≤0.2 %) vía subprocess y consumimos sus
GeoTIFF de salida.

El motor emite en EPSG:32719 (UTM 19S) sobre el mismo DEM del proyecto, por lo que su
salida `*_specific_yield.tif` es directamente comparable, celda a celda, con nuestro
`data/results/aptitud/mapa_probabilidad_aptitud.tif`. Antes de devolver la ruta validamos que la
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


class MotorSinCorreccionTileError(RuntimeError):
    """El binario no tiene la corrección de `--tile` con `--ghi` (solarpv-rs v0.4.0 original)."""


# Parche que corrige `--tile` con `--ghi` en solarpv-rs v0.4.0 (ver README, sección del motor).
PARCHE_TILE_GHI = "parches/solarpv-rs_tile_ghi_origen.patch"

# Binarios que ya pasaron `verificar_tile_con_ghi` en este proceso.
_BINARIOS_VERIFICADOS = set()


def _resolver_binario(binario: str) -> str:
    """Devuelve la ruta absoluta al ejecutable del motor o lanza un error accionable.

    Acepta tanto una ruta a un archivo (p. ej. 'solarpv-rs/target/release/solarpv')
    como un comando en el PATH (p. ej. 'solarpv').
    """
    candidatos = [binario]
    if os.name == 'nt' and not binario.lower().endswith('.exe'):
        candidatos.append(binario + '.exe')

    for cand in candidatos:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # Caso 2: comando disponible en el PATH.
    encontrado = shutil.which(binario)
    if encontrado:
        return encontrado

    raise MotorNoDisponibleError(
        f"No se encontró el binario del motor solarpv-rs en '{binario}'.\n"
        "Compílalo con:\n"
        "    git clone https://github.com/franciscoparrao/solarpv-rs.git\n"
        "    cd solarpv-rs && cargo build --release -p solarpv-cli\n"
        "y ajusta 'solarpv.binario' en config.yaml a la ruta del ejecutable "
        "(por defecto solarpv-rs/target/release/solarpv)."
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
    tile: int = None,
    horizon_radius: int = None,
    ghi: str = None,
    ghi_monthly: str = None,
    ghi_unit: str = "kwh",
) -> list:
    """Arma la lista de argumentos para `solarpv` (montaje fijo o seguidor).

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
    if tile is not None and int(tile) > 0:
        cmd += ["--tile", str(tile)]
    if horizon_radius is not None:
        cmd += ["--horizon-radius", str(horizon_radius)]

    if ghi_monthly:
        cmd += ["--ghi-monthly", ghi_monthly]
        if ghi_unit:
            cmd += ["--ghi-unit", ghi_unit]
    elif ghi:
        cmd += ["--ghi", ghi]
        if ghi_unit:
            cmd += ["--ghi-unit", ghi_unit]

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


def verificar_tile_con_ghi(binario: str) -> None:
    """Comprueba que el binario lea bien el GHI observado cuando tesela (`--tile` + `--ghi`).

    En solarpv-rs v0.4.0 cada tesela lee el GHI con sus índices locales, aunque el GHI está
    alineado al DEM completo: todas las teselas leen la esquina noroeste y, en nuestro DEM
    (mar sin dato), el rendimiento sale 0; donde la esquina sí tiene GHI, sale con la
    nubosidad de otra zona. El parche `PARCHE_TILE_GHI` lo corrige. Como el binario no expone
    si está parchado, se prueba: un DEM sintético de 40×40 celdas con un GHI en gradiente y
    un bloque sin dato en la esquina noroeste (como el mar) se corre con y sin `--tile`. Con
    la corrección ambas salidas coinciden; sin ella, celdas con GHI salen en 0 o cambian. Son
    pocos segundos frente a la hora que tarda la corrida real.
    """
    import tempfile
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    n, bloque_sin_dato, tesela = 40, 8, 16
    perfil = {'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'width': n, 'height': n,
              'crs': 'EPSG:32719', 'transform': from_origin(400000, 7400000, 90, 90)}
    filas, columnas = np.mgrid[0:n, 0:n]
    ghi = (5.0 + 0.05 * filas + 0.03 * columnas).astype('float32')   # kWh/m²/día
    ghi[:bloque_sin_dato, :bloque_sin_dato] = np.nan
    dem = (1000.0 + 5.0 * columnas).astype('float32')                  # ladera suave

    with tempfile.TemporaryDirectory(prefix='solarpv_verificacion_') as tmp:
        rutas = {'dem': os.path.join(tmp, 'dem.tif'), 'ghi': os.path.join(tmp, 'ghi.tif')}
        for nombre, datos in (('dem', dem), ('ghi', ghi)):
            with rasterio.open(rutas[nombre], 'w', **perfil, nodata=np.nan) as dst:
                dst.write(datos, 1)

        salidas = {}
        for etiqueta, tile in (('sin_tile', None), ('con_tile', tesela)):
            prefijo = os.path.join(tmp, etiqueta)
            cmd = construir_comando(binario, rutas['dem'], prefijo, lat=-23.6, lon=-69.5,
                                    date="2026-01-01", mount="tilt", tilt=23, tile=tile,
                                    horizon_radius=4, ghi=rutas['ghi'], ghi_unit="kwh")
            resultado = subprocess.run(cmd, capture_output=True, text=True)
            if resultado.returncode != 0:
                raise RuntimeError(f"El motor falló en la verificación de --tile con --ghi "
                                   f"(código {resultado.returncode}).\n{resultado.stderr}")
            with rasterio.open(f"{prefijo}_specific_yield.tif") as src:
                salidas[etiqueta] = src.read(1).astype('float64')

    con_ghi = np.isfinite(ghi)
    referencia, teselado = salidas['sin_tile'][con_ghi], salidas['con_tile'][con_ghi]
    # La comparación solo vale si la referencia es válida en todas las celdas con GHI: un NaN
    # haría que np.max devolviera NaN y la condición de abajo nunca fallara.
    if not (np.isfinite(referencia).all() and (referencia > 0).all()):
        raise RuntimeError("La verificación de --tile no es concluyente: la corrida sin --tile "
                           "dejó celdas con GHI sin rendimiento. Revisa la versión del motor.")
    dif_max = float(np.max(np.abs(teselado - referencia) / referencia))
    if not np.isfinite(teselado).all() or (teselado <= 0).any() or dif_max > 1e-6:
        raise MotorSinCorreccionTileError(
            f"El motor en '{binario}' no lee bien el GHI con --tile (solarpv-rs v0.4.0 sin "
            f"corregir): {int((teselado <= 0).sum())} celdas con GHI salieron en 0 y la "
            f"diferencia máxima frente a la corrida sin --tile es {dif_max:.2%}.\n"
            "Aplica el parche y recompila:\n"
            f"    cd solarpv-rs && git apply ../{PARCHE_TILE_GHI}\n"
            "    cargo build --release -p solarpv-cli\n"
            "(o deja 'solarpv.tile' en 0 si el DEM cabe en memoria sin teselar)."
        )
    _BINARIOS_VERIFICADOS.add(binario)
    print("  [OK] El motor lee bien el GHI con --tile (corrección aplicada).")


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


def ghi_a_kwh_dia(ghi):
    """Devuelve el GHI en kWh/m²/día (lo que espera `--ghi-unit kwh`) y la unidad detectada.

    Indicación del profesor: sobre el desierto un GHI medio diario vale ~6-8 kWh/m²/día y
    uno anual ~2.000-2.900 kWh/m²/año. Se decide por la mediana de las celdas con dato: el
    anual se divide por 365; otra escala (W/m², MJ/m²) se rechaza en vez de correr el motor
    con unidades equivocadas, porque el error no se notaría en la salida.
    """
    import numpy as np

    mediana = float(np.nanmedian(ghi))
    if 1.0 <= mediana <= 12.0:
        return ghi, "kWh/m²/día"
    if 365.0 <= mediana <= 4400.0:
        return ghi / 365.0, "kWh/m²/año (dividido por 365)"
    raise ValueError(f"GHI con mediana {mediana:.2f}: no parece kWh/m²/día (~6-8 en el desierto) "
                     f"ni kWh/m²/año (~2.000-2.900). Revisa las unidades del ráster de origen.")


def alinear_ghi_a_dem(ghi_origen: str, dem_path: str, salida: str) -> str:
    """Remuestrea el GHI observado (Explorador Solar) a la grilla exacta del DEM.

    El motor exige que `--ghi` calce celda a celda con el DEM, así que se reproyecta a su
    CRS (EPSG:32719), transform y tamaño, con remuestreo bilineal: el GHI es un campo
    continuo y suave (~1 km), a diferencia del aspect. Sale en kWh/m²/día (`ghi_unit: kwh`):
    si el origen viene en kWh/m²/año se divide por 365 (`ghi_a_kwh_dia`). Las celdas sin
    dato o en cero (mar, bordes del producto)
    se escriben como NaN: el motor las deja en rendimiento 0 y los análisis las excluyen.
    Se omite si `salida` ya es más nueva que el GHI de origen y el DEM.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from src.utils import _esta_actualizado

    if _esta_actualizado([ghi_origen, dem_path], [salida]):
        print(f"  [OK] GHI alineado al DEM ya está actualizado: {salida}")
        return salida

    with rasterio.open(ghi_origen) as src:
        ghi = src.read(1).astype(np.float32)
        sin_dato = ~np.isfinite(ghi) | (ghi <= 0)
        if src.nodata is not None:
            sin_dato |= ghi == src.nodata
        ghi[sin_dato] = np.nan
        ghi, unidad = ghi_a_kwh_dia(ghi)
        print(f"  Unidad del GHI de origen: {unidad}")
        src_transform, src_crs = src.transform, src.crs

    with rasterio.open(dem_path) as dem:
        perfil = {'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'nodata': np.nan,
                  'crs': dem.crs, 'transform': dem.transform,
                  'width': dem.width, 'height': dem.height,
                  'compress': 'deflate', 'tiled': True, 'blockxsize': 512, 'blockysize': 512}

    destino = np.full((perfil['height'], perfil['width']), np.nan, dtype=np.float32)
    reproject(source=ghi, destination=destino,
              src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=perfil['transform'], dst_crs=perfil['crs'], dst_nodata=np.nan,
              resampling=Resampling.bilinear)

    os.makedirs(os.path.dirname(salida) or '.', exist_ok=True)
    with rasterio.open(salida, 'w', **perfil) as dst:
        dst.write(destino, 1)
    validos = destino[np.isfinite(destino)]
    print(f"  GHI alineado al DEM ({perfil['width']}x{perfil['height']}): "
          f"mín {validos.min():.2f} | mediana {np.median(validos):.2f} | máx {validos.max():.2f} "
          f"kWh/m²/día | {int((~np.isfinite(destino)).sum()):,} celdas sin dato -> {salida}")
    return salida


def preparar_ghi(cfg: dict, ruta_abs) -> str:
    """Ruta absoluta del GHI alineado que usará el motor, o None si se corre a cielo despejado.

    Si `cfg` declara `ghi_origen`, (re)genera `cfg['ghi']` alineándolo a `cfg['dem']`. Así
    cada zona (norte, Coquimbo) obtiene su propio GHI en la grilla de su DEM.
    """
    if not cfg.get('ghi'):
        return None
    ghi = ruta_abs(cfg['ghi'])
    origen = ruta_abs(cfg['ghi_origen']) if cfg.get('ghi_origen') else None
    if origen and os.path.exists(origen):
        alinear_ghi_a_dem(origen, ruta_abs(cfg['dem']), ghi)
    elif os.path.exists(ghi):
        # El origen se descarga aparte del Explorador Solar (no viene en los datos de Drive):
        # quien ya tiene el GHI alineado puede correr el motor sin él.
        if origen:
            print(f"  [AVISO] No está el GHI de origen ({origen}); se usa el ya alineado: {ghi}")
    else:
        de_origen = f" ni el de origen ({origen})" if origen else ""
        raise FileNotFoundError(
            f"No se encontró el GHI observado: ni el alineado ({ghi}){de_origen}. Descarga "
            "el GHI anual del Explorador Solar (ver README) o quita 'solarpv.ghi' para correr "
            "a cielo despejado.")
    return ghi


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
    binario: str = "solarpv-rs/target/release/solarpv",
    mount: str = "tilt",
    tilt=None,
    surface_azimuth=None,
    gcr=None,
    per_cell_lat: bool = True,
    svf: bool = True,
    tile: int = None,
    horizon_radius: int = None,
    ghi: str = None,
    ghi_monthly: str = None,
    ghi_unit: str = "kwh",
    target_srid: int = 32719,
    bounds_utm: dict = None,
    resolucion_m: float = None,
    dry_run: bool = False,
) -> str:
    """Corre `solarpv` sobre `dem_path` y devuelve la ruta al `*_specific_yield.tif`.

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
            per_cell_lat=per_cell_lat, svf=svf, tile=tile, horizon_radius=horizon_radius,
            ghi=ghi, ghi_monthly=ghi_monthly, ghi_unit=ghi_unit,
        )
        if resolucion_m:
            print(f"[dry-run] (se remuestrearía el DEM a {resolucion_m:.0f} m antes de correr)")
        print("[dry-run] Comando que se ejecutaría:\n    " + " ".join(cmd))
        return salida_esperada

    # Remuestreo opcional del DEM para controlar runtime/memoria (ver _remuestrear_dem).
    dem_efectivo = _remuestrear_dem(dem_path, resolucion_m) if resolucion_m else dem_path

    # El motor exige que el GHI calce celda a celda con el DEM que recibe: si el DEM se
    # remuestreó, el GHI alineado al DEM original ya no calza y se realinea al remuestreado.
    if resolucion_m and ghi:
        ghi = alinear_ghi_a_dem(ghi, dem_efectivo,
                                os.path.splitext(dem_efectivo)[0] + "_ghi.tif")

    binario_abs = _resolver_binario(binario)
    # Con GHI observado y teselas, un binario v0.4.0 sin corregir daría rendimiento 0 o
    # con el GHI de otra zona sin avisar: se comprueba antes de la corrida larga (una vez
    # por binario y proceso: la etapa 10 corre tres montajes seguidos).
    if (ghi or ghi_monthly) and tile is not None and int(tile) > 0 \
            and binario_abs not in _BINARIOS_VERIFICADOS:
        verificar_tile_con_ghi(binario_abs)

    cmd = construir_comando(
        binario_abs, dem_efectivo, out_prefix, lat, lon, date,
        mount=mount, tilt=tilt, surface_azimuth=surface_azimuth, gcr=gcr,
        per_cell_lat=per_cell_lat, svf=svf, tile=tile, horizon_radius=horizon_radius,
        ghi=ghi, ghi_monthly=ghi_monthly, ghi_unit=ghi_unit,
    )

    print(f"Ejecutando solarpv (montaje '{mount}')...\n    " + " ".join(cmd))
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"El motor solarpv falló (código {resultado.returncode}).\n"
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
