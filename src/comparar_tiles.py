"""Comparación del teselado del motor solarpv-rs con GHI observado (análisis aparte).

En solarpv-rs v0.4.0, `--tile` junto con `--ghi` lee el GHI con los índices locales de cada
tesela, aunque el GHI está alineado al DEM completo: todas las teselas leen la esquina
noroeste. En nuestro DEM esa esquina es mar sin dato y el rendimiento salía 0. Mientras se
corregía el motor, el pipeline teselaba desde Python; hoy usa el motor corregido
(`parches/solarpv-rs_tile_ghi_origen.patch`) y este módulo conserva la comparación que
respalda ese cambio. Sobre recortes del DEM real compara:

  sin_tile      el motor sin --tile (referencia: no tesela, así que no hay nada que leer mal)
  con_tile      el motor con --tile (con el binario corregido debe coincidir con la referencia)
  python        teselado hecho desde Python: recorta DEM y GHI con la misma ventana y corre
                el motor sin --tile en cada tesela (lo que el motor debería hacer por dentro)
  original      opcional: v0.4.0 sin corregir con --tile, para mostrar el error

No se reimplementa física: toda celda la calcula el motor.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import rasterio
from rasterio.windows import Window

from src.solar_yield import construir_comando

# Salidas que el motor escribe como {prefijo}_{producto}.tif.
PRODUCTOS_MOTOR = ('poa', 'ac', 'specific_yield')


def recortar_raster(fuente: str, ventana: Window, destino: str) -> str:
    """Escribe en `destino` la `ventana` de `fuente`, con su georreferencia."""
    with rasterio.open(fuente) as src:
        perfil = src.profile.copy()
        perfil.update(width=int(ventana.width), height=int(ventana.height),
                      transform=src.window_transform(ventana))
        with rasterio.open(destino, 'w', **perfil) as dst:
            dst.write(src.read(1, window=ventana), 1)
    return destino


def correr_motor(binario: str, dem: str, ghi: str, prefijo: str, tile, opciones: dict) -> str:
    """Corre el motor una vez y devuelve la ruta de su `specific_yield`."""
    cmd = construir_comando(binario, dem, prefijo, ghi=ghi, tile=tile, **opciones)
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(f"El motor falló (código {resultado.returncode}).\n{resultado.stderr}")
    return f"{prefijo}_specific_yield.tif"


def correr_teselado_python(binario: str, dem_path: str, ghi_path: str, out_prefix: str,
                           tile: int, halo: int, opciones: dict) -> str:
    """Tesela desde Python y une los interiores en `{out_prefix}_{producto}.tif`.

    Hace lo mismo que `tiled_energies` del motor, pero recortando DEM y GHI con la misma
    ventana: cada tesela se agranda con un halo de `halo` celdas (el radio del horizonte),
    el motor corre sin `--tile` sobre ese recorte y solo se copia el interior. Las teselas
    cuyo interior no tiene GHI se escriben en 0 sin correr el motor, que es lo que este
    devolvería ahí. Devuelve la ruta del `specific_yield`.
    """
    with rasterio.open(dem_path) as dem:
        filas, columnas = dem.height, dem.width
        perfil_salida = {'driver': 'GTiff', 'dtype': 'float64', 'count': 1,
                         'width': columnas, 'height': filas,
                         'crs': dem.crs, 'transform': dem.transform}

    esquinas = [(r0, c0) for r0 in range(0, filas, tile) for c0 in range(0, columnas, tile)]
    destinos = {p: rasterio.open(f"{out_prefix}_{p}.tif", 'w', **perfil_salida)
                for p in PRODUCTOS_MOTOR}
    try:
        with tempfile.TemporaryDirectory(prefix='solarpv_teselas_') as tmp, \
                rasterio.open(ghi_path) as ghi:
            for r0, c0 in esquinas:
                alto, ancho = min(tile, filas - r0), min(tile, columnas - c0)
                interior = Window(c0, r0, ancho, alto)

                ghi_interior = ghi.read(1, window=interior)
                if not (np.isfinite(ghi_interior) & (ghi_interior > 0)).any():
                    for destino in destinos.values():
                        destino.write(np.zeros((alto, ancho)), 1, window=interior)
                    continue

                er0, er1 = max(r0 - halo, 0), min(r0 + alto + halo, filas)
                ec0, ec1 = max(c0 - halo, 0), min(c0 + ancho + halo, columnas)
                ventana = Window(ec0, er0, ec1 - ec0, er1 - er0)
                dem_tesela = recortar_raster(dem_path, ventana, os.path.join(tmp, 'dem.tif'))
                ghi_tesela = recortar_raster(ghi_path, ventana, os.path.join(tmp, 'ghi.tif'))

                prefijo = os.path.join(tmp, 'tesela')
                correr_motor(binario, dem_tesela, ghi_tesela, prefijo, None, opciones)

                del_interior = Window(c0 - ec0, r0 - er0, ancho, alto)
                for producto, destino in destinos.items():
                    with rasterio.open(f"{prefijo}_{producto}.tif") as src:
                        destino.write(src.read(1, window=del_interior).astype('float64'), 1,
                                      window=interior)
    finally:
        for destino in destinos.values():
            destino.close()
    return f"{out_prefix}_specific_yield.tif"


def metricas(a: np.ndarray, b: np.ndarray, mascara: np.ndarray) -> dict:
    """Cuánto difiere `a` de la referencia `b` en las celdas de `mascara` (con GHI)."""
    rel = np.abs(a[mascara] - b[mascara]) / b[mascara] * 100
    return {
        'celdas_identicas_pct': round(float((a[mascara] == b[mascara]).mean() * 100), 2),
        'dif_rel_mediana_pct': round(float(np.median(rel)), 4),
        'dif_rel_max_pct': round(float(rel.max()), 4),
        'celdas_dif_mayor_0_01pct': int((rel > 0.01).sum()),
        'celdas_con_ghi_en_0': int((a[mascara] <= 0).sum()),
    }


def comparar_recorte(nombre: str, lat: float, lon: float, dem_path: str, ghi_path: str,
                     dir_salida: str, binario: str, opciones: dict, tamano: int, tesela: int,
                     binario_original: str = None) -> tuple:
    """Corre las variantes sobre un recorte de `tamano` celdas con esquina NO en (lat, lon).

    Devuelve (métricas, GHI del recorte, rendimientos por variante).
    """
    from pyproj import Transformer

    with rasterio.open(dem_path) as dem:
        x, y = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True).transform(lon, lat)
        c0, r0 = (int(v) for v in ~dem.transform * (x, y))
        if not (0 <= r0 <= dem.height - tamano and 0 <= c0 <= dem.width - tamano):
            raise ValueError(f"El recorte '{nombre}' ({lat}, {lon}) se sale del DEM.")
    ventana = Window(c0, r0, tamano, tamano)

    d = os.path.join(dir_salida, nombre)
    os.makedirs(d, exist_ok=True)
    dem_recorte = recortar_raster(dem_path, ventana, os.path.join(d, 'dem.tif'))
    ghi_recorte = recortar_raster(ghi_path, ventana, os.path.join(d, 'ghi.tif'))
    halo = int(opciones.get('horizon_radius') or 100)

    rutas = {
        'sin_tile': correr_motor(binario, dem_recorte, ghi_recorte,
                                 os.path.join(d, 'sin_tile'), None, opciones),
        'con_tile': correr_motor(binario, dem_recorte, ghi_recorte,
                                 os.path.join(d, 'con_tile'), tesela, opciones),
        'python': correr_teselado_python(binario, dem_recorte, ghi_recorte,
                                         os.path.join(d, 'python'), tesela, halo, opciones),
    }
    if binario_original:
        rutas['original'] = correr_motor(binario_original, dem_recorte, ghi_recorte,
                                         os.path.join(d, 'original'), tesela, opciones)

    with rasterio.open(ghi_recorte) as g:
        ghi = g.read(1).astype('float64')
    rend = {}
    for clave, ruta in rutas.items():
        with rasterio.open(ruta) as src:
            rend[clave] = src.read(1).astype('float64')
    con_ghi = np.isfinite(ghi) & (ghi > 0) & (rend['sin_tile'] > 0)

    resultado = {
        'recorte': {'lat': lat, 'lon': lon, 'fila': r0, 'columna': c0, 'tamano': tamano},
        'pct_celdas_con_ghi': round(float(con_ghi.mean() * 100), 1),
        'con_tile_vs_sin_tile': metricas(rend['con_tile'], rend['sin_tile'], con_ghi),
        'con_tile_vs_python': metricas(rend['con_tile'], rend['python'], con_ghi),
        'python_vs_sin_tile': metricas(rend['python'], rend['sin_tile'], con_ghi),
    }
    if 'original' in rend:
        resultado['original_vs_sin_tile'] = metricas(rend['original'], rend['sin_tile'], con_ghi)
    return resultado, ghi, rend


def figura_comparacion(datos: dict, resumen: dict, tesela: int, salida_png: str) -> str:
    """Mapas de rendimiento de cada variante y de su diferencia frente al motor sin --tile.

    `datos[nombre] = (ghi, rendimientos)`. Escala secuencial azul para el rendimiento y
    divergente azul–gris–rojo para las diferencias; gris claro = sin dato de GHI.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    secuencial = LinearSegmentedColormap.from_list(
        'azul', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    divergente = LinearSegmentedColormap.from_list(
        'div', ['#104281', '#3987e5', '#f0efec', '#e34948', '#8f1d1c'])
    secuencial.set_bad('#bdbcb8')
    divergente.set_bad('#bdbcb8')
    tinta, tinta_2 = '#1f1f1e', '#5f5e5a'
    plt.rcParams.update({'font.size': 9, 'text.color': tinta, 'axes.labelcolor': tinta_2,
                         'axes.edgecolor': '#d6d5d0', 'figure.facecolor': '#fcfcfb'})

    con_original = all('original' in rend for _, rend in datos.values())
    paneles = [('sin_tile', 'Motor sin --tile\n(referencia)')]
    if con_original:
        paneles.append(('original', f'v0.4.0 original con --tile {tesela}'))
    paneles += [('python', f'Teselado en Python\n(teselas de {tesela} + halo)'),
                ('con_tile', f'Motor corregido con --tile {tesela}')]
    difs = []
    if con_original:
        difs.append(('original', 'v0.4.0 original con --tile\nvs. sin --tile', 30))
    difs.append(('con_tile', 'Corregido con --tile\nvs. sin --tile', 0.15))

    n_col = 1 + len(paneles) + len(difs)
    fig, ejes = plt.subplots(len(datos), n_col, figsize=(3.6 * n_col, 4.0 * len(datos)),
                             constrained_layout=True, squeeze=False)
    for fila, (nombre, (ghi, rend)) in enumerate(datos.items()):
        sin_ghi = ~(np.isfinite(ghi) & (ghi > 0))
        masc = lambda a: np.ma.masked_where(sin_ghi | ~np.isfinite(a), a)

        im = ejes[fila, 0].imshow(masc(ghi), cmap=secuencial, vmin=5.0, vmax=7.2)
        ejes[fila, 0].set_title('GHI observado', fontsize=9)
        ejes[fila, 0].set_ylabel(nombre.replace('_', ' '), fontsize=10, color=tinta)
        fig.colorbar(im, ax=ejes[fila, 0], shrink=0.75, label='kWh/m²/día')

        for j, (clave, titulo) in enumerate(paneles, start=1):
            im = ejes[fila, j].imshow(masc(rend[clave]), cmap=secuencial, vmin=1600, vmax=2250)
            ejes[fila, j].set_title(titulo, fontsize=9)
        fig.colorbar(im, ax=ejes[fila, 1:1 + len(paneles)], shrink=0.75,
                     label='kWh/kWp/año')

        for j, (clave, titulo, lim) in enumerate(difs, start=1 + len(paneles)):
            ref = rend['sin_tile']
            rel = np.full(ref.shape, np.nan)
            ok = ref > 0
            rel[ok] = (rend[clave][ok] - ref[ok]) / ref[ok] * 100
            m = resumen[nombre][f'{clave}_vs_sin_tile']
            im = ejes[fila, j].imshow(masc(rel), cmap=divergente, vmin=-lim, vmax=lim)
            ejes[fila, j].set_title(f"{titulo}\nmáx |Δ| {m['dif_rel_max_pct']:.2f} %", fontsize=9)
            for k in range(tesela, ref.shape[0], tesela):
                ejes[fila, j].axhline(k, color=tinta_2, lw=0.6, ls=(0, (3, 3)))
                ejes[fila, j].axvline(k, color=tinta_2, lw=0.6, ls=(0, (3, 3)))
            fig.colorbar(im, ax=ejes[fila, j], shrink=0.75, label=f'% (±{lim:g})')

    for ax in ejes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('solarpv-rs con --ghi: motor sin --tile, teselado en Python y --tile '
                 '(gris = sin GHI; líneas = bordes de tesela)', fontsize=11)
    fig.savefig(salida_png, dpi=130)
    plt.close(fig)
    return salida_png


def ejecutar_comparacion(recortes: dict, dem_path: str, ghi_path: str, dir_salida: str,
                         binario: str, opciones: dict, tamano: int, tesela: int,
                         binario_original: str = None) -> dict:
    """Corre todos los recortes, escribe `resumen.json` y la figura, y devuelve el resumen."""
    os.makedirs(dir_salida, exist_ok=True)
    resumen, datos = {}, {}
    for nombre, esquina in recortes.items():
        print(f"  Recorte '{nombre}' ({tamano}×{tamano} celdas, --tile {tesela})...")
        resumen[nombre], ghi, rend = comparar_recorte(
            nombre, esquina['lat'], esquina['lon'], dem_path, ghi_path, dir_salida,
            binario, opciones, tamano, tesela, binario_original)
        datos[nombre] = (ghi, rend)

    with open(os.path.join(dir_salida, 'resumen.json'), 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    figura_comparacion(datos, resumen, tesela, os.path.join(dir_salida, 'comparacion_tiles.png'))
    return resumen
