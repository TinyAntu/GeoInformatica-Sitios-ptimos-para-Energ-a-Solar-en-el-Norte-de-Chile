"""Reproduce el error de solarpv-rs v0.4.0 con `--tile` y `--ghi`, y verifica su corrección.

El error: con `--tile`, el motor recorta el DEM por tesela pero no el GHI observado. Cada
tesela lee el GHI completo con sus índices LOCALES, así que todas leen la esquina noroeste
del ráster. Si esa esquina no tiene dato (mar), el rendimiento sale 0; si lo tiene, cada
tesela usa el GHI de otra zona y el error es de pocos por ciento.

Este script es autónomo a propósito (no importa nada del proyecto), para que cualquiera con
un binario de solarpv-rs pueda reproducirlo. Arma una escena sintética parecida a la costa
de Antofagasta, en EPSG:32719 a 90 m:

  - DEM con relieve, que sube desde la costa hacia el interior (el horizonte importa);
  - GHI en kWh/m²/día con gradiente costa -> interior y una franja de "mar" sin dato al
    oeste, que es justo lo que cada tesela termina leyendo con v0.4.0.

Por cada binario corre el motor sin `--tile` (referencia: no tesela, no hay nada que leer
mal) y con `--tile`, y compara en las celdas con GHI. Un binario está corregido si con
`--tile` ninguna celda con GHI sale en 0 y la diferencia con la referencia queda bajo la
tolerancia (los bordes de tesela ya tenían diferencias de ~0,1 % a cielo despejado).

Escribe en `--salida`: los rásters de entrada y de cada corrida, `resumen.json` y la figura
`reproduccion_bug_tile_ghi.png`. Devuelve 1 si el binario `--corregido` no pasa.

Requisitos: Python 3.10+, numpy, rasterio, matplotlib.

Uso (las rutas son ejemplos: reemplázalas por las de tus binarios):
    python reproducir_bug_tile_ghi.py --original C:/solarpv-v040/target/release/solarpv.exe --corregido C:/solarpv-rs/target/release/solarpv.exe
    python reproducir_bug_tile_ghi.py --original ~/solarpv-rs/target/release/solarpv    # solo muestra el error

En este repositorio: el binario corregido es solarpv-rs/target/release/solarpv y el
original quedó copiado en solarpv-rs/target/v040_original/solarpv (los valores por defecto).
Para compilar el original desde cero: `git worktree add ../solarpv-v040 8f17b9f` dentro de
solarpv-rs y `cargo build --release -p solarpv-cli` en esa carpeta.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

DIRECTORIO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Escena sintética: 256 x 256 celdas de 90 m (≈ 23 km), esquina NO frente a Antofagasta.
LADO = 256
RESOLUCION_M = 90.0
ORIGEN_UTM = (350000.0, 7400000.0)      # x, y de la esquina noroeste (EPSG:32719)
LAT, LON = -23.5, -70.4                  # solo encabezado: --per-cell-lat recalcula por celda
COLUMNAS_MAR = 40                        # franja oeste sin GHI (≈ 3,6 km)

# Opciones del motor: las mismas del proyecto, a escala de la escena.
TESELA = 64                              # 4 x 4 = 16 teselas
RADIO_HORIZONTE = 16                     # celdas; también es el halo de cada tesela
OPCIONES_MOTOR = ['--date', '2026-01-01', '--annual', '--per-cell-lat', '--svf',
                  '--horizon-radius', str(RADIO_HORIZONTE), '--ghi-unit', 'kwh',
                  '--mount', 'tilt', '--tilt', '23', '--surface-azimuth', '0']

COLOR_CERO = '#e8a13a'                   # celdas con GHI cuyo rendimiento salió 0


def escena_sintetica() -> tuple:
    """DEM (m) y GHI (kWh/m²/día, NaN = sin dato) de la escena."""
    filas, columnas = np.mgrid[0:LADO, 0:LADO].astype('float64')
    # Pendiente costa -> interior, con lomajes y un cerro: el sombreado por horizonte
    # (--svf) cambia de celda a celda, como en el DEM real.
    dem = (50.0 + 7.0 * columnas
           + 120.0 * np.sin(2 * np.pi * filas / 70.0) * np.cos(2 * np.pi * columnas / 55.0)
           + 600.0 * np.exp(-((filas - 150.0) ** 2 + (columnas - 170.0) ** 2) / (2 * 25.0 ** 2)))
    # El GHI crece hacia el interior (menos nubosidad costera) y un poco hacia el norte:
    # con un GHI uniforme el error no se vería, porque todas las celdas leerían lo mismo.
    tierra = (columnas - COLUMNAS_MAR) / (LADO - COLUMNAS_MAR)
    ghi = 5.6 + 1.8 * tierra + 0.3 * (1.0 - filas / LADO)
    ghi[:, :COLUMNAS_MAR] = np.nan
    return dem.astype('float32'), ghi.astype('float32')


def escribir_raster(ruta: str, datos: np.ndarray) -> str:
    perfil = {'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'width': LADO, 'height': LADO,
              'crs': CRS.from_epsg(32719), 'nodata': np.nan,
              'transform': from_origin(*ORIGEN_UTM, RESOLUCION_M, RESOLUCION_M)}
    with rasterio.open(ruta, 'w', **perfil) as dst:
        dst.write(datos, 1)
    return ruta


def resolver_binario(ruta: str) -> str:
    """Ruta absoluta del binario; en Windows acepta la ruta sin `.exe`."""
    for candidato in (ruta, ruta + '.exe'):
        if os.path.isfile(candidato):
            return os.path.abspath(candidato)
        if not os.path.isabs(candidato) and os.path.isfile(os.path.join(DIRECTORIO_RAIZ, candidato)):
            return os.path.join(DIRECTORIO_RAIZ, candidato)
    encontrado = shutil.which(ruta)
    if encontrado:
        return encontrado
    raise FileNotFoundError(f"No se encontró el binario de solarpv-rs: {ruta}")


def correr_motor(binario: str, dem: str, ghi: str, prefijo: str, tesela: int = None) -> tuple:
    """Corre el motor y devuelve (rendimiento específico, comando, segundos)."""
    cmd = [binario, '--dem', dem, '--lat', str(LAT), '--lon', str(LON),
           '--out-prefix', prefijo, '--ghi', ghi] + OPCIONES_MOTOR
    if tesela:
        cmd += ['--tile', str(tesela)]
    inicio = time.perf_counter()
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    segundos = time.perf_counter() - inicio
    if resultado.returncode != 0:
        raise RuntimeError(f"El motor falló (código {resultado.returncode}):\n"
                           f"{' '.join(cmd)}\n{resultado.stderr}")
    with rasterio.open(f"{prefijo}_specific_yield.tif") as src:
        return src.read(1).astype('float64'), cmd, segundos


def comparar(con_tile: np.ndarray, sin_tile: np.ndarray, con_ghi: np.ndarray) -> dict:
    """Diferencia de la corrida con --tile frente a la referencia, en las celdas con GHI."""
    a, b = con_tile[con_ghi], sin_tile[con_ghi]
    rel = np.abs(a - b) / b * 100
    return {
        'celdas_con_ghi': int(con_ghi.sum()),
        'celdas_con_ghi_en_0': int((a <= 0).sum()),
        'celdas_identicas_pct': round(float((a == b).mean() * 100), 2),
        'dif_rel_mediana_pct': round(float(np.median(rel)), 4),
        'dif_rel_max_pct': round(float(rel.max()), 4),
    }


def figura(ghi: np.ndarray, corridas: dict, resumen: dict, salida_png: str) -> str:
    """Una fila por binario: GHI, sin --tile, con --tile y la diferencia relativa (%)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    secuencial = LinearSegmentedColormap.from_list(
        'azul', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    divergente = LinearSegmentedColormap.from_list(
        'div', ['#104281', '#3987e5', '#f0efec', '#e34948', '#8f1d1c'])
    for cmap in (secuencial, divergente):
        cmap.set_bad('#bdbcb8')                       # gris = sin dato de GHI
    tinta, tinta_2 = '#1f1f1e', '#5f5e5a'
    plt.rcParams.update({'font.size': 9, 'text.color': tinta, 'axes.labelcolor': tinta_2,
                         'axes.edgecolor': '#d6d5d0', 'figure.facecolor': '#fcfcfb'})

    sin_ghi = ~np.isfinite(ghi)
    mascara = lambda a: np.ma.masked_where(sin_ghi | ~np.isfinite(a), a)
    referencia = next(iter(corridas.values()))['sin_tile']
    vmin, vmax = np.percentile(referencia[~sin_ghi], [2, 98])

    fig, ejes = plt.subplots(len(corridas), 4, figsize=(15, 3.9 * len(corridas)),
                             constrained_layout=True, squeeze=False)
    for fila, (nombre, rend) in enumerate(corridas.items()):
        m = resumen['binarios'][nombre]['con_tile_vs_sin_tile']
        # Escala del error por fila: v0.4.0 se equivoca en decenas de %, el corregido no.
        limite = 30.0 if m['dif_rel_max_pct'] > 5 else max(0.5, resumen['tolerancia_pct'])

        im = ejes[fila, 0].imshow(mascara(ghi), cmap=secuencial, vmin=5.5, vmax=7.7)
        ejes[fila, 0].set_title('GHI observado (entrada)')
        ejes[fila, 0].set_ylabel(resumen['binarios'][nombre]['etiqueta'], fontsize=10, color=tinta)
        fig.colorbar(im, ax=ejes[fila, 0], shrink=0.8, label='kWh/m²/día')

        for j, (clave, titulo) in enumerate((('sin_tile', 'Sin --tile (referencia)'),
                                             ('con_tile', f'Con --tile {TESELA}')), start=1):
            im = ejes[fila, j].imshow(mascara(rend[clave]), cmap=secuencial, vmin=vmin, vmax=vmax)
            ejes[fila, j].set_title(titulo)
            # En la escala de color un 0 se confundiría con "rendimiento bajo": las celdas que
            # tienen GHI y aun así salen en 0 se pintan aparte, que es el síntoma del error.
            ceros = ~sin_ghi & (rend[clave] <= 0)
            if ceros.any():
                ejes[fila, j].imshow(np.ma.masked_where(~ceros, ceros.astype(float)),
                                     cmap=ListedColormap([COLOR_CERO]), vmin=0, vmax=1)
                ejes[fila, j].legend(handles=[Patch(color=COLOR_CERO, label='rendimiento 0 con GHI')],
                                     loc='lower left', fontsize=8, framealpha=0.9)
        fig.colorbar(im, ax=ejes[fila, 1:3], shrink=0.8, label='kWh/kWp/año')

        ref = rend['sin_tile']
        rel = np.full(ref.shape, np.nan)
        ok = ref > 0
        rel[ok] = (rend['con_tile'][ok] - ref[ok]) / ref[ok] * 100
        im = ejes[fila, 3].imshow(mascara(rel), cmap=divergente, vmin=-limite, vmax=limite)
        ejes[fila, 3].set_title(f"Con --tile vs. sin --tile\n"
                                f"máx |Δ| {m['dif_rel_max_pct']:.2f} % · "
                                f"{m['celdas_con_ghi_en_0']:,} celdas con GHI en 0")
        fig.colorbar(im, ax=ejes[fila, 3], shrink=0.8, label=f'% (escala ±{limite:g})')

        for j in (2, 3):                              # bordes de tesela
            for k in range(TESELA, LADO, TESELA):
                ejes[fila, j].axhline(k - 0.5, color=tinta_2, lw=0.6, ls=(0, (3, 3)))
                ejes[fila, j].axvline(k - 0.5, color=tinta_2, lw=0.6, ls=(0, (3, 3)))

    for ax in ejes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f'solarpv-rs con --ghi y --tile {TESELA}: escena sintética de {LADO} × {LADO} '
                 f'celdas a {RESOLUCION_M:g} m (gris = sin GHI; líneas = bordes de tesela)',
                 fontsize=11)
    fig.savefig(salida_png, dpi=130)
    plt.close(fig)
    return salida_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--original', default='solarpv-rs/target/v040_original/solarpv',
                        help='Binario de solarpv-rs v0.4.0 sin corregir (muestra el error)')
    parser.add_argument('--corregido', default='solarpv-rs/target/release/solarpv',
                        help='Binario compilado con la corrección (debe pasar)')
    parser.add_argument('--salida', default='reproduccion_bug_tile_ghi',
                        help='Carpeta de salida (rásters, resumen.json y la figura)')
    parser.add_argument('--tolerancia', type=float, default=0.5,
                        help='Diferencia relativa máxima aceptada con --tile, en %% (def. 0,5)')
    args = parser.parse_args()

    binarios = {}
    for nombre, ruta, etiqueta in (('original', args.original, 'v0.4.0 sin corregir'),
                                   ('corregido', args.corregido, 'Corregido')):
        try:
            binarios[nombre] = (resolver_binario(ruta), etiqueta)
        except FileNotFoundError as e:
            print(f"[AVISO] {e}; se omite '{nombre}'.")
    if not binarios:
        print("ERROR: no se encontró ningún binario. Usa --original y/o --corregido.")
        return 2

    salida = os.path.abspath(args.salida)
    os.makedirs(salida, exist_ok=True)
    dem, ghi = escena_sintetica()
    ruta_dem = escribir_raster(os.path.join(salida, 'dem.tif'), dem)
    ruta_ghi = escribir_raster(os.path.join(salida, 'ghi.tif'), ghi)
    con_ghi_entrada = np.isfinite(ghi)
    print(f"Escena: {LADO} x {LADO} celdas a {RESOLUCION_M:g} m, {TESELA}x{TESELA} por tesela, "
          f"{int(con_ghi_entrada.sum()):,} celdas con GHI.\n")

    resumen = {'escena': {'lado_celdas': LADO, 'resolucion_m': RESOLUCION_M,
                          'tesela': TESELA, 'radio_horizonte': RADIO_HORIZONTE,
                          'columnas_sin_ghi': COLUMNAS_MAR},
               'tolerancia_pct': args.tolerancia, 'binarios': {}}
    corridas = {}
    for nombre, (binario, etiqueta) in binarios.items():
        print(f"== {etiqueta}: {binario}")
        rend = {}
        for clave, tesela in (('sin_tile', None), ('con_tile', TESELA)):
            rend[clave], cmd, segundos = correr_motor(
                binario, ruta_dem, ruta_ghi, os.path.join(salida, f'{nombre}_{clave}'), tesela)
            print(f"   {clave:8s} {segundos:5.1f} s   {' '.join(os.path.basename(c) if os.path.isabs(c) else c for c in cmd)}")
        # Celdas con GHI en las que la referencia sí calculó rendimiento.
        con_ghi = con_ghi_entrada & (rend['sin_tile'] > 0)
        m = comparar(rend['con_tile'], rend['sin_tile'], con_ghi)
        correcto = m['celdas_con_ghi_en_0'] == 0 and m['dif_rel_max_pct'] <= args.tolerancia
        resumen['binarios'][nombre] = {'binario': binario, 'etiqueta': etiqueta,
                                       'con_tile_vs_sin_tile': m,
                                       'veredicto': 'CORRECTO' if correcto else 'CON ERROR'}
        corridas[nombre] = rend
        print(f"   celdas con GHI en 0: {m['celdas_con_ghi_en_0']:,} de {m['celdas_con_ghi']:,} | "
              f"idénticas: {m['celdas_identicas_pct']} % | dif. mediana {m['dif_rel_mediana_pct']} % "
              f"| dif. máx {m['dif_rel_max_pct']} %  ->  {resumen['binarios'][nombre]['veredicto']}\n")

    # La corrección no debe tocar la salida sin --tile: ambos binarios deben coincidir ahí.
    if len(corridas) == 2:
        iguales = bool(np.array_equal(corridas['original']['sin_tile'],
                                      corridas['corregido']['sin_tile'], equal_nan=True))
        resumen['sin_tile_original_igual_a_corregido'] = iguales
        print(f"Sin --tile, original y corregido idénticos bit a bit: {'sí' if iguales else 'NO'}")

    with open(os.path.join(salida, 'resumen.json'), 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    png = figura(ghi, corridas, resumen, os.path.join(salida, 'reproduccion_bug_tile_ghi.png'))
    print(f"\nFigura: {png}\nResumen: {os.path.join(salida, 'resumen.json')}")

    corregido = resumen['binarios'].get('corregido')
    return 1 if corregido and corregido['veredicto'] != 'CORRECTO' else 0


if __name__ == '__main__':
    sys.exit(main())
