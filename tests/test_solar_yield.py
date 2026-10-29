"""Pruebas del wrapper del motor solarpv-rs (src/solar_yield.py).

No requieren el binario Rust ni GDAL del sistema más allá de rasterio: usan rasters
sintéticos en un directorio temporal y simulan el subprocess con mocks.
"""

import os
import sys
import types
import unittest
import tempfile
from unittest import mock
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
except (ImportError, Exception):
    rasterio = None
    from_origin = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import _asegurar_proj_lib
_asegurar_proj_lib()

from src.solar_yield import (
    construir_comando, generar_mapa_rendimiento, _resolver_binario,
    _reestampar_crs, _validar_raster_salida, alinear_ghi_a_dem, preparar_ghi, ghi_a_kwh_dia,
    MotorNoDisponibleError, SalidaMotorInvalidaError, MotorSinCorreccionTileError,
    _BINARIOS_VERIFICADOS,
)

# Caja UTM de la zona de estudio (misma convención que config.postgis_validation.bounds_utm).
BOUNDS_UTM = {"min_x": 100000.0, "max_x": 900000.0, "min_y": 6000000.0, "max_y": 8500000.0}


def _escribir_raster(path, data, crs="EPSG:32719", origin=(300000, 7010000), res=1000, nodata=-9999.0):
    h, w = data.shape
    transform = from_origin(origin[0], origin[1], res, res)
    if crs == "EPSG:32719":
        crs_obj = rasterio.crs.CRS.from_dict({'proj': 'utm', 'zone': 19, 'south': True, 'datum': 'WGS84', 'units': 'm'})
    elif crs == "EPSG:4326":
        crs_obj = rasterio.crs.CRS.from_dict({'proj': 'longlat', 'datum': 'WGS84'})
    elif crs is not None:
        crs_obj = crs
    else:
        crs_obj = None

    perfil = dict(driver="GTiff", dtype="float32", count=1, width=w, height=h,
                  transform=transform, nodata=nodata)
    if crs_obj is not None:
        perfil["crs"] = crs_obj
    with rasterio.open(path, "w", **perfil) as d:
        d.write(data.astype("float32"), 1)
    return path


class TestConstruirComando(unittest.TestCase):
    def test_fijo(self):
        cmd = construir_comando("bin", "d.tif", "out", -23.6, -69.5, "2026-01-01",
                                mount="tilt", tilt=23, surface_azimuth=0)
        self.assertIn("--mount", cmd)
        self.assertEqual(cmd[cmd.index("--mount") + 1], "tilt")
        self.assertIn("--tilt", cmd)
        self.assertEqual(cmd[cmd.index("--tilt") + 1], "23")
        self.assertIn("--annual", cmd)
        self.assertIn("--per-cell-lat", cmd)
        self.assertIn("--svf", cmd)

    def test_fijo_tilt0(self):
        cmd = construir_comando("bin", "d.tif", "out", -23.6, -69.5, "2026-01-01",
                                mount="tilt", tilt=0, surface_azimuth=0)
        self.assertIn("--mount", cmd)
        self.assertEqual(cmd[cmd.index("--mount") + 1], "tilt")
        self.assertIn("--tilt", cmd)
        self.assertEqual(cmd[cmd.index("--tilt") + 1], "0")

    def test_seguidor(self):
        cmd = construir_comando("bin", "d.tif", "out", -23.6, -69.5, "2026-01-01",
                                mount="tracker", gcr=0.3)
        self.assertEqual(cmd[cmd.index("--mount") + 1], "tracker")
        self.assertIn("--gcr", cmd)
        self.assertNotIn("--tilt", cmd)

    def test_opciones_v04(self):
        cmd = construir_comando("bin", "d.tif", "out", -23.6, -69.5, "2026-01-01",
                                mount="tilt", tilt=23, tile=1024, horizon_radius=40,
                                ghi_monthly="data/raw/ghi_MM.tif", ghi_unit="kwh")
        self.assertIn("--tile", cmd)
        self.assertEqual(cmd[cmd.index("--tile") + 1], "1024")
        self.assertIn("--horizon-radius", cmd)
        self.assertEqual(cmd[cmd.index("--horizon-radius") + 1], "40")
        self.assertIn("--ghi-monthly", cmd)
        self.assertEqual(cmd[cmd.index("--ghi-monthly") + 1], "data/raw/ghi_MM.tif")
        self.assertIn("--ghi-unit", cmd)
        self.assertEqual(cmd[cmd.index("--ghi-unit") + 1], "kwh")

    def test_montaje_invalido(self):
        with self.assertRaises(ValueError):
            construir_comando("bin", "d.tif", "out", 0, 0, "2026-01-01", mount="foo")


class TestErroresAccionables(unittest.TestCase):
    def test_binario_ausente(self):
        with self.assertRaises(MotorNoDisponibleError):
            _resolver_binario("binario_que_no_existe_xyz")

    def test_dem_ausente(self):
        with self.assertRaises(FileNotFoundError):
            generar_mapa_rendimiento("no_existe.tif", "out", -23.6, -69.5, "2026-01-01")


@unittest.skipIf(rasterio is None, "rasterio no disponible en el entorno")
class TestValidacionRaster(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_reestampar_crs_cuando_falta_epsg(self):
        p = _escribir_raster(os.path.join(self.dir, "sin_crs.tif"),
                             np.ones((5, 5)), crs=None)
        # El motor emite sin EPSG: la primera vez re-estampa (True), la segunda no (False).
        self.assertTrue(_reestampar_crs(p, 32719))
        with rasterio.open(p) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            es_valid_crs = (epsg == 32719 or (src.crs and "19S" in str(src.crs)))
            self.assertTrue(es_valid_crs)
        self.assertFalse(_reestampar_crs(p, 32719))

    def test_rechaza_crs_incorrecto(self):
        p = _escribir_raster(os.path.join(self.dir, "wgs84.tif"), np.ones((5, 5)),
                             crs="EPSG:4326", origin=(-69, -23), res=0.01)
        with self.assertRaises(SalidaMotorInvalidaError):
            _validar_raster_salida(p, 32719, BOUNDS_UTM)

    def test_rechaza_extension_fuera_de_zona(self):
        # 32719 pero fuera de la caja de estudio (origen al oeste de min_x).
        p = _escribir_raster(os.path.join(self.dir, "fuera.tif"), np.ones((5, 5)),
                             origin=(0, 7010000))
        with self.assertRaises(SalidaMotorInvalidaError):
            _validar_raster_salida(p, 32719, BOUNDS_UTM)

    def test_acepta_raster_valido(self):
        p = _escribir_raster(os.path.join(self.dir, "ok.tif"), np.ones((5, 5)))
        _validar_raster_salida(p, 32719, BOUNDS_UTM)  # no debe lanzar


@unittest.skipIf(rasterio is None, "rasterio no disponible en el entorno")
class TestAlinearGhi(unittest.TestCase):
    """El GHI del Explorador (EPSG:4326, ~1 km) debe calzar celda a celda con el DEM."""

    def test_alinea_a_la_grilla_del_dem_y_deja_ceros_como_nan(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Fuente: 6.5 kWh/m²/día, con un borde en cero (como el mar) al oeste de -71.0°,
        # que cae dentro del DEM (≈ -71.0° a -70.9° en lat -27°).
        ghi = np.full((400, 400), 6.5)
        ghi[:, :100] = 0.0
        origen = _escribir_raster(os.path.join(tmp.name, "ghi_ll.tif"), ghi, crs="EPSG:4326",
                                  origin=(-72.0, -25.0), res=0.01)
        dem = _escribir_raster(os.path.join(tmp.name, "dem.tif"), np.ones((10, 12)) * 1000.0)
        salida = os.path.join(tmp.name, "proc", "ghi_32719.tif")

        alinear_ghi_a_dem(origen, dem, salida)

        with rasterio.open(salida) as s, rasterio.open(dem) as d:
            self.assertEqual((s.width, s.height), (d.width, d.height))
            self.assertEqual(s.transform, d.transform)
            self.assertEqual(s.crs, d.crs)
            a = s.read(1)
        self.assertTrue(np.isnan(a).any(), "los ceros de la fuente deben quedar sin dato")
        self.assertTrue(np.isfinite(a).any())
        # Sin ceros que el bilineal mezcle con el dato: el borde no se "oscurece".
        np.testing.assert_allclose(a[np.isfinite(a)], 6.5, atol=1e-4)

        # Incremental: una segunda llamada no reescribe.
        mtime = os.path.getmtime(salida)
        alinear_ghi_a_dem(origen, dem, salida)
        self.assertEqual(mtime, os.path.getmtime(salida))

    def test_unidades_del_ghi(self):
        diario = np.array([[6.2, 7.4], [np.nan, 5.0]])
        a, unidad = ghi_a_kwh_dia(diario)
        np.testing.assert_array_equal(a, diario)
        self.assertIn("día", unidad)
        # Anual (kWh/m²/año): se divide por 365.
        a, unidad = ghi_a_kwh_dia(diario * 365.0)
        np.testing.assert_allclose(a, diario)
        self.assertIn("365", unidad)
        # W/m² (~250 en el desierto) no es ninguna de las dos: falla en voz alta.
        with self.assertRaises(ValueError):
            ghi_a_kwh_dia(np.array([[250.0, 270.0]]))

    def test_alinear_convierte_ghi_anual_a_diario(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        origen = _escribir_raster(os.path.join(tmp.name, "ghi_anual.tif"),
                                  np.full((400, 400), 6.5 * 365.0), crs="EPSG:4326",
                                  origin=(-72.0, -25.0), res=0.01)
        dem = _escribir_raster(os.path.join(tmp.name, "dem.tif"), np.ones((10, 12)) * 1000.0)
        salida = os.path.join(tmp.name, "ghi_32719.tif")
        alinear_ghi_a_dem(origen, dem, salida)
        with rasterio.open(salida) as s:
            a = s.read(1)
        np.testing.assert_allclose(a[np.isfinite(a)], 6.5, atol=1e-3)

    def test_preparar_ghi_sin_origen_usa_el_alineado(self):
        """El origen se descarga aparte: con el GHI ya alineado el motor puede correr igual."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        alineado = _escribir_raster(os.path.join(tmp.name, "ghi_32719.tif"), np.full((4, 4), 6.5))
        cfg = {'dem': 'dem.tif', 'ghi': alineado,
               'ghi_origen': os.path.join(tmp.name, "no_existe.tif")}
        self.assertEqual(preparar_ghi(cfg, lambda r: r), alineado)
        # Sin ninguno de los dos: error accionable, no un traceback de rasterio.
        cfg['ghi'] = os.path.join(tmp.name, "tampoco.tif")
        with self.assertRaisesRegex(FileNotFoundError, "Explorador Solar"):
            preparar_ghi(cfg, lambda r: r)

    def test_preparar_ghi_sin_ghi_es_cielo_despejado(self):
        self.assertIsNone(preparar_ghi({'dem': 'dem.tif'}, lambda r: r))
        # Sin ghi_origen, usa el GHI ya alineado tal cual.
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, 'g.tif'), 'wb').close()
            self.assertEqual(preparar_ghi({'dem': 'dem.tif', 'ghi': 'g.tif'},
                                          lambda r: os.path.join(tmp, r)),
                             os.path.join(tmp, 'g.tif'))


@unittest.skipIf(rasterio is None, "rasterio no disponible en el entorno")
class TestFlujoConMotorSimulado(unittest.TestCase):
    """Simula una corrida exitosa del motor (sin binario real) y valida el post-proceso."""

    def test_flujo_completo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dem = _escribir_raster(os.path.join(tmp.name, "dem.tif"), np.ones((6, 6)) * 1000.0)
        out_prefix = os.path.join(tmp.name, "rendimiento_fijo")

        def _fake_run(cmd, **kwargs):
            # El "motor" escribe la salida sin EPSG (LOCAL_CS), como el real.
            _escribir_raster(f"{out_prefix}_specific_yield.tif",
                             np.ones((6, 6)) * 1800.0, crs=None)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch("src.solar_yield._resolver_binario", return_value="/fake/solarpv"), \
             mock.patch("src.solar_yield.subprocess.run", side_effect=_fake_run):
            salida = generar_mapa_rendimiento(
                dem, out_prefix, -23.6, -69.5, "2026-01-01",
                target_srid=32719, bounds_utm=BOUNDS_UTM,
            )

        self.assertTrue(os.path.exists(salida))
        with rasterio.open(salida) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            self.assertTrue(epsg == 32719 or (src.crs and "19S" in str(src.crs)))  # se re-estampó
            self.assertAlmostEqual(float(src.read(1).mean()), 1800.0, places=1)


def _motor_simulado(con_error_de_tile: bool, llamadas: list):
    """`subprocess.run` falso: rendimiento = 100 × GHI de cada celda.

    Con `con_error_de_tile`, imita solarpv-rs v0.4.0 sin corregir: con --tile, cada tesela
    lee el GHI de la esquina noroeste (aquí, sin dato), así que todo sale 0.
    """
    def _run(cmd, **kwargs):
        llamadas.append(cmd)
        prefijo = cmd[cmd.index("--out-prefix") + 1]
        with rasterio.open(cmd[cmd.index("--ghi") + 1]) as g:
            valor = np.nan_to_num(g.read(1).astype("float64") * 100.0)
        if con_error_de_tile and "--tile" in cmd:
            valor = np.zeros_like(valor)
        for producto in ("poa", "ac", "specific_yield"):
            _escribir_raster(f"{prefijo}_{producto}.tif", valor, crs=None)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return _run


@unittest.skipIf(rasterio is None, "rasterio no disponible en el entorno")
class TestTileConGhi(unittest.TestCase):
    """Con GHI observado y teselas, el pipeline usa --tile del motor corregido y verifica
    antes de la corrida larga que el binario no tenga el error de v0.4.0."""

    def setUp(self):
        # La verificación se recuerda por binario dentro del proceso: cada test parte limpio.
        _BINARIOS_VERIFICADOS.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dem = _escribir_raster(os.path.join(tmp.name, "dem.tif"), np.ones((10, 12)) * 1000.0)
        ghi_arr = (np.arange(120, dtype="float64").reshape(10, 12) % 17 + 5.0)
        self.ghi = _escribir_raster(os.path.join(tmp.name, "ghi.tif"), ghi_arr, nodata=np.nan)
        self.out_prefix = os.path.join(tmp.name, "rendimiento_fijo")

    def _generar(self, con_error_de_tile, llamadas):
        with mock.patch("src.solar_yield._resolver_binario", return_value="/fake/solarpv"), \
             mock.patch("src.solar_yield.subprocess.run",
                        side_effect=_motor_simulado(con_error_de_tile, llamadas)):
            return generar_mapa_rendimiento(
                self.dem, self.out_prefix, -23.6, -69.5, "2026-01-01", tile=4,
                horizon_radius=2, ghi=self.ghi, target_srid=32719, bounds_utm=BOUNDS_UTM)

    def test_motor_corregido_corre_con_tile(self):
        llamadas = []
        salida = self._generar(con_error_de_tile=False, llamadas=llamadas)
        # 2 corridas de verificación (sin y con --tile) + la corrida real, con --tile.
        self.assertEqual(len(llamadas), 3)
        corrida = llamadas[-1]
        self.assertEqual(corrida[corrida.index("--tile") + 1], "4")
        self.assertEqual(corrida[corrida.index("--ghi") + 1], self.ghi)
        self.assertTrue(os.path.exists(salida))

    def test_dem_remuestreado_recibe_ghi_en_su_misma_grilla(self):
        """Con resolucion_m el DEM cambia de grilla: el GHI que recibe el motor debe calzar
        con ese DEM, no con el original (el motor exige que calcen celda a celda)."""
        grillas = []

        def _run(cmd, **kwargs):
            with rasterio.open(cmd[cmd.index("--dem") + 1]) as d, \
                 rasterio.open(cmd[cmd.index("--ghi") + 1]) as g:
                grillas.append(((d.width, d.height, d.transform), (g.width, g.height, g.transform)))
                forma = (d.height, d.width)
            prefijo = cmd[cmd.index("--out-prefix") + 1]
            _escribir_raster(f"{prefijo}_specific_yield.tif", np.ones(forma) * 1800.0, crs=None,
                             res=2000)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        # GHI realista (kWh/m²/día): al realinearse pasa por el control de unidades.
        ghi = _escribir_raster(self.out_prefix + "_ghi_diario.tif",
                               np.linspace(5.5, 7.5, 120).reshape(10, 12), nodata=np.nan)
        with mock.patch("src.solar_yield._resolver_binario", return_value="/fake/solarpv"), \
             mock.patch("src.solar_yield.subprocess.run", side_effect=_run):
            generar_mapa_rendimiento(self.dem, self.out_prefix, -23.6, -69.5, "2026-01-01",
                                     ghi=ghi, resolucion_m=2000, target_srid=32719,
                                     bounds_utm=BOUNDS_UTM)
        grilla_dem, grilla_ghi = grillas[0]
        self.assertEqual(grilla_dem, grilla_ghi)
        self.assertEqual(grilla_dem[:2], (6, 5))          # 12x10 celdas de 1 km -> 6x5 de 2 km

    def test_verifica_una_sola_vez_por_binario(self):
        """La etapa 10 corre tres montajes seguidos: la verificación no se repite."""
        llamadas = []
        self._generar(con_error_de_tile=False, llamadas=llamadas)
        self._generar(con_error_de_tile=False, llamadas=llamadas)
        self.assertEqual(len(llamadas), 3 + 1)

    def test_motor_sin_corregir_se_rechaza_antes_de_correr(self):
        llamadas = []
        with self.assertRaises(MotorSinCorreccionTileError) as ctx:
            self._generar(con_error_de_tile=True, llamadas=llamadas)
        self.assertIn("git apply", str(ctx.exception))
        self.assertEqual(len(llamadas), 2)               # nunca llegó a la corrida real
        self.assertFalse(os.path.exists(f"{self.out_prefix}_specific_yield.tif"))

    def test_sin_ghi_no_verifica(self):
        """A cielo despejado --tile nunca tuvo el error: no se gasta la verificación."""
        llamadas = []

        def _run(cmd, **kwargs):
            llamadas.append(cmd)
            prefijo = cmd[cmd.index("--out-prefix") + 1]
            _escribir_raster(f"{prefijo}_specific_yield.tif", np.ones((10, 12)) * 1800.0, crs=None)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch("src.solar_yield._resolver_binario", return_value="/fake/solarpv"), \
             mock.patch("src.solar_yield.subprocess.run", side_effect=_run):
            generar_mapa_rendimiento(self.dem, self.out_prefix, -23.6, -69.5, "2026-01-01",
                                     tile=4, target_srid=32719, bounds_utm=BOUNDS_UTM)
        self.assertEqual(len(llamadas), 1)


if __name__ == "__main__":
    unittest.main()
