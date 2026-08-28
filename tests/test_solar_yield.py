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
import rasterio
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.solar_yield import (
    construir_comando, generar_mapa_rendimiento, _resolver_binario,
    _reestampar_crs, _validar_raster_salida,
    MotorNoDisponibleError, SalidaMotorInvalidaError,
)

# Caja UTM de la zona de estudio (misma convención que config.postgis_validation.bounds_utm).
BOUNDS_UTM = {"min_x": 100000.0, "max_x": 900000.0, "min_y": 6000000.0, "max_y": 8500000.0}


def _escribir_raster(path, data, crs="EPSG:32719", origin=(300000, 7010000), res=1000, nodata=-9999.0):
    h, w = data.shape
    transform = from_origin(origin[0], origin[1], res, res)
    perfil = dict(driver="GTiff", dtype="float32", count=1, width=w, height=h,
                  transform=transform, nodata=nodata)
    if crs is not None:
        perfil["crs"] = crs
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
        self.assertIn("--annual", cmd)
        self.assertIn("--per-cell-lat", cmd)
        self.assertIn("--svf", cmd)

    def test_seguidor(self):
        cmd = construir_comando("bin", "d.tif", "out", -23.6, -69.5, "2026-01-01",
                                mount="tracker", gcr=0.3)
        self.assertEqual(cmd[cmd.index("--mount") + 1], "tracker")
        self.assertIn("--gcr", cmd)
        self.assertNotIn("--tilt", cmd)

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
            self.assertEqual(src.crs.to_epsg(), 32719)
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
            self.assertEqual(src.crs.to_epsg(), 32719)  # se re-estampó
            self.assertAlmostEqual(float(src.read(1).mean()), 1800.0, places=1)


if __name__ == "__main__":
    unittest.main()
