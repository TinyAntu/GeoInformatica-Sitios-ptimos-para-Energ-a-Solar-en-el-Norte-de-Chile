"""Pruebas de la comparación fijo vs. seguidor (src/comparacion_montaje.py, T3).

Rasters sintéticos en EPSG:32719; no requiere el motor.
"""

import os
import sys
import unittest
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.comparacion_montaje import comparar
from src.cruce_aptitud_rendimiento import NODATA


def _escribir(path, data):
    h, w = data.shape
    transform = from_origin(300000, 7010000, 1000, 1000)
    perfil = dict(driver="GTiff", dtype="float32", count=1, width=w, height=h,
                  crs="EPSG:32719", transform=transform, nodata=NODATA)
    with rasterio.open(path, "w", **perfil) as d:
        d.write(data.astype("float32"), 1)
    return path


class TestComparacion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        # Aptitud: la mitad apta (>=0.7).
        self.apt = np.array([[0.9, 0.8, 0.2, 0.3],
                             [0.75, 0.95, 0.1, 0.4],
                             [0.85, 0.72, 0.5, 0.6],
                             [0.71, 0.90, 0.0, 0.65]], dtype="float32")
        # Fijo constante 1400; seguidor constante 2100 -> ganancia esperada +50%.
        self.fijo = np.full((4, 4), 1400.0, dtype="float32")
        self.seg = np.full((4, 4), 2100.0, dtype="float32")
        self.apt_p = _escribir(os.path.join(self.dir, "apt.tif"), self.apt)
        self.fijo_p = _escribir(os.path.join(self.dir, "fijo.tif"), self.fijo)
        self.seg_p = _escribir(os.path.join(self.dir, "seg.tif"), self.seg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ganancia_seguidor(self):
        out_j = os.path.join(self.dir, "comp.json")
        stats = comparar(self.fijo_p, self.seg_p, self.apt_p, 0.70, out_j)
        self.assertTrue(os.path.exists(out_j))
        self.assertAlmostEqual(stats["ganancia_seguidor_media_pct"], 50.0, places=1)
        self.assertEqual(stats["celdas_aptas"], int((self.apt >= 0.70).sum()))
        self.assertAlmostEqual(stats["rendimiento_fijo_aptas"]["media"], 1400.0, places=1)
        self.assertAlmostEqual(stats["rendimiento_seguidor_aptas"]["media"], 2100.0, places=1)

    def test_sin_aptas_lanza(self):
        apt_baja = _escribir(os.path.join(self.dir, "apt_baja.tif"),
                             np.full((4, 4), 0.1, dtype="float32"))
        with self.assertRaises(ValueError):
            comparar(self.fijo_p, self.seg_p, apt_baja, 0.70,
                     os.path.join(self.dir, "x.json"))


if __name__ == "__main__":
    unittest.main()
