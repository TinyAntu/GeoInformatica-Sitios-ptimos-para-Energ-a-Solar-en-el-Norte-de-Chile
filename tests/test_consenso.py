"""Pruebas del consenso/divergencia entre perfiles (src/consenso_perfiles.py, Brecha 6)."""

import os
import sys
import unittest
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.consenso_perfiles import analizar_consenso, NODATA


def _escribir(path, data):
    h, w = data.shape
    transform = from_origin(300000, 7010000, 1000, 1000)
    perfil = dict(driver="GTiff", dtype="float32", count=1, width=w, height=h,
                  crs="EPSG:32719", transform=transform, nodata=NODATA)
    with rasterio.open(path, "w", **perfil) as d:
        d.write(data.astype("float32"), 1)
    return path


class TestConsenso(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        # 10x10; valores crecientes: el top 10% (percentil 90) son los más altos de cada perfil.
        base_vals = np.linspace(0, 1, 100).reshape(10, 10).astype("float32")
        # Balanceado y conservador comparten el orden; agresivo lo invierte -> consenso parcial.
        self.balanceado = _escribir(os.path.join(self.dir, "bal.tif"), base_vals)
        self.conservador = _escribir(os.path.join(self.dir, "cons.tif"), base_vals)
        self.agresivo = _escribir(os.path.join(self.dir, "agr.tif"), base_vals[::-1, ::-1])

    def tearDown(self):
        self.tmp.cleanup()

    def test_categorias_y_conteos(self):
        out_r = os.path.join(self.dir, "consenso.tif")
        out_j = os.path.join(self.dir, "consenso.json")
        stats = analizar_consenso(self.balanceado, self.conservador, self.agresivo, 90, out_r, out_j)

        with rasterio.open(out_r) as d:
            self.assertEqual(d.crs.to_epsg(), 32719)
            self.assertEqual(d.nodata, NODATA)
            vals = np.unique(d.read(1))
        # Solo categorías válidas 0..3 (más el nodata si lo hubiera; aquí todo es válido).
        self.assertTrue(set(vals.tolist()).issubset({0.0, 1.0, 2.0, 3.0}))
        # Cada perfil marca ~10% (10 de 100 celdas) como apto.
        for n in stats["apto_por_perfil"].values():
            self.assertEqual(n, 10)
        self.assertEqual(stats["celdas_validas"], 100)
        # Consenso + divergencias + no_apto cubren el total.
        suma = (stats["consenso_3_perfiles"]["celdas"] + stats["divergencia_2_perfiles"]["celdas"]
                + stats["divergencia_1_perfil"]["celdas"] + stats["no_apto"]["celdas"])
        self.assertEqual(suma, 100)

    def test_sin_celdas_comunes_lanza(self):
        # Un perfil todo nodata -> sin base válida común.
        vacio = _escribir(os.path.join(self.dir, "vacio.tif"),
                          np.full((10, 10), NODATA, dtype="float32"))
        with self.assertRaises(ValueError):
            analizar_consenso(self.balanceado, self.conservador, vacio, 90,
                              os.path.join(self.dir, "x.tif"), os.path.join(self.dir, "x.json"))


if __name__ == "__main__":
    unittest.main()
