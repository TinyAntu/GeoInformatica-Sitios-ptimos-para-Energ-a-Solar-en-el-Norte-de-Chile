"""Pruebas del cruce aptitud x rendimiento (src/cruce_aptitud_rendimiento.py, T2).

Usa rasters sintéticos en EPSG:32719 sobre la misma grilla; no requiere el motor ni PostGIS.
"""

import os
import sys
import json
import unittest
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cruce_aptitud_rendimiento import cruzar, NODATA

TRANSFORM = from_origin(300000, 7010000, 1000, 1000)


UTM19S_CRS = rasterio.crs.CRS.from_dict({'proj': 'utm', 'zone': 19, 'south': True, 'datum': 'WGS84', 'units': 'm'})


def _escribir(path, data, origin=(300000, 7010000)):
    h, w = data.shape
    transform = from_origin(origin[0], origin[1], 1000, 1000)
    perfil = dict(driver="GTiff", dtype="float32", count=1, width=w, height=h,
                  crs=UTM19S_CRS, transform=transform, nodata=NODATA)
    with rasterio.open(path, "w", **perfil) as d:
        d.write(data.astype("float32"), 1)
    return path


class TestCruce(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        # Aptitud: nodata, exclusión (0.0), aptas (>=0.7) y no-aptas (<0.7).
        self.apt = np.array([
            [NODATA, 0.0,   0.90, 0.30],
            [0.80,   0.75,  0.50, NODATA],
            [0.95,   0.20,  0.0,  0.85],
            [0.10,   0.70,  0.72, 0.40],
        ], dtype="float32")
        # Rendimiento: todo finito, mismo grid.
        self.rend = (np.arange(16, dtype="float32").reshape(4, 4) * 100 + 1000)
        self.apt_path = _escribir(os.path.join(self.dir, "apt.tif"), self.apt)
        self.rend_path = _escribir(os.path.join(self.dir, "rend.tif"), self.rend)

    def tearDown(self):
        self.tmp.cleanup()

    def _correr(self):
        out_a = os.path.join(self.dir, "en_aptas.tif")
        out_r = os.path.join(self.dir, "ranking.tif")
        out_j = os.path.join(self.dir, "stats.json")
        stats = cruzar(self.apt_path, self.rend_path, 0.70, out_a, out_r, out_j)
        return out_a, out_r, out_j, stats

    def test_rendimiento_en_aptas(self):
        out_a, _, _, stats = self._correr()
        with rasterio.open(out_a) as d:
            epsg = d.crs.to_epsg() if d.crs else None
            self.assertTrue(epsg == 32719 or (d.crs and "19S" in str(d.crs)))
            self.assertEqual(d.nodata, NODATA)
            en_aptas = d.read(1)
        aptas = (self.apt != NODATA) & np.isfinite(self.apt) & (self.apt >= 0.70)
        # Donde es apta debe estar el rendimiento; fuera, nodata.
        np.testing.assert_allclose(en_aptas[aptas], self.rend[aptas], rtol=1e-4)
        self.assertTrue(np.all(en_aptas[~aptas] == NODATA))
        self.assertEqual(stats["celdas_aptas"], int(aptas.sum()))

    def test_ranking_exclusion_y_nodata(self):
        _, out_r, _, _ = self._correr()
        with rasterio.open(out_r) as d:
            ranking = d.read(1)
        # Exclusión (prob 0.0) -> score 0.0; nodata -> NODATA.
        cero_apt = (self.apt == 0.0)
        self.assertTrue(np.all(ranking[cero_apt] == 0.0))
        self.assertTrue(np.all(ranking[self.apt == NODATA] == NODATA))
        # En celdas válidas el ranking cae en [0, 1].
        valido = ranking != NODATA
        self.assertTrue(ranking[valido].min() >= 0.0)
        self.assertTrue(ranking[valido].max() <= 1.0)

    def test_json_generado(self):
        _, _, out_j, stats = self._correr()
        self.assertTrue(os.path.exists(out_j))
        with open(out_j, encoding="utf-8") as f:
            disco = json.load(f)
        self.assertEqual(disco["celdas_aptas"], stats["celdas_aptas"])

    def test_sin_celdas_comunes_lanza(self):
        # Rendimiento en una extensión disjunta: no hay solape -> ValueError.
        rend_lejos = _escribir(os.path.join(self.dir, "rend_lejos.tif"),
                               self.rend, origin=(800000, 6300000))
        with self.assertRaises(ValueError):
            cruzar(self.apt_path, rend_lejos, 0.70,
                   os.path.join(self.dir, "a.tif"),
                   os.path.join(self.dir, "b.tif"),
                   os.path.join(self.dir, "c.json"))


if __name__ == "__main__":
    unittest.main()
