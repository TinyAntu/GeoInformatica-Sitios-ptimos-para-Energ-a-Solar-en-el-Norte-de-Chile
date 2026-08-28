"""Pruebas del helper de reconstrucción de rasters SHAP espaciales (Brecha 8).

El cálculo SHAP completo requiere el modelo y los rasters del proyecto (pesado); aquí se
prueba la lógica reusable de reconstrucción raster (valores válidos + NODATA) en aislado.
"""

import os
import sys
import unittest
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.explainability_spatial import _escribir_raster, NODATA


class TestEscribirRasterSHAP(unittest.TestCase):
    def test_reconstruye_con_nodata(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        utm_crs = rasterio.crs.CRS.from_dict({'proj': 'utm', 'zone': 19, 'south': True, 'datum': 'WGS84', 'units': 'm'})
        base_meta = {
            "crs": utm_crs,
            "transform": from_origin(300000, 7010000, 1000, 1000),
            "width": 4, "height": 3,
        }
        # 12 celdas; válidas alternadas.
        valid = np.array([[True, False, True, False],
                          [False, True, True, False],
                          [True, True, False, True]])
        valores = np.array([-0.3, 0.1, 0.25, -0.05, 0.4, 0.15, -0.2], dtype="float32")
        path = os.path.join(tmp.name, "shap.tif")
        _escribir_raster(valores, valid, base_meta, path)

        with rasterio.open(path) as d:
            epsg = d.crs.to_epsg() if d.crs else None
            self.assertTrue(epsg == 32719 or (d.crs and "19S" in str(d.crs)))
            self.assertEqual(d.nodata, NODATA)
            arr = d.read(1)
        # Las celdas válidas tienen los valores (con signo); el resto NODATA.
        np.testing.assert_allclose(arr[valid], valores, rtol=1e-5)
        self.assertTrue(np.all(arr[~valid] == NODATA))


if __name__ == "__main__":
    unittest.main()
