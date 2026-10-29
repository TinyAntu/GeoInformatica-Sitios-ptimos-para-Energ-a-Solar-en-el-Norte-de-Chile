"""Pruebas del análisis aparte de teselado (src/comparar_tiles.py).

No requieren el binario Rust: simulan el motor con un `subprocess.run` falso que devuelve
rendimiento = 100 × GHI de cada celda del recorte que recibe.
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
from src.utils import _asegurar_proj_lib
_asegurar_proj_lib()

from src.comparar_tiles import correr_teselado_python, metricas

UTM19S = rasterio.crs.CRS.from_dict({'proj': 'utm', 'zone': 19, 'south': True,
                                     'datum': 'WGS84', 'units': 'm'})
OPCIONES = dict(lat=-23.6, lon=-69.5, date="2026-01-01", mount="tilt", tilt=23,
                horizon_radius=2, ghi_unit="kwh")


def _escribir(path, data, transform=None, crs=UTM19S):
    h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=1, width=w, height=h,
                       transform=transform or from_origin(300000, 7010000, 1000, 1000),
                       crs=crs, nodata=np.nan) as d:
        d.write(data.astype("float32"), 1)
    return path


class TestTeseladoPython(unittest.TestCase):
    def test_cada_celda_sale_de_su_ghi_y_teselas_sin_ghi_no_corren(self):
        """Recorta DEM y GHI con la misma ventana: cada celda usa SU GHI (no el de la
        esquina, que es el error de --tile en v0.4.0) y las teselas sin GHI quedan en 0
        sin invocar al motor."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dem = _escribir(os.path.join(tmp.name, "dem.tif"), np.ones((10, 12)) * 1000.0)
        ghi_arr = (np.arange(120, dtype="float64").reshape(10, 12) % 17 + 5.0)
        ghi_arr[:4, :4] = np.nan                 # una tesela completa sin GHI (como el mar)
        ghi = _escribir(os.path.join(tmp.name, "ghi.tif"), ghi_arr)
        llamadas = []

        def _fake_run(cmd, **kwargs):
            llamadas.append(cmd)
            self.assertNotIn("--tile", cmd)
            prefijo = cmd[cmd.index("--out-prefix") + 1]
            with rasterio.open(cmd[cmd.index("--ghi") + 1]) as g:
                valor = np.nan_to_num(g.read(1).astype("float64") * 100.0)
            for producto in ("poa", "ac", "specific_yield"):
                _escribir(f"{prefijo}_{producto}.tif", valor, crs=None)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch("src.comparar_tiles.subprocess.run", side_effect=_fake_run):
            salida = correr_teselado_python("/fake/solarpv", dem, ghi,
                                            os.path.join(tmp.name, "python"), 4, 2, OPCIONES)

        with rasterio.open(salida) as src, rasterio.open(dem) as d:
            self.assertEqual(src.transform, d.transform)
            resultado = src.read(1)
        np.testing.assert_allclose(resultado, np.nan_to_num(ghi_arr * 100.0))
        self.assertEqual(len(llamadas), 8)       # 3 × 3 teselas, una sin GHI


class TestMetricas(unittest.TestCase):
    def test_identicas_y_diferencias(self):
        ref = np.full((2, 3), 2000.0)
        otro = ref.copy()
        otro[0, 0] = 2002.0                      # +0,1 %
        otro[1, 2] = 0.0                         # celda con GHI que sale en 0
        m = metricas(otro, ref, np.ones_like(ref, dtype=bool))
        self.assertAlmostEqual(m['celdas_identicas_pct'], 66.67, places=2)
        self.assertEqual(m['celdas_con_ghi_en_0'], 1)
        self.assertEqual(m['dif_rel_max_pct'], 100.0)
        self.assertEqual(m['celdas_dif_mayor_0_01pct'], 2)


if __name__ == "__main__":
    unittest.main()
