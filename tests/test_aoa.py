"""Tests unitarios para el Área de Aplicabilidad (AOA) e Índice de Disimilitud (DI)."""

import os
import sys
import unittest
import tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FEATURES
from src.aoa import (
    extraer_pesos_modelo,
    ajustar_espacio_aoa,
    calcular_di_y_aoa,
    escribir_rasters_aoa,
    evaluar_aoa_zona,
)


class MockRandomForest:
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances)


class TestAOA(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 50 muestras de entrenamiento sintéticas en 7 features
        self.X_train = np.random.normal(loc=10.0, scale=2.0, size=(50, len(FEATURES)))
        self.pesos = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])

    def test_extraer_pesos_modelo(self):
        mock_rf = MockRandomForest([0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
        pesos = extraer_pesos_modelo(mock_rf, FEATURES)
        self.assertAlmostEqual(pesos.sum(), 1.0, places=6)
        self.assertEqual(len(pesos), len(FEATURES))

        # Modelo sin feature_importances_ genera pesos uniformes
        pesos_unif = extraer_pesos_modelo(object(), FEATURES)
        self.assertAlmostEqual(pesos_unif.sum(), 1.0, places=6)
        self.assertAlmostEqual(pesos_unif[0], 1.0 / len(FEATURES), places=6)

    def test_ajustar_espacio_aoa(self):
        ajuste = ajustar_espacio_aoa(self.X_train, self.pesos)
        self.assertGreater(ajuste.dist_media_train, 0.0)
        self.assertGreater(ajuste.umbral_di, 0.0)
        # El umbral Q3 + 1.5*IQR debe ser >= Q3
        self.assertGreaterEqual(ajuste.umbral_di, ajuste.di_train_stats['di_q75'])

    def test_puntos_cercanos_y_lejanos(self):
        ajuste = ajustar_espacio_aoa(self.X_train, self.pesos)

        # Punto idéntico a la media del entrenamiento -> debe estar dentro del AOA
        X_in = np.array([np.mean(self.X_train, axis=0)])
        di_in, aoa_in = calcular_di_y_aoa(ajuste, X_in)
        self.assertTrue(aoa_in[0])
        self.assertLess(di_in[0], ajuste.umbral_di)

        # Punto a 20 desviaciones estándar -> debe estar fuera del AOA (extrapolación)
        X_out = np.array([np.mean(self.X_train, axis=0) + 20.0 * np.std(self.X_train, axis=0)])
        di_out, aoa_out = calcular_di_y_aoa(ajuste, X_out)
        self.assertFalse(aoa_out[0])
        self.assertGreater(di_out[0], ajuste.umbral_di)

    def test_equivalencia_chunking(self):
        ajuste = ajustar_espacio_aoa(self.X_train, self.pesos)
        X_pred = np.random.normal(loc=10.0, scale=3.0, size=(120, len(FEATURES)))

        di_completo, aoa_completo = calcular_di_y_aoa(ajuste, X_pred, tamano_lote=1000)
        di_chunked, aoa_chunked = calcular_di_y_aoa(ajuste, X_pred, tamano_lote=17)

        np.testing.assert_allclose(di_completo, di_chunked, rtol=1e-5)
        np.testing.assert_array_equal(aoa_completo, aoa_chunked)

    def test_evaluar_aoa_zona_y_escritura_rasters(self):
        ajuste = ajustar_espacio_aoa(self.X_train, self.pesos)
        shape = (20, 20)
        valido = np.ones(shape, dtype=bool)
        X_zona = np.random.normal(loc=10.0, scale=2.5, size=(400, len(FEATURES)))

        with tempfile.TemporaryDirectory() as tmpdir:
            transform = from_origin(100000, 7000000, 500, 500)
            base_meta = {
                'crs': 'EPSG:32719',
                'transform': transform,
                'width': 20,
                'height': 20,
            }
            plantas_rc = [(5, 5), (10, 10)]

            res = evaluar_aoa_zona(
                ajuste_aoa=ajuste,
                X_zona=X_zona,
                valido=valido,
                base_meta=base_meta,
                dir_salida=tmpdir,
                plantas_rc=plantas_rc,
            )

            self.assertIn('pct_dentro_aoa', res)
            self.assertIn('pct_fuera_aoa', res)
            self.assertAlmostEqual(res['pct_dentro_aoa'] + res['pct_fuera_aoa'], 100.0, places=1)
            self.assertEqual(res['plantas_en_aoa']['n_plantas_evaluadas'], 2)

            self.assertTrue(os.path.exists(res['rasters']['di']))
            self.assertTrue(os.path.exists(res['rasters']['aoa']))

            with rasterio.open(res['rasters']['di']) as src:
                self.assertEqual(src.shape, (20, 20))
                self.assertEqual(src.dtypes[0], 'float32')

            with rasterio.open(res['rasters']['aoa']) as src:
                self.assertEqual(src.shape, (20, 20))
                self.assertEqual(src.dtypes[0], 'uint8')


if __name__ == '__main__':
    unittest.main()
