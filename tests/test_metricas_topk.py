"""Tests de las métricas top-K sobre una grilla sintética con valores conocidos a mano."""

import os
import sys
import unittest

import numpy as np
from rasterio.transform import from_origin

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metricas_topk import (
    calcular_metricas_topk,
    contrastar_con_umbrales_pep1,
    _plantas_a_filas_columnas,
    _precision_por_area,
)


class TestPrecisionPorArea(unittest.TestCase):
    def setUp(self):
        self.valido = np.ones((10, 10), dtype=bool)
        self.area_px_m2 = 250000.0

    def test_precision_y_techo_con_huella_parcialmente_capturada(self):
        # Huella de 10 px; el top-10% (10 px de la fila 9) captura 4 de ellos.
        top = np.zeros((10, 10), dtype=bool)
        top[9, :] = True
        huella = np.zeros((10, 10), dtype=bool)
        huella[9, 0:4] = True   # dentro del top
        huella[0, 0:6] = True   # fuera del top
        res = _precision_por_area(top, huella, self.valido, 100, 10, self.area_px_m2)
        self.assertAlmostEqual(res['precision'], 0.4, places=6)          # 4 / 10 px del top
        self.assertAlmostEqual(res['precision_techo_modelo_perfecto'], 1.0, places=6)  # 10/10
        self.assertAlmostEqual(res['pct_del_techo_alcanzado'], 40.0, places=1)
        self.assertAlmostEqual(res['precision_base_azar'], 0.10, places=6)  # 10 px / 100

    def test_techo_limita_la_precision_cuando_la_huella_es_menor_que_el_top(self):
        # Huella de 2 px y top de 10 px: ni capturando todo se pasa de 0.2.
        top = np.zeros((10, 10), dtype=bool)
        top[9, :] = True
        huella = np.zeros((10, 10), dtype=bool)
        huella[9, 0:2] = True
        res = _precision_por_area(top, huella, self.valido, 100, 10, self.area_px_m2)
        self.assertAlmostEqual(res['precision_techo_modelo_perfecto'], 0.2, places=6)
        self.assertAlmostEqual(res['precision'], 0.2, places=6)
        self.assertAlmostEqual(res['pct_del_techo_alcanzado'], 100.0, places=1)

    def test_sin_huella_devuelve_none(self):
        top = np.ones((10, 10), dtype=bool)
        huella = np.zeros((10, 10), dtype=bool)
        self.assertIsNone(
            _precision_por_area(top, huella, self.valido, 100, 100, self.area_px_m2))


class TestContrasteUmbralesPEP1(unittest.TestCase):
    def test_marca_el_umbral_como_inalcanzable_si_el_techo_no_lo_permite(self):
        combinado = {
            '1': {'precision_zonas': 0.11,
                  'precision_area': {'precision': 0.0535,
                                     'precision_techo_modelo_perfecto': 0.0712,
                                     'pct_del_techo_alcanzado': 75.1}},
            '3': {'recall': 0.781},
        }
        c = contrastar_con_umbrales_pep1(combinado)
        self.assertFalse(c['precision_k1']['cumple'])
        # El techo (7,12%) está por debajo del umbral de PEP1 (15%): inalcanzable.
        self.assertFalse(c['precision_k1']['umbral_alcanzable'])
        self.assertFalse(c['recall_k3']['cumple'])
        self.assertTrue(c['recall_k3']['umbral_alcanzable'])


class TestCalcularMetricasTopK(unittest.TestCase):
    def setUp(self):
        # Grilla 10x10 = 100 píxeles, todos válidos. La probabilidad crece con el índice
        # plano, así que el top-10% son exactamente los 10 últimos (valores 90..99).
        self.prob = np.arange(100, dtype=np.float32).reshape(10, 10)
        self.valido = np.ones((10, 10), dtype=bool)
        self.area_px_m2 = 250000.0  # 500 m x 500 m

    def test_umbral_selecciona_la_fraccion_pedida(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[9, 9] = True
        res = calcular_metricas_topk(self.prob, self.valido, mask, [(9, 9)],
                                     self.area_px_m2, ks=(10.0,))
        # percentil 90 de 0..99 = 89.1 -> se seleccionan los valores >= 89.1, es decir 90..99
        self.assertEqual(res['por_k']['10']['n_pixeles_top'], 10)
        self.assertAlmostEqual(res['por_k']['10']['pct_area_seleccionada'], 10.0, places=3)

    def test_recall_cuenta_plantas_por_centroide(self):
        # 4 plantas: dos dentro del top-10% (filas 9) y dos fuera (fila 0).
        plantas_rc = [(9, 5), (9, 8), (0, 1), (0, 2)]
        mask = np.zeros((10, 10), dtype=bool)
        for f, c in plantas_rc:
            mask[f, c] = True
        res = calcular_metricas_topk(self.prob, self.valido, mask, plantas_rc,
                                     self.area_px_m2, ks=(10.0,))
        self.assertEqual(res['n_plantas_evaluadas'], 4)
        self.assertEqual(res['por_k']['10']['plantas_capturadas'], 2)
        self.assertAlmostEqual(res['por_k']['10']['recall'], 0.5, places=4)

    def test_precision_se_mide_sobre_zonas_conexas(self):
        # El top-10% es la fila 9 completa: 10 píxeles contiguos = UNA sola zona conexa.
        mask = np.zeros((10, 10), dtype=bool)
        mask[9, 0] = True
        res = calcular_metricas_topk(self.prob, self.valido, mask, [(9, 0)],
                                     self.area_px_m2, ks=(10.0,))
        zonas = res['por_k']['10']['precision_zonas']
        self.assertEqual(zonas['n_zonas_candidatas'], 1)
        self.assertEqual(zonas['n_zonas_con_planta'], 1)
        self.assertAlmostEqual(zonas['precision'], 1.0, places=4)

    def test_zonas_separadas_se_cuentan_por_separado(self):
        # Probabilidad con dos bloques altos y disjuntos: esquina superior izquierda y
        # esquina inferior derecha. Solo uno contiene planta -> precisión 1/2.
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[0:2, 0:2] = 10.0   # zona A (4 px), con planta
        prob[8:10, 8:10] = 10.0  # zona B (4 px), sin planta
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        res = calcular_metricas_topk(prob, self.valido, mask, [(0, 0)],
                                     self.area_px_m2, ks=(8.0,))
        zonas = res['por_k']['8']['precision_zonas']
        self.assertEqual(zonas['n_zonas_candidatas'], 2)
        self.assertEqual(zonas['n_zonas_con_planta'], 1)
        self.assertAlmostEqual(zonas['precision'], 0.5, places=4)

    def test_zonas_usan_8_conectividad(self):
        # Dos píxeles que solo se tocan en diagonal deben ser UNA zona con 8-conectividad.
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[4, 4] = 10.0
        prob[5, 5] = 10.0
        mask = np.zeros((10, 10), dtype=bool)
        mask[4, 4] = True
        res = calcular_metricas_topk(prob, self.valido, mask, [(4, 4)],
                                     self.area_px_m2, ks=(2.0,))
        self.assertEqual(res['por_k']['2']['precision_zonas']['n_zonas_candidatas'], 1)

    def test_base_azar_de_zonas_crece_con_el_tamano_de_la_zona(self):
        # Una zona que cubre casi toda el área válida tiene alta probabilidad de contener
        # una planta por puro tamaño: la base al azar debe reflejarlo (cercana a 1).
        prob = np.zeros((10, 10), dtype=np.float32)
        prob[1:, :] = 10.0  # 90 de 100 píxeles
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        res = calcular_metricas_topk(prob, self.valido, mask, [(5, 5)],
                                     self.area_px_m2, ks=(90.0,))
        zonas = res['por_k']['90']['precision_zonas']
        self.assertEqual(zonas['n_zonas_candidatas'], 1)
        self.assertGreater(zonas['precision_base_azar'], 0.8)

    def test_pixeles_validos_se_cuentan_sobre_la_mascara(self):
        # Invalida la mitad de la grilla: deben quedar 50 píxeles válidos, no 100.
        valido = np.zeros((10, 10), dtype=bool)
        valido[5:, :] = True
        mask = np.zeros((10, 10), dtype=bool)
        mask[9, 0] = True
        res = calcular_metricas_topk(self.prob, valido, mask, [(9, 0)],
                                     self.area_px_m2, ks=(10.0,))
        self.assertEqual(res['n_pixeles_validos'], 50)

    def test_recall_es_monotono_en_k(self):
        plantas_rc = [(9, 5), (8, 5), (0, 1)]
        mask = np.zeros((10, 10), dtype=bool)
        for f, c in plantas_rc:
            mask[f, c] = True
        res = calcular_metricas_topk(self.prob, self.valido, mask, plantas_rc,
                                     self.area_px_m2, ks=(1.0, 3.0, 30.0))
        recalls = [res['por_k'][k]['recall'] for k in ('1', '3', '30')]
        self.assertLessEqual(recalls[0], recalls[1])
        self.assertLessEqual(recalls[1], recalls[2])

    def test_plantas_fuera_de_validos_no_se_evaluan(self):
        valido = np.zeros((10, 10), dtype=bool)
        valido[5:, :] = True
        mask = np.zeros((10, 10), dtype=bool)
        mask[9, 0] = True
        res = calcular_metricas_topk(self.prob, valido, mask, [(9, 0), (0, 0)],
                                     self.area_px_m2, ks=(10.0,))
        self.assertEqual(res['n_plantas_evaluadas'], 1)

    def test_grilla_sin_pixeles_validos_falla_en_voz_alta(self):
        with self.assertRaises(ValueError):
            calcular_metricas_topk(self.prob, np.zeros((10, 10), dtype=bool),
                                   np.zeros((10, 10), dtype=bool), [], self.area_px_m2)


class TestBarridoDeK(unittest.TestCase):
    def test_el_barrido_por_defecto_incluye_los_k_de_pep1(self):
        # contrastar_con_umbrales_pep1 busca las claves '1' y '3' por nombre: si el barrido
        # por defecto dejara de incluirlas, el contraste quedaría vacío en silencio.
        from src.metricas_topk import KS_DEFECTO, K_RECALL_PEP1, K_PRECISION_PEP1
        claves = {f"{k:g}" for k in KS_DEFECTO}
        self.assertIn(K_RECALL_PEP1, claves)
        self.assertIn(K_PRECISION_PEP1, claves)

    def test_recall_no_decrece_a_lo_largo_del_barrido(self):
        from src.metricas_topk import KS_DEFECTO
        prob = np.arange(100, dtype=np.float32).reshape(10, 10)
        valido = np.ones((10, 10), dtype=bool)
        plantas_rc = [(9, 9), (5, 5), (0, 0)]
        mask = np.zeros((10, 10), dtype=bool)
        for f, c in plantas_rc:
            mask[f, c] = True
        res = calcular_metricas_topk(prob, valido, mask, plantas_rc, 250000.0,
                                     ks=KS_DEFECTO)
        recalls = [res['por_k'][f"{k:g}"]['recall'] for k in sorted(KS_DEFECTO)]
        for anterior, siguiente in zip(recalls, recalls[1:]):
            self.assertLessEqual(anterior, siguiente)


class TestPlantasAFilasColumnas(unittest.TestCase):
    def test_convierte_centroides_y_descarta_fuera_de_grilla(self):
        import geopandas as gpd
        from shapely.geometry import box

        # Origen (0, 5000) con píxeles de 500 m: grilla 10x10 cubre x[0,5000], y[0,5000].
        transform = from_origin(0, 5000, 500, 500)
        dentro = box(1000, 1000, 1500, 1500)      # centroide (1250, 1250)
        fuera = box(90000, 90000, 90500, 90500)   # muy lejos de la grilla
        plantas = gpd.GeoDataFrame(geometry=[dentro, fuera], crs='EPSG:32719')

        rc = _plantas_a_filas_columnas(plantas, transform, (10, 10))
        self.assertEqual(len(rc), 1)
        # y=1250 -> fila (5000-1250)/500 = 7 ; x=1250 -> columna 2
        self.assertEqual(rc[0], (7, 2))


if __name__ == '__main__':
    unittest.main()
