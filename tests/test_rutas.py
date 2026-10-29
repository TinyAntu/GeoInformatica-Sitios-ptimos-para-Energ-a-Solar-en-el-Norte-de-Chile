"""Pruebas de la estructura de carpetas de resultados y figuras (src/rutas.py)."""

import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rutas import (ruta_resultado, ruta_figura, prefijo_rendimiento, dir_figuras,
                       dir_analisis, dir_figuras_analisis, TEMAS)


class TestRutas(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.raiz = tmp.name
        r = lambda *p: os.path.join(self.raiz, *p)
        self.config = {
            'paths': {'results': {'base': r('results'), 'figuras': r('figures'),
                                  'analisis': r('results', 'analisis'),
                                  'figuras_analisis': r('figures', 'analisis')}},
            'transferibilidad': {'dir_resultados': r('results', 'coquimbo'),
                                 'dir_figuras': r('figures', 'coquimbo')},
        }

    def test_resultado_del_norte_va_a_su_tema(self):
        ruta = ruta_resultado(self.config, 'aptitud', 'mapa_probabilidad_aptitud.tif')
        self.assertEqual(ruta, os.path.join(self.raiz, 'results', 'aptitud',
                                            'mapa_probabilidad_aptitud.tif'))
        self.assertTrue(os.path.isdir(os.path.dirname(ruta)))   # crea la carpeta del tema

    def test_zona_repite_los_temas_en_su_carpeta(self):
        ruta = ruta_resultado(self.config, 'cruce', 'cruce_aptitud_rendimiento.json',
                              zona='transferibilidad')
        self.assertEqual(ruta, os.path.join(self.raiz, 'results', 'coquimbo', 'cruce',
                                            'cruce_aptitud_rendimiento.json'))
        self.assertEqual(ruta_figura(self.config, 'transferibilidad', 'mapa_aoa.png',
                                     zona='transferibilidad'),
                         os.path.join(self.raiz, 'figures', 'coquimbo', 'transferibilidad',
                                      'mapa_aoa.png'))

    def test_prefijo_del_motor(self):
        self.assertEqual(prefijo_rendimiento(self.config),
                         os.path.join(self.raiz, 'results', 'rendimiento', 'rendimiento'))
        self.assertEqual(prefijo_rendimiento(self.config, 'transferibilidad'),
                         os.path.join(self.raiz, 'results', 'coquimbo', 'rendimiento',
                                      'rendimiento'))

    def test_figuras_y_analisis_van_a_su_raiz(self):
        self.assertEqual(dir_figuras(self.config, 'explicabilidad'),
                         os.path.join(self.raiz, 'figures', 'explicabilidad'))
        self.assertEqual(dir_analisis(self.config, 'desacople'),
                         os.path.join(self.raiz, 'results', 'analisis', 'desacople'))
        self.assertEqual(dir_figuras_analisis(self.config, 'sensibilidad_muestreo'),
                         os.path.join(self.raiz, 'figures', 'analisis', 'sensibilidad_muestreo'))

    def test_tema_desconocido_falla_sin_crear_carpetas(self):
        with self.assertRaises(ValueError):
            ruta_resultado(self.config, 'aptitd', 'x.tif')
        self.assertFalse(os.path.exists(os.path.join(self.raiz, 'results', 'aptitd')))

    def test_zona_inexistente_falla(self):
        with self.assertRaises(KeyError):
            ruta_resultado(self.config, 'aptitud', 'x.tif', zona='no_existe')

    def test_temas_documentados(self):
        self.assertEqual(TEMAS, ('modelo', 'aptitud', 'rendimiento', 'cruce', 'explicabilidad',
                                 'postgis', 'transferibilidad'))


if __name__ == "__main__":
    unittest.main()
