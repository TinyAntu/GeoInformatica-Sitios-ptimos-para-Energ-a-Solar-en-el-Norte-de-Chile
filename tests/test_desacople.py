"""Pruebas unitarias para el módulo de desacople aptitud-rendimiento.

Verifica el filtrado de capas no dedicadas y la alineación/cruce de arrays
con datos sintéticos en un directorio temporal, sin depender de archivos pesados.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.desacople_aptitud_rendimiento import alinear_y_cruzar
from src.auditoria_infraestructura import filtrar_infraestructura, asignar_tipo_por_subestacion


class TestDesacople(unittest.TestCase):

    def test_filtrado_dedicadas(self):
        """Verifica que el filtrado 'sin_dedicadas' descarte correctamente las entidades dedicadas."""
        subest = gpd.GeoDataFrame({
            'TIPO': ['DEDICADO', 'ZONAL', 'NACIONAL'],
            'NOMBRE': ['SE_A', 'SE_B', 'SE_C'],
            'REGION': ['02', '02', '03'],
            'geometry': [Point(300000, 7000000), Point(305000, 7000000), Point(310000, 7000000)],
        }, crs='EPSG:32719')

        visibles = filtrar_infraestructura(subest, 'sin_dedicadas')
        self.assertEqual(len(visibles), 2)
        self.assertNotIn('DEDICADO', visibles['TIPO'].values)

    def test_asignar_tipo_almacenamiento(self):
        """Verifica que el almacenamiento herede el tipo DEDICADO de su subestación de conexión."""
        subest = gpd.GeoDataFrame({
            'TIPO': ['DEDICADO', 'NACIONAL'],
            'NOMBRE': ['SE_SOLAR', 'SE_TRONCAL'],
            'geometry': [Point(300000, 7000000), Point(305000, 7000000)],
        }, crs='EPSG:32719')

        almac = gpd.GeoDataFrame({
            'NOMBRE_SE': ['SE_SOLAR', 'SE_TRONCAL', 'SE_DESCONOCIDA'],
            'geometry': [Point(300010, 7000010), Point(305010, 7000010), Point(310010, 7000010)],
        }, crs='EPSG:32719')

        con_tipo = asignar_tipo_por_subestacion(almac, subest)
        self.assertEqual(con_tipo.loc[0, 'TIPO'], 'DEDICADO')
        self.assertEqual(con_tipo.loc[1, 'TIPO'], 'NACIONAL')
        self.assertEqual(con_tipo.loc[2, 'TIPO'], 'S/I')

        filtradas = filtrar_infraestructura(con_tipo, 'sin_dedicadas')
        self.assertEqual(len(filtradas), 2)
        self.assertNotIn('DEDICADO', filtradas['TIPO'].values)

    def _rasters(self, tmpdir, aptitud, rendimiento):
        """Escribe aptitud y rendimiento en la misma grilla de 100 m (EPSG:32719)."""
        import rasterio
        from rasterio.transform import from_origin
        rutas = {}
        for nombre, arr in (('aptitud', aptitud), ('rendimiento', rendimiento)):
            rutas[nombre] = os.path.join(tmpdir, f'{nombre}.tif')
            meta = {'driver': 'GTiff', 'dtype': 'float32', 'count': 1, 'crs': 'EPSG:32719',
                    'transform': from_origin(300000, 7000000, 100, 100),
                    'width': arr.shape[1], 'height': arr.shape[0], 'nodata': -9999.0}
            with rasterio.open(rutas[nombre], 'w', **meta) as dst:
                dst.write(arr.astype(np.float32), 1)
        return rutas

    def test_excluye_celdas_sin_ghi_y_zonas_de_exclusion(self):
        """Rendimiento 0 (sin GHI en el motor) y zonas de exclusión no entran al análisis."""
        aptitud = np.full((10, 10), 0.5, dtype=np.float32)
        aptitud[0, :] = 0.9                       # una fila de celdas aptas
        rendimiento = np.full((10, 10), 2000.0, dtype=np.float32)
        rendimiento[:, 9] = 0.0                   # última columna sin dato de GHI
        exclusion = np.zeros((10, 10), dtype=bool)
        exclusion[5, :5] = True                   # cinco celdas en una zona de exclusión
        with tempfile.TemporaryDirectory() as tmp:
            rutas = self._rasters(tmp, aptitud, rendimiento)
            cruce = alinear_y_cruzar(rutas['aptitud'], rutas['rendimiento'], umbral=0.7,
                                     mascara_exclusion=exclusion)
        stats = cruce['stats']
        self.assertEqual(stats['celdas_excluidas_sin_ghi'], 10)
        self.assertEqual(stats['celdas_excluidas_zonas_exclusion'], 5)
        self.assertEqual(stats['celdas_validas'], 100 - 10 - 5)
        self.assertEqual(stats['celdas_aptas'], 9)  # la fila apta sin su celda sin GHI
        # Sin las celdas en cero, la media regional es exactamente la del terreno con dato.
        self.assertEqual(stats['rendimiento_region']['media'], 2000.0)

    def test_sin_celdas_aptas_falla_con_mensaje_claro(self):
        """Sin celdas aptas no hay desacople que medir: error explícito, no uno de scipy."""
        from src.desacople_aptitud_rendimiento import calcular_estadisticas_desacople
        aptitud = np.full((10, 10), 0.3, dtype=np.float32)       # nada llega a 0,70
        rendimiento = np.full((10, 10), 2000.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            rutas = self._rasters(tmp, aptitud, rendimiento)
            cruce = alinear_y_cruzar(rutas['aptitud'], rutas['rendimiento'], umbral=0.7)
        with self.assertRaisesRegex(ValueError, "celdas aptas"):
            calcular_estadisticas_desacople(cruce, n_bootstrap=10)

    def test_asignar_bloques_a_celdas(self):
        """Verifica la asignación de identificadores de bloques regulares de 30 km."""
        from src.desacople_aptitud_rendimiento import asignar_bloques_a_celdas
        from affine import Affine

        transform = Affine(100.0, 0.0, 300000.0, 0.0, -100.0, 7000000.0)
        mascara = np.zeros((10, 10), dtype=bool)
        mascara[0, 0] = True
        mascara[5, 5] = True

        bloques = asignar_bloques_a_celdas(transform, mascara, tamano_bloque_km=30.0)
        self.assertEqual(len(bloques), 2)
        # Ambas celdas caen dentro del mismo bloque de 30 km
        self.assertEqual(bloques[0], bloques[1])

    def test_calcular_estadisticas_y_exportar_csv(self):
        """Verifica el cálculo de estadísticas de desacople, correlación y exportación a CSV."""
        from src.desacople_aptitud_rendimiento import (
            calcular_estadisticas_desacople,
            exportar_tabla_csv,
        )
        from affine import Affine

        n_rows, n_cols = 40, 40
        transform = Affine(1000.0, 0.0, 300000.0, 0.0, -1000.0, 7000000.0)
        
        # Simular aptitud y rendimiento con correlación positiva
        rng = np.random.default_rng(42)
        aptitud = rng.uniform(0.1, 0.95, size=(n_rows, n_cols)).astype(np.float32)
        rendimiento = (1800.0 + aptitud * 300.0 + rng.normal(0, 30, size=(n_rows, n_cols))).astype(np.float32)
        
        base_valida = np.ones((n_rows, n_cols), dtype=bool)
        aptas = base_valida & (aptitud >= 0.70)

        cruce = {
            'aptitud': aptitud,
            'rendimiento': rendimiento,
            'base_valida': base_valida,
            'aptas': aptas,
            'transform': transform,
            'crs': 'EPSG:32719',
            'stats': {
                'umbral_probabilidad': 0.70,
                'celdas_validas': int(base_valida.sum()),
                'celdas_aptas': int(aptas.sum()),
                'pct_aptas': round(100.0 * float(aptas.sum()) / float(base_valida.sum()), 2),
                'rendimiento_region': {'media': float(rendimiento.mean()), 'mediana': float(np.median(rendimiento))},
                'rendimiento_aptas': {'media': float(rendimiento[aptas].mean()), 'mediana': float(np.median(rendimiento[aptas]))},
                'ganancia_media_pct': 5.2,
                'ganancia_mediana_pct': 4.8,
            }
        }

        res = calcular_estadisticas_desacople(cruce, tamano_bloque_km=15.0, n_bootstrap=50, random_state=42)
        
        self.assertIn('region', res)
        self.assertIn('aptas', res)
        self.assertIn('spearman_celda', res['region'])
        self.assertIn('spearman_bloque', res['region'])
        self.assertGreater(res['region']['spearman_celda']['rho'], 0.0)
        self.assertGreater(res['region']['spearman_bloque']['rho'], 0.0)
        self.assertIn('ic_95_inferior', res['region']['spearman_bloque'])
        self.assertIn('ic_95_superior', res['region']['spearman_bloque'])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, 'tabla_desacople_test.csv')
            df = exportar_tabla_csv(res, out_csv)
            self.assertTrue(os.path.exists(out_csv))
            self.assertGreater(len(df), 5)
            self.assertIn('metrica', df.columns)
            self.assertIn('valor', df.columns)


if __name__ == '__main__':
    unittest.main()
