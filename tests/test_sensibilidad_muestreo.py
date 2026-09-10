"""Tests unitarios para el análisis de sensibilidad al diseño de pseudo-ausencias."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.stats import kendalltau, spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FEATURES
from src.sampling import generar_dataset_muestras
from src.sensibilidad_muestreo import (
    _calcular_estadisticas_distribucion,
    FAMILIA_RED,
    FAMILIA_RECURSO,
)


class TestEstrategiasPseudoAusencias(unittest.TestCase):
    def setUp(self):
        # Geometrías sintéticas para tests rápidos
        poly_norte = Polygon([(100000, 7000000), (500000, 7000000),
                              (500000, 7500000), (100000, 7500000)])
        self.regiones_gdf = gpd.GeoDataFrame(
            {'REGION': ['Antofagasta'], 'geometry': [poly_norte]},
            crs='EPSG:32719'
        )
        # 3 plantas sintéticas separadas
        self.plantas_gdf = gpd.GeoDataFrame(
            {'REGION': ['Antofagasta', 'Antofagasta', 'Antofagasta'],
             'POTENCIAMW': [50.0, 100.0, 150.0],
             'geometry': [Point(200000, 7100000), Point(300000, 7200000), Point(400000, 7300000)]},
            crs='EPSG:32719'
        )
        self.lineas_gdf = gpd.GeoDataFrame(
            {'REGION': ['Antofagasta'],
             'geometry': [Polygon([(190000, 7050000), (410000, 7050000),
                                   (410000, 7350000), (190000, 7350000)]).exterior]},
            crs='EPSG:32719'
        )
        self.vectores = {
            'regiones': self.regiones_gdf,
            'fotovoltaicas': self.plantas_gdf,
            'lineas': self.lineas_gdf,
            'almacenamiento': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719'),
            'subestaciones': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719'),
            'areas_pobladas': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719'),
            'masas_lacustres': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719'),
            'snap': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719'),
        }
        self.criterios = {
            'ghi_min': 160,
            'slope_max': 15,
            'elev_max': 3500,
            'dist_max': 20000,
        }

    def test_estrategia_invalida_lanza_error(self):
        # Debe fallar si se pasa una estrategia desconocida
        with self.assertRaises(ValueError):
            from src.sampling import generar_dataset_muestras
            # Creando rutas_rasters dummy que no se alcanzarán a leer antes de fallar o con mock
            generar_dataset_muestras(
                self.vectores,
                {'ghi_32719': 'dummy', 'slope': 'dummy', 'aspect': 'dummy', 'dem_32719': 'dummy'},
                self.criterios,
                estrategia_negativos='estrategia_inexistente'
            )

    def test_macro_familias_cubren_todos_los_features(self):
        # Las macro-familias deben ser disjuntas y cubrir exactamente FEATURES
        union_familias = set(FAMILIA_RED + FAMILIA_RECURSO)
        self.assertEqual(union_familias, set(FEATURES))
        self.assertEqual(len(FAMILIA_RED) + len(FAMILIA_RECURSO), len(FEATURES))

    def test_calculo_estadisticas_distribucion(self):
        df_dummy = pd.DataFrame({
            'slope': [1.0, 5.0, 10.0, 15.0, 20.0],
            'ghi': [180.0, 200.0, 220.0, 240.0, 260.0],
        })
        stats = _calcular_estadisticas_distribucion(df_dummy, ['slope', 'ghi'])
        self.assertIn('slope', stats)
        self.assertIn('ghi', stats)
        self.assertEqual(stats['slope']['min'], 1.0)
        self.assertEqual(stats['slope']['max'], 20.0)
        self.assertEqual(stats['slope']['mediana'], 10.0)
        self.assertEqual(stats['ghi']['media'], 220.0)

    def test_correlacion_rangos_invariante_con_orden_identico(self):
        # Si dos vectores de importancia tienen el mismo orden relativo, Kendall y Spearman deben ser 1.0
        v1 = [40.0, 25.0, 15.0, 10.0, 5.0, 3.0, 2.0]
        v2 = [50.0, 20.0, 12.0, 8.0, 6.0, 2.5, 1.5]
        tau, _ = kendalltau(v1, v2)
        rho, _ = spearmanr(v1, v2)
        self.assertAlmostEqual(tau, 1.0, places=4)
        self.assertAlmostEqual(rho, 1.0, places=4)

    def test_tres_estrategias_con_rasters_sinteticos(self):
        import tempfile
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmpdir:
            transform = from_origin(100000, 7500000, 1000, 1000)
            meta = {
                'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
                'width': 400, 'height': 500, 'count': 1, 'crs': 'EPSG:32719',
                'transform': transform,
            }

            # Crear rasters sintéticos
            rutas_rasters = {}
            for nombre, valor_base in [('ghi_32719', 200.0), ('slope', 10.0),
                                       ('aspect', 45.0), ('dem_32719', 1500.0)]:
                p = os.path.join(tmpdir, f"{nombre}.tif")
                arr = np.full((500, 400), valor_base, dtype=np.float32)
                with rasterio.open(p, 'w', **meta) as dst:
                    dst.write(arr, 1)
                rutas_rasters[nombre] = p

            for est in ['ahp_filtrado', 'fondo_aleatorio', 'fondo_objetivo']:
                pos, neg = generar_dataset_muestras(
                    self.vectores, rutas_rasters, self.criterios,
                    ratio=2, random_state=42, estrategia_negativos=est
                )
                self.assertGreater(len(pos), 0)
                self.assertGreater(len(neg), 0)
                for f in FEATURES:
                    self.assertIn(f, neg.columns)
                self.assertTrue((neg['clase'] == 0).all())
                self.assertTrue((pos['clase'] == 1).all())


if __name__ == '__main__':
    unittest.main()
