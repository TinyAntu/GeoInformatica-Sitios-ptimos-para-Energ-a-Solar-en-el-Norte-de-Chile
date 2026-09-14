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
    cargar_hiperparametros_del_modelo,
    ESTRATEGIAS,
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
        # Subestaciones reales en el fixture: son el "grupo objetivo" del que la estrategia
        # TGB extrae sus pseudo-ausencias. Con la capa vacía esa estrategia no puede correr.
        self.subestaciones_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(250000, 7150000), Point(350000, 7250000)]},
            crs='EPSG:32719'
        )
        self.vectores = {
            'regiones': self.regiones_gdf,
            'fotovoltaicas': self.plantas_gdf,
            'lineas': self.lineas_gdf,
            'almacenamiento': gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719'),
            'subestaciones': self.subestaciones_gdf,
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

    def _rasters_con_gradiente(self, tmpdir):
        """Rásters sintéticos con GRADIENTE, no constantes.

        Con valores constantes que cumplen todos los criterios, los filtros geofísicos nunca
        se activan y 'ahp_filtrado' colapsa al mismo pool que 'sin_filtros_geofisicos': el
        test pasaba aunque la lógica de filtros estuviera rota. El gradiente hace que una
        parte del territorio quede fuera de rango, que es lo que separa los diseños.
        """
        import rasterio
        from rasterio.transform import from_origin

        h, w = 500, 400
        transform = from_origin(100000, 7500000, 1000, 1000)
        meta = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
                'width': w, 'height': h, 'count': 1, 'crs': 'EPSG:32719',
                'transform': transform}

        filas = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        columnas = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

        capas = {
            # GHI de 120 (bajo el mínimo de 160) a 280 según la fila
            'ghi_32719': 120.0 + 160.0 * filas + 0.0 * columnas,
            # Pendiente de 0 a 30 (el máximo permitido es 15) según la columna
            'slope': 0.0 + 30.0 * columnas + 0.0 * filas,
            'aspect': 45.0 + 0.0 * filas + 0.0 * columnas,
            # Elevación de 500 a 4500 (el máximo permitido es 3500)
            'dem_32719': 500.0 + 4000.0 * filas + 0.0 * columnas,
        }

        rutas = {}
        for nombre, arr in capas.items():
            p = os.path.join(tmpdir, f"{nombre}.tif")
            with rasterio.open(p, 'w', **meta) as dst:
                dst.write(np.ascontiguousarray(arr, dtype=np.float32), 1)
            rutas[nombre] = p
        return rutas

    def test_cada_estrategia_produce_un_pool_valido(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            rutas = self._rasters_con_gradiente(tmpdir)
            for est in ESTRATEGIAS:
                pos, neg = generar_dataset_muestras(
                    self.vectores, rutas, self.criterios,
                    ratio=2, random_state=42, estrategia_negativos=est
                )
                self.assertGreater(len(pos), 0, f"{est}: sin positivas")
                self.assertGreater(len(neg), 0, f"{est}: pool de negativos vacío")
                for f in FEATURES:
                    self.assertIn(f, neg.columns, f"{est}: falta la feature {f}")
                self.assertTrue((neg['clase'] == 0).all())
                self.assertTrue((pos['clase'] == 1).all())

    def test_los_disenos_producen_pools_distintos(self):
        """El punto entero del análisis: si dos diseños dieran el mismo pool, no hay
        sensibilidad que medir. Este test falla si alguien neutraliza los filtros."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            rutas = self._rasters_con_gradiente(tmpdir)
            pools = {}
            for est in ESTRATEGIAS:
                _, neg = generar_dataset_muestras(
                    self.vectores, rutas, self.criterios,
                    ratio=2, random_state=42, estrategia_negativos=est
                )
                pools[est] = neg

            # El fondo sin filtros técnicos debe ser estrictamente mayor que la línea base.
            self.assertGreater(len(pools['fondo_aleatorio']), len(pools['ahp_filtrado']),
                               "El fondo aleatorio no puede tener menos negativos que la línea base")
            # Soltar los filtros geofísicos solo puede ampliar el pool, nunca reducirlo.
            self.assertGreaterEqual(len(pools['sin_filtros_geofisicos']), len(pools['ahp_filtrado']))
            # Y con gradiente debe ampliarlo de verdad, no quedar idéntico.
            self.assertNotEqual(len(pools['sin_filtros_geofisicos']), len(pools['ahp_filtrado']),
                                "La ablación geofísica quedó idéntica a la línea base: "
                                "los filtros no se están aplicando")

    def test_grupo_objetivo_muestrea_cerca_de_la_infraestructura(self):
        """El TGB debe concentrar sus negativos junto a subestaciones/almacenamiento."""
        import tempfile
        from src.sampling import TGB_RADIO_M

        with tempfile.TemporaryDirectory() as tmpdir:
            rutas = self._rasters_con_gradiente(tmpdir)
            _, neg_tgb = generar_dataset_muestras(
                self.vectores, rutas, self.criterios,
                ratio=2, random_state=42, estrategia_negativos='grupo_objetivo')
            _, neg_base = generar_dataset_muestras(
                self.vectores, rutas, self.criterios,
                ratio=2, random_state=42, estrategia_negativos='fondo_aleatorio')

        self.assertGreater(len(neg_tgb), 0)
        d_tgb = neg_tgb.geometry.apply(lambda g: self.subestaciones_gdf.distance(g).min())
        self.assertLessEqual(float(d_tgb.max()), TGB_RADIO_M + 1.0,
                             "Hay negativos del TGB fuera del radio de la infraestructura")

        # Y debe estar sistemáticamente más cerca que el fondo aleatorio: si no, no está
        # replicando ningún sesgo de prospección y el diseño no aporta nada.
        d_base = neg_base.geometry.apply(lambda g: self.subestaciones_gdf.distance(g).min())
        self.assertLess(float(d_tgb.median()), float(d_base.median()))

    def test_grupo_objetivo_aborta_sin_infraestructura(self):
        """Sin subestaciones ni almacenamiento el TGB no tiene de dónde muestrear: debe
        fallar en voz alta, no caer en silencio a un muestreo uniforme."""
        import tempfile
        vacio = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32719')
        vectores = dict(self.vectores, subestaciones=vacio, almacenamiento=vacio)
        with tempfile.TemporaryDirectory() as tmpdir:
            rutas = self._rasters_con_gradiente(tmpdir)
            with self.assertRaises(ValueError):
                generar_dataset_muestras(
                    vectores, rutas, self.criterios,
                    ratio=2, random_state=42, estrategia_negativos='grupo_objetivo')

    def test_alias_en_desuso_sigue_funcionando(self):
        """'fondo_objetivo' era el nombre original; no debe romper a quien aún lo use."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            rutas = self._rasters_con_gradiente(tmpdir)
            _, neg_alias = generar_dataset_muestras(
                self.vectores, rutas, self.criterios,
                ratio=2, random_state=42, estrategia_negativos='fondo_objetivo')
            _, neg_nuevo = generar_dataset_muestras(
                self.vectores, rutas, self.criterios,
                ratio=2, random_state=42, estrategia_negativos='sin_filtros_geofisicos')
            self.assertEqual(len(neg_alias), len(neg_nuevo))


class TestHiperparametrosYReporte(unittest.TestCase):
    """Cubre los dos defectos que hacían inválido o frágil el reporte."""

    def test_hiperparametros_salen_del_modelo_no_de_constantes(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ruta = os.path.join(tmp, 'model_rf_metrics.json')
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump({'best_params': {'n_estimators': 600, 'max_depth': 10,
                                           'min_samples_leaf': 2}}, f)
            params, origen = cargar_hiperparametros_del_modelo(ruta)
        self.assertEqual(params['n_estimators'], 600)
        self.assertEqual(params['max_depth'], 10)
        self.assertEqual(params['min_samples_leaf'], 2)
        # Optuna no lo elige, pero src/modeling.py lo fija: sin él los modelos no son comparables.
        self.assertEqual(params['class_weight'], 'balanced')
        self.assertIn('Optuna', origen)

    def test_falla_ruidosamente_si_no_hay_modelo_entrenado(self):
        """Antes se caía en constantes arbitrarias en silencio; ahora debe abortar."""
        with self.assertRaises(FileNotFoundError):
            cargar_hiperparametros_del_modelo('/ruta/que/no/existe/model_rf_metrics.json')

    def test_ratio_none_no_rompe_el_formateo(self):
        """Regresión: `.get(clave, 0)` no protege si la clave existe con valor None."""
        fam = {'acceso_red_pct': 100.0, 'recurso_topografia_pct': 0.0,
               'ratio_red_vs_recurso': None}
        ratio = fam.get('ratio_red_vs_recurso')
        texto = f"{ratio:.2f}x" if ratio is not None else "n/d (recurso = 0%)"
        self.assertEqual(texto, "n/d (recurso = 0%)")


if __name__ == '__main__':
    unittest.main()
