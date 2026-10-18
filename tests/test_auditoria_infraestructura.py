"""Tests de la auditoría de fuga temporal en las distancias a la red (estudio aparte del pipeline).

Usan geometrías sintéticas en EPSG:32719 con distancias conocidas: no dependen de data/.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import metricas_topk
from src.auditoria_infraestructura import (
    CAPA_POR_FEATURE,
    EXPERIMENTOS,
    variantes_del_experimento,
    tiene_mapa,
    asignar_tipo_por_subestacion,
    preparar_infraestructura,
    asignar_fechas_referencia,
    filtrar_infraestructura,
    distancia_a_infraestructura,
    diagnosticar_fuga,
    auc_univariado,
)

CRS = 'EPSG:32719'


def _subestaciones():
    """Tres subestaciones sobre el eje x, a 1, 2 y 5 km del origen (300000, 7000000).

    - A (1 km): DEDICADA, entró en 2020, mismo dueño que la planta.
    - B (2 km): ZONAL, entró en 2010.
    - C (5 km): NACIONAL, sin fecha.
    """
    capa = gpd.GeoDataFrame(
        {
            'REGION': ['02', '02', '02'],
            'TIPO': ['DEDICADO', 'ZONAL', 'NACIONAL'],
            'PROPIEDAD': ['SOLAR SPA', 'TRANSELEC S.A.', 'CEN'],
            'F_OPERACIO': [pd.Timestamp('2020-03-01'), pd.Timestamp('2010-01-01'), pd.NaT],
            'geometry': [Point(301000, 7000000), Point(302000, 7000000), Point(305000, 7000000)],
        },
        crs=CRS,
    )
    return preparar_infraestructura(capa, codigos=['02', '03'])


def _lineas():
    """Dos líneas verticales (norte-sur) a 500 m y 3 km al este del origen.

    - L1 (500 m): DEDICADA, 220 kV, entró en 2021, mismo dueño que la planta.
    - L2 (3 km): NACIONAL, entró en 2005.
    La capa de líneas usa nombres de región en vez de códigos, como la capa real.
    """
    capa = gpd.GeoDataFrame(
        {
            'REGION': ['ANTOFAGASTA', 'ANTOFAGASTA'],
            'TIPO': ['DEDICADO', 'NACIONAL'],
            'PROPIEDAD': ['SOLAR SPA', 'TRANSELEC S.A.'],
            'F_OPERACIO': [pd.Timestamp('2021-05-01'), pd.Timestamp('2005-01-01')],
            'geometry': [LineString([(300500, 6990000), (300500, 7010000)]),
                         LineString([(303000, 6990000), (303000, 7010000)])],
        },
        crs=CRS,
    )
    return preparar_infraestructura(capa, codigos=['02', '03', 'Antofagasta', 'Atacama'])


class TestPrepararYFiltrar(unittest.TestCase):

    def setUp(self):
        self.subest = _subestaciones()

    def test_preparar_filtra_por_region_y_agrega_fecha(self):
        capa = gpd.GeoDataFrame(
            {'REGION': ['02', '04'], 'TIPO': ['ZONAL', 'ZONAL'], 'PROPIEDAD': ['X', 'Y'],
             'F_OPERACIO': ['2015-01-01', '2016-01-01'],
             'geometry': [Point(0, 0), Point(1, 1)]},
            crs=CRS,
        )
        preparada = preparar_infraestructura(capa, codigos=[2, 3, '02', '03'])
        self.assertEqual(len(preparada), 1, "La subestación de Coquimbo (04) debe quedar fuera.")
        self.assertEqual(preparada['f_op'].iloc[0], pd.Timestamp('2015-01-01'))

    def test_preparar_lineas_por_nombre_de_region(self):
        self.assertEqual(len(_lineas()), 2)

    def test_todas_no_filtra(self):
        self.assertEqual(len(filtrar_infraestructura(self.subest, 'todas')), 3)

    def test_sin_dedicadas_excluye_dedicado(self):
        visibles = filtrar_infraestructura(self.subest, 'sin_dedicadas')
        self.assertEqual(sorted(visibles['TIPO']), ['NACIONAL', 'ZONAL'])

    def test_sin_dedicadas_en_lineas(self):
        visibles = filtrar_infraestructura(_lineas(), 'sin_dedicadas')
        self.assertEqual(list(visibles['TIPO']), ['NACIONAL'])

    def test_previas_aplica_margen_y_excluye_sin_fecha(self):
        # Planta de 2020-06: A (2020-03) cae dentro del margen de 12 meses -> no es previa.
        # C no tiene fecha -> no se puede afirmar que existía -> se excluye.
        visibles = filtrar_infraestructura(self.subest, 'previas', pd.Timestamp('2020-06-01'), 12)
        self.assertEqual(list(visibles['TIPO']), ['ZONAL'])

    def test_previas_sin_margen_acepta_la_del_mismo_anio(self):
        visibles = filtrar_infraestructura(self.subest, 'previas', pd.Timestamp('2020-06-01'), 0)
        self.assertEqual(sorted(visibles['TIPO']), ['DEDICADO', 'ZONAL'])

    def test_previas_exige_fecha_de_corte(self):
        with self.assertRaises(ValueError):
            filtrar_infraestructura(self.subest, 'previas')

    def test_variante_desconocida(self):
        with self.assertRaises(ValueError):
            filtrar_infraestructura(self.subest, 'inventada')


class TestDistancias(unittest.TestCase):

    def setUp(self):
        self.subest = _subestaciones()
        self.muestras = gpd.GeoDataFrame(
            {'clase': [1, 0], 'geometry': [Point(300000, 7000000), Point(300000, 7000000)]},
            crs=CRS,
        )

    def test_distancias_conocidas_por_variante(self):
        d_todas = distancia_a_infraestructura(self.muestras, self.subest, 'todas')
        d_sin_ded = distancia_a_infraestructura(self.muestras, self.subest, 'sin_dedicadas')
        np.testing.assert_allclose(d_todas, [1000.0, 1000.0])
        np.testing.assert_allclose(d_sin_ded, [2000.0, 2000.0])

    def test_previas_usa_la_fecha_de_cada_muestra(self):
        fechas = pd.Series([pd.Timestamp('2022-01-01'), pd.Timestamp('2020-06-01')],
                           index=self.muestras.index)
        d = distancia_a_infraestructura(self.muestras, self.subest, 'previas', fechas, 12)
        # 2022: A (2020-03) ya es previa -> 1 km. 2020-06: solo B es previa -> 2 km.
        np.testing.assert_allclose(d, [1000.0, 2000.0])

    def test_previas_sin_red_visible_da_nan(self):
        fechas = pd.Series([pd.Timestamp('2005-01-01')] * 2, index=self.muestras.index)
        d = distancia_a_infraestructura(self.muestras, self.subest, 'previas', fechas, 12)
        self.assertTrue(np.isnan(d).all())

    def test_distancia_perpendicular_a_lineas(self):
        lineas = _lineas()
        fechas = pd.Series([pd.Timestamp('2021-12-01'), pd.Timestamp('2023-01-01')],
                           index=self.muestras.index)
        np.testing.assert_allclose(
            distancia_a_infraestructura(self.muestras, lineas, 'todas'), [500.0, 500.0])
        np.testing.assert_allclose(
            distancia_a_infraestructura(self.muestras, lineas, 'sin_dedicadas'), [3000.0, 3000.0])
        # 2021-12: L1 (2021-05) no cumple el margen -> 3 km. 2023: L1 ya es previa -> 500 m.
        np.testing.assert_allclose(
            distancia_a_infraestructura(self.muestras, lineas, 'previas', fechas, 12),
            [3000.0, 500.0])


class TestFechasYDiagnostico(unittest.TestCase):

    def setUp(self):
        self.subest = _subestaciones()
        self.plantas = gpd.GeoDataFrame(
            {'PROPIEDAD': ['SOLAR SPA', 'OTRA SPA'],
             'F_OPERACIO': [pd.Timestamp('2020-06-01', tz='UTC'), pd.NaT],
             'geometry': [Point(300000, 7000000), Point(400000, 7100000)]},
            crs=CRS,
        )
        self.muestras = gpd.GeoDataFrame(
            {'clase': [1, 1, 0, 0, 0],
             'geometry': [Point(300000, 7000000), Point(400000, 7100000),
                          Point(350000, 7050000), Point(360000, 7060000),
                          Point(370000, 7070000)]},
            crs=CRS,
        )

    def test_empareja_positivas_y_sortea_el_resto(self):
        fechas = asignar_fechas_referencia(self.muestras, self.plantas, random_state=42)
        self.assertEqual(fechas.loc[0, 'fecha_ref'], pd.Timestamp('2020-06-01'))
        self.assertFalse(fechas.loc[0, 'fecha_imputada'])
        self.assertEqual(fechas.loc[0, 'propiedad_planta'], 'SOLAR SPA')
        self.assertAlmostEqual(fechas.loc[0, 'dist_emparejamiento_m'], 0.0)
        # La planta sin fecha y los 3 negativos reciben una fecha sorteada entre las reales.
        self.assertEqual(int(fechas['fecha_imputada'].sum()), 4)
        self.assertTrue((fechas['fecha_ref'] == pd.Timestamp('2020-06-01')).all())

    def test_sorteo_reproducible(self):
        plantas = self.plantas.copy()
        plantas['F_OPERACIO'] = [pd.Timestamp('2015-01-01', tz='UTC'),
                                 pd.Timestamp('2021-01-01', tz='UTC')]
        a = asignar_fechas_referencia(self.muestras, plantas, random_state=42)
        b = asignar_fechas_referencia(self.muestras, plantas, random_state=42)
        pd.testing.assert_series_equal(a['fecha_ref'], b['fecha_ref'])
        fechas_posibles = {pd.Timestamp('2015-01-01'), pd.Timestamp('2021-01-01')}
        self.assertTrue(set(a['fecha_ref']) <= fechas_posibles)

    def test_diagnostico_detecta_las_tres_huellas(self):
        fechas = asignar_fechas_referencia(self.muestras, self.plantas, random_state=42)
        diag = diagnosticar_fuga(self.muestras, self.subest, fechas, margen_meses=12)
        # Solo la planta 0 tiene fecha real. Su más cercana es A: dedicada, no previa
        # (2020-03 > 2019-06) y del mismo dueño.
        self.assertEqual(diag['n_plantas_con_fecha'], 1)
        self.assertEqual(diag['pct_mas_cercana_dedicada'], 100.0)
        self.assertEqual(diag['pct_mas_cercana_no_previa'], 100.0)
        self.assertEqual(diag['pct_mas_cercana_mismo_dueno'], 100.0)

    def test_diagnostico_sobre_lineas(self):
        fechas = asignar_fechas_referencia(self.muestras, self.plantas, random_state=42)
        diag = diagnosticar_fuga(self.muestras, _lineas(), fechas, margen_meses=12)
        # La línea más cercana a la planta 0 es L1: dedicada, de 2021 (posterior a la planta
        # de 2020-06) y del mismo dueño.
        self.assertEqual(diag['tipos_mas_cercana'], {'DEDICADO': 1})
        self.assertEqual(diag['pct_mas_cercana_no_previa'], 100.0)
        self.assertEqual(diag['pct_mas_cercana_mismo_dueno'], 100.0)


class TestExperimentos(unittest.TestCase):

    def test_linea_base_no_modifica_ninguna_feature(self):
        self.assertEqual(set(variantes_del_experimento('linea_base').values()), {'todas'})

    def test_ambas_modifica_red_y_deja_el_almacenamiento(self):
        for experimento in ('ambas_sin_dedicadas', 'ambas_previas'):
            variantes = variantes_del_experimento(experimento)
            self.assertNotEqual(variantes['dist_subestaciones'], 'todas', experimento)
            self.assertNotEqual(variantes['dist_transmision'], 'todas', experimento)
            self.assertEqual(variantes['dist_almacen'], 'todas', experimento)

    def test_tres_modifica_todas_las_distancias(self):
        for experimento in ('tres_sin_dedicadas', 'tres_previas'):
            self.assertNotIn('todas', variantes_del_experimento(experimento).values(), experimento)

    def test_cada_experimento_declara_solo_features_auditadas(self):
        for experimento, declaradas in EXPERIMENTOS.items():
            self.assertTrue(set(declaradas) <= set(CAPA_POR_FEATURE), experimento)

    def test_solo_los_experimentos_sin_previas_tienen_mapa(self):
        con_mapa = {e for e in EXPERIMENTOS if tiene_mapa(e)}
        self.assertEqual(con_mapa, {'linea_base', 'sub_sin_dedicadas', 'lin_sin_dedicadas',
                                    'alm_sin_dedicadas', 'ambas_sin_dedicadas',
                                    'tres_sin_dedicadas'})


class TestTipoDelAlmacenamiento(unittest.TestCase):
    """El almacenamiento hereda el TIPO de su subestación de conexión (NOMBRE_SE)."""

    def setUp(self):
        self.subestaciones = gpd.GeoDataFrame(
            {'NOMBRE': ['S/E Planta Sol', 'S/E Zonal Norte'], 'TIPO': ['DEDICADO', 'ZONAL'],
             'geometry': [Point(0, 0), Point(1, 1)]},
            crs=CRS,
        )
        self.almacenamiento = gpd.GeoDataFrame(
            {'REGION': ['02', '02', '02'],
             'NOMBRE_SE': [' s/e planta sol ', 'S/E ZONAL NORTE', 'S/E INEXISTENTE'],
             'PROPIEDAD': ['SOLAR SPA', 'X', 'Y'],
             'F_OPERACIO': [pd.Timestamp('2024-01-01')] * 3,
             'geometry': [Point(10, 10), Point(20, 20), Point(30, 30)]},
            crs=CRS,
        )

    def test_hereda_tipo_por_nombre_normalizado(self):
        con_tipo = asignar_tipo_por_subestacion(self.almacenamiento, self.subestaciones)
        self.assertEqual(list(con_tipo['TIPO']), ['DEDICADO', 'ZONAL', 'S/I'])

    def test_sin_dedicadas_descarta_el_bess_de_la_central(self):
        con_tipo = asignar_tipo_por_subestacion(self.almacenamiento, self.subestaciones)
        capa = preparar_infraestructura(con_tipo, codigos=['02'])
        visibles = filtrar_infraestructura(capa, 'sin_dedicadas')
        self.assertEqual(list(visibles['NOMBRE_SE']), ['S/E ZONAL NORTE', 'S/E INEXISTENTE'])


class TestOptimizacion(unittest.TestCase):
    """La optimización por experimento debe ser reproducible y respetar el espacio de búsqueda."""

    OPTUNA_CONFIG = {
        'n_trials': 3,
        'n_estimators': {'min': 10, 'max': 30, 'step': 10},
        'max_depth': {'min': 2, 'max': 4},
        'min_samples_leaf': {'min': 1, 'max': 3},
    }

    def setUp(self):
        # 40 puntos separados 40 km entre sí (cada uno en su propio bloque de 30 km), en dos
        # regiones, con positivas algo más cerca de la red para que haya señal que aprender.
        rng = np.random.default_rng(0)
        n = 40
        clase = np.array([1 if i % 4 == 0 else 0 for i in range(n)])
        datos = {f: rng.normal(size=n) for f in ('slope', 'ghi', 'elev', 'northness')}
        for f in CAPA_POR_FEATURE:
            datos[f] = rng.uniform(5000, 20000, size=n) - clase * 4000
        self.muestras = gpd.GeoDataFrame(
            {**datos, 'clase': clase,
             'REGION': ['Antofagasta' if i < n // 2 else 'Atacama' for i in range(n)]},
            geometry=[Point(200000 + 40000 * i, 7000000) for i in range(n)],
            crs=CRS,
        )

    def test_reproducible_y_dentro_del_espacio(self):
        from src.auditoria_infraestructura import optimizar_hiperparametros

        a, brier_a = optimizar_hiperparametros(self.muestras, self.OPTUNA_CONFIG, 30, 42)
        b, brier_b = optimizar_hiperparametros(self.muestras, self.OPTUNA_CONFIG, 30, 42)
        self.assertEqual(a, b, "La misma semilla debe dar los mismos hiperparámetros.")
        # El RF entrena en paralelo (n_jobs=-1) y el orden de la suma en punto flotante
        # varía entre corridas: el Brier coincide hasta ~1e-16, no bit a bit.
        self.assertAlmostEqual(brier_a, brier_b, places=10)
        self.assertEqual(a['class_weight'], 'balanced')
        self.assertIn(a['n_estimators'], (10, 20, 30))
        self.assertTrue(2 <= a['max_depth'] <= 4)
        self.assertTrue(1 <= a['min_samples_leaf'] <= 3)
        self.assertTrue(0.0 <= brier_a <= 1.0)


class TestAucUnivariado(unittest.TestCase):

    def test_auc_perfecto_si_las_positivas_estan_mas_cerca(self):
        distancia = np.array([100.0, 200.0, 5000.0, 9000.0])
        clase = np.array([1, 1, 0, 0])
        self.assertEqual(auc_univariado(distancia, clase), 1.0)

    def test_ignora_nan(self):
        distancia = np.array([100.0, np.nan, 5000.0, 9000.0])
        clase = np.array([1, 1, 0, 0])
        self.assertEqual(auc_univariado(distancia, clase), 1.0)


class TestCacheGrillaPorCapa(unittest.TestCase):
    """La caché de metricas_topk no debe mezclar grillas construidas con capas distintas."""

    def _config(self, ruta_subestaciones, ruta_lineas='lineas.shp'):
        return {'paths': {'raw': {'vectores': {
            'lineas': ruta_lineas, 'almacenamiento': 'almacen.shp',
            'subestaciones': ruta_subestaciones}}}}

    def test_distinta_capa_no_comparte_cache(self):
        metricas_topk._CACHE_GRILLA.clear()
        construidas = []

        def construir_falso(config, resolucion_m):
            vectores = config['paths']['raw']['vectores']
            clave = (vectores['subestaciones'], vectores['lineas'])
            construidas.append(clave)
            return clave

        with mock.patch('src.explainability_spatial._construir_features', construir_falso):
            metricas_topk._grilla_features(self._config('todas.shp'), 500)
            metricas_topk._grilla_features(self._config('sin_dedicadas.gpkg'), 500)
            metricas_topk._grilla_features(self._config('todas.shp', 'lineas_sin.gpkg'), 500)
            metricas_topk._grilla_features(self._config('todas.shp'), 500)
        metricas_topk._CACHE_GRILLA.clear()

        self.assertEqual(construidas, [('todas.shp', 'lineas.shp'),
                                       ('sin_dedicadas.gpkg', 'lineas.shp'),
                                       ('todas.shp', 'lineas_sin.gpkg')],
                         "La misma combinación de capas debe reutilizar la caché; otra, no.")


if __name__ == '__main__':
    unittest.main()
