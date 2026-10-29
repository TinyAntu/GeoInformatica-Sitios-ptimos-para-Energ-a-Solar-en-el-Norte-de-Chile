"""Tests del control de fuga en las capas de infraestructura (src/infraestructura.py).

Datos sintéticos en EPSG:32719, sin depender de data/.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import FEATURES
from src.infraestructura import (
    excluir_dedicadas,
    control_fuga_activo,
    rutas_vectores_efectivas,
    generar_capas_sin_dedicadas,
)
from src.sampling import generar_dataset_muestras
from src.auditoria_infraestructura import experimento_del_pipeline

CRS = 'EPSG:32719'


def _config(excluir, dir_procesadas='data/processed/infraestructura', vectores=None):
    return {
        'paths': {
            'raw': {'vectores': vectores or {
                'regiones': 'regiones.shp', 'fotovoltaicas': 'plantas.gdb',
                'lineas': 'lineas.shp', 'almacenamiento': 'almacen.shp',
                'subestaciones': 'subest.shp', 'snap': 'snap.shp',
            }},
            'processed': {'infraestructura': dir_procesadas},
        },
        'control_fuga': {'excluir_dedicadas': excluir},
    }


class TestReglaYRutas(unittest.TestCase):

    def test_excluir_dedicadas(self):
        capa = gpd.GeoDataFrame({'TIPO': ['DEDICADO', ' dedicado ', 'ZONAL', 'S/I'],
                                 'geometry': [Point(i, 0) for i in range(4)]}, crs=CRS)
        self.assertEqual(list(excluir_dedicadas(capa)['TIPO']), ['ZONAL', 'S/I'])

    def test_rutas_sin_control_son_las_crudas(self):
        config = _config(False)
        self.assertFalse(control_fuga_activo(config))
        self.assertEqual(rutas_vectores_efectivas(config), config['paths']['raw']['vectores'])
        self.assertEqual(experimento_del_pipeline(config), 'linea_base')

    def test_rutas_con_control_reemplazan_solo_la_infraestructura(self):
        config = _config(True, dir_procesadas='proc/infra')
        rutas = rutas_vectores_efectivas(config)
        self.assertEqual(rutas['lineas'], os.path.join('proc/infra', 'lineas.gpkg'))
        self.assertEqual(rutas['subestaciones'], os.path.join('proc/infra', 'subestaciones.gpkg'))
        self.assertEqual(rutas['almacenamiento'], os.path.join('proc/infra', 'almacenamiento.gpkg'))
        # Regiones, plantas y exclusiones no cambian, y el config original queda intacto.
        self.assertEqual(rutas['regiones'], 'regiones.shp')
        self.assertEqual(rutas['snap'], 'snap.shp')
        self.assertEqual(config['paths']['raw']['vectores']['lineas'], 'lineas.shp')
        self.assertEqual(experimento_del_pipeline(config), 'tres_sin_dedicadas')

    def test_sin_bloque_control_fuga_queda_desactivado(self):
        config = _config(True)
        del config['control_fuga']
        self.assertFalse(control_fuga_activo(config))


class TestGenerarCapas(unittest.TestCase):

    def test_escribe_capas_filtradas_en_32719_y_es_incremental(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Capas "crudas" en otro CRS (como las del CEN) con infraestructura dedicada.
            subest = gpd.GeoDataFrame(
                {'NOMBRE': ['S/E SOLAR', 'S/E TRONCAL'], 'TIPO': ['DEDICADO', 'NACIONAL'],
                 'REGION': ['02', '04'], 'geometry': [Point(300000, 7000000), Point(310000, 7000000)]},
                crs=CRS).to_crs(epsg=4326)
            lineas = gpd.GeoDataFrame(
                {'TIPO': ['DEDICADO', 'ZONAL'], 'REGION': ['ANTOFAGASTA', 'ATACAMA;COQUIMBO'],
                 'geometry': [LineString([(300000, 7000000), (301000, 7000000)]),
                              LineString([(300000, 7100000), (400000, 7100000)])]},
                crs=CRS).to_crs(epsg=4326)
            almacen = gpd.GeoDataFrame(
                {'NOMBRE_SE': ['S/E SOLAR', 'S/E TRONCAL'], 'REGION': ['02', '04'],
                 'geometry': [Point(300050, 7000050), Point(310050, 7000050)]},
                crs=CRS).to_crs(epsg=4326)
            vectores = {}
            for nombre, capa in (('subestaciones', subest), ('lineas', lineas), ('almacenamiento', almacen)):
                vectores[nombre] = os.path.join(tmp, f'{nombre}_crudo.gpkg')
                capa.to_file(vectores[nombre], driver='GPKG')
            config = _config(True, dir_procesadas=os.path.join(tmp, 'infra'), vectores=vectores)

            salidas = generar_capas_sin_dedicadas(config)

            for nombre, ruta in salidas.items():
                capa = gpd.read_file(ruta)
                self.assertEqual(capa.crs.to_epsg(), 32719, nombre)
                self.assertEqual(len(capa), 1, f"{nombre}: debía quedar solo lo no dedicado")
                self.assertIn('REGION', capa.columns, f"{nombre}: REGION se necesita para recortar")
            # El BESS de la central (conectado a una S/E dedicada) se excluye; el otro queda.
            self.assertEqual(list(gpd.read_file(salidas['almacenamiento'])['NOMBRE_SE']), ['S/E TRONCAL'])
            # Las capas son nacionales: lo que está en Coquimbo (04) se conserva.
            self.assertEqual(list(gpd.read_file(salidas['subestaciones'])['REGION']), ['04'])

            # Segunda llamada: al día, no se reescribe.
            mtimes = {n: os.path.getmtime(r) for n, r in salidas.items()}
            generar_capas_sin_dedicadas(config)
            self.assertEqual(mtimes, {n: os.path.getmtime(r) for n, r in salidas.items()})

    def test_editar_solo_el_dbf_invalida_las_capas(self):
        """El TIPO vive en el .dbf: editarlo (p. ej. en QGIS) debe regenerar las capas."""
        with tempfile.TemporaryDirectory() as tmp:
            capa = gpd.GeoDataFrame({'TIPO': ['NACIONAL'], 'NOMBRE': ['S/E A'], 'NOMBRE_SE': ['S/E A'],
                                     'geometry': [Point(300000, 7000000)]}, crs=CRS)
            vectores = {}
            for nombre in ('subestaciones', 'lineas', 'almacenamiento'):
                vectores[nombre] = os.path.join(tmp, f'{nombre}.shp')
                capa.to_file(vectores[nombre])
            config = _config(True, dir_procesadas=os.path.join(tmp, 'infra'), vectores=vectores)
            salidas = generar_capas_sin_dedicadas(config)
            antes = os.path.getmtime(salidas['subestaciones'])

            dbf = os.path.join(tmp, 'subestaciones.dbf')
            os.utime(dbf, (antes + 10, antes + 10))   # solo cambió el .dbf, no el .shp
            generar_capas_sin_dedicadas(config)
            self.assertGreater(os.path.getmtime(salidas['subestaciones']), antes)


class TestMuestreoConControlDeFuga(unittest.TestCase):
    """Con control de fuga, las VARIABLES excluyen lo dedicado, pero el criterio de
    pseudo-ausencias (dist_max) se mide contra la red completa: mismos puntos de entrenamiento."""

    def setUp(self):
        region = Polygon([(100000, 7000000), (500000, 7000000), (500000, 7500000), (100000, 7500000)])
        anillo = Polygon([(190000, 7050000), (410000, 7050000),
                          (410000, 7350000), (190000, 7350000)]).exterior
        # Una línea dedicada que cruza el centro: con ella, una franja central de ~40 km queda
        # a < 20 km de la red, cuando contra el anillo está a más de 100 km.
        dedicada = LineString([(220000, 7200000), (380000, 7200000)])
        self.lineas_completas = gpd.GeoDataFrame(
            {'REGION': ['Antofagasta', 'Antofagasta'], 'TIPO': ['NACIONAL', 'DEDICADO'],
             'geometry': [anillo, dedicada]}, crs=CRS)
        self.lineas_sin_dedicadas = excluir_dedicadas(self.lineas_completas)
        vacia = gpd.GeoDataFrame(columns=['geometry'], crs=CRS)
        self.vectores = {
            'regiones': gpd.GeoDataFrame({'REGION': ['Antofagasta'], 'geometry': [region]}, crs=CRS),
            'fotovoltaicas': gpd.GeoDataFrame(
                {'REGION': ['Antofagasta'] * 3, 'POTENCIAMW': [50.0, 100.0, 150.0],
                 'geometry': [Point(200000, 7100000), Point(300000, 7200000), Point(400000, 7300000)]},
                crs=CRS),
            'almacenamiento': vacia,
            'subestaciones': gpd.GeoDataFrame(
                {'geometry': [Point(250000, 7150000), Point(350000, 7250000)]}, crs=CRS),
            'areas_pobladas': vacia, 'masas_lacustres': vacia, 'snap': vacia,
        }
        self.criterios = {'ghi_min': 160, 'slope_max': 15, 'elev_max': 3500, 'dist_max': 20000}

    def _rasters(self, tmpdir):
        import rasterio
        from rasterio.transform import from_origin
        h, w = 500, 400
        meta = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0, 'width': w, 'height': h,
                'count': 1, 'crs': CRS, 'transform': from_origin(100000, 7500000, 1000, 1000)}
        valores = {'ghi_32719': 250.0, 'slope': 2.0, 'aspect': 45.0, 'dem_32719': 1000.0}
        rutas = {}
        for nombre, valor in valores.items():
            rutas[nombre] = os.path.join(tmpdir, f'{nombre}.tif')
            with rasterio.open(rutas[nombre], 'w', **meta) as dst:
                dst.write(np.full((h, w), valor, dtype=np.float32), 1)
        return rutas

    def _muestrear(self, rutas, lineas, lineas_criterio=None):
        # ratio alto: más candidatos sorteados, para que algunos caigan en la franja central.
        vectores = {**self.vectores, 'lineas': lineas}
        return generar_dataset_muestras(vectores, rutas, self.criterios, ratio=20, random_state=42,
                                        lineas_criterio=lineas_criterio)

    @staticmethod
    def _coordenadas(gdf):
        return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])

    def test_mismos_negativos_y_variables_sin_dedicadas(self):
        with tempfile.TemporaryDirectory() as tmp:
            rutas = self._rasters(tmp)
            pos_orig, neg_orig = self._muestrear(rutas, self.lineas_completas)
            pos_ctrl, neg_ctrl = self._muestrear(rutas, self.lineas_sin_dedicadas,
                                                 lineas_criterio=self.lineas_completas)

        # Mismo pool de negativos, en el mismo orden: el sorteo posterior es idéntico.
        self.assertEqual(len(neg_ctrl), len(neg_orig))
        np.testing.assert_array_equal(self._coordenadas(neg_ctrl), self._coordenadas(neg_orig))
        np.testing.assert_array_equal(self._coordenadas(pos_ctrl), self._coordenadas(pos_orig))
        # Pero la variable que ve el modelo ya no usa la línea dedicada: nunca más cerca.
        self.assertTrue((neg_ctrl['dist_transmision'].values
                         >= neg_orig['dist_transmision'].values - 1e-6).all())
        self.assertGreater(pos_ctrl['dist_transmision'].iloc[1], pos_orig['dist_transmision'].iloc[1])
        # Las columnas son las de siempre: nada auxiliar que el shapefile trunque (invariante 2).
        self.assertEqual(sorted(c for c in neg_ctrl.columns if c.startswith('dist_')),
                         sorted(f for f in FEATURES if f.startswith('dist_')))

    def test_sin_criterio_de_red_completa_los_negativos_cambiarian(self):
        """El motivo del parámetro: medir el criterio contra la red filtrada cambia el pool."""
        with tempfile.TemporaryDirectory() as tmp:
            rutas = self._rasters(tmp)
            _, neg_orig = self._muestrear(rutas, self.lineas_completas)
            _, neg_sin_criterio = self._muestrear(rutas, self.lineas_sin_dedicadas)
        # Sin la línea dedicada, la franja central queda a > 20 km y sus candidatos se pierden.
        self.assertLess(len(neg_sin_criterio), len(neg_orig))


if __name__ == '__main__':
    unittest.main()
