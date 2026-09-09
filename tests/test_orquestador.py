"""Fija el comportamiento del orquestador de fases.

El pipeline no tenía ninguna prueba sobre su orquestación: los demás tests cubren módulos
de `src/`, pero nadie verificaba que las 18 etapas se invocaran, en el orden correcto y con
los argumentos correctos. Este test es la red de seguridad del refactor a fases.

La secuencia esperada es la del pipeline previo a la reorganización, con el ÚNICO
reordenamiento acordado: la etapa 7 (PostGIS) pasó del medio al bloque final de
persistencia. Si un cambio futuro altera el orden o los argumentos, este test lo detecta
sin necesidad de correr el pipeline completo (que tarda alrededor de una hora).

No toca disco ni datos: sustituye `_correr_etapa` por un grabador.
"""

import os
import sys
import shutil
import tempfile
import unittest
from unittest import mock

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, directorio_raiz)
sys.path.insert(0, os.path.join(directorio_raiz, 'scripts'))

import run_pipeline


# Orden canónico: (numero_de_etapa, script). La etapa 1-2 fusiona las antiguas 1 y 2, que
# siempre corrieron juntas y sin chequeo propio en el orquestador.
SECUENCIA_COMPLETA = [
    ('1-2', 'run_preprocesamiento.py'),
    ('3',   'run_entrenamiento.py'),
    ('4',   'run_spatial_validation.py'),
    ('5',   'generate_suitability_map.py'),
    ('6',   'profiles.py'),
    ('8',   'run_solar_yield.py'),
    ('9',   'run_cruce.py'),
    ('10',  'run_comparacion_montaje.py'),
    ('11',  'run_consenso.py'),
    ('12',  'run_shap.py'),
    ('13',  'run_shap_spatial.py'),
    ('14',  'run_metricas_topk.py'),
    ('7',   'validate_and_load_postgis.py'),
    ('16',  'generar_figuras_informe.py'),
    ('17',  'generate_web_assets.py'),
]


class GrabadorDeEtapas:
    """Sustituye a `_correr_etapa`: registra la invocación en vez de lanzar el subproceso."""

    def __init__(self, fallan=()):
        self.llamadas = []
        self.fallan = set(fallan)  # scripts que devuelven False (p. ej. motor Rust ausente)

    def __call__(self, nombre_etapa, script, args_extra=None, motor_best_effort=False):
        self.llamadas.append((nombre_etapa, script, list(args_extra or []), motor_best_effort))
        return script not in self.fallan

    @property
    def scripts(self):
        return [script for _, script, _, _ in self.llamadas]

    def args_de(self, script):
        return next(args for _, s, args, _ in self.llamadas if s == script)


class TestOrquestadorDeFases(unittest.TestCase):

    def _correr(self, argv, fallan=()):
        """Corre el orquestador con las etapas simuladas y devuelve el grabador."""
        grabador = GrabadorDeEtapas(fallan=fallan)
        # `_esta_actualizado` devuelve False para que ninguna etapa se omita por frescura:
        # aquí se prueba la ESTRUCTURA del pipeline, no su idempotencia.
        with mock.patch.object(run_pipeline, '_correr_etapa', grabador), \
             mock.patch.object(run_pipeline, '_esta_actualizado', return_value=False):
            codigo = run_pipeline.main(argv)
        self.assertEqual(codigo, 0)
        return grabador

    def test_las_18_etapas_estan_declaradas(self):
        """Ninguna etapa se perdió al reagrupar: la tabla las contiene todas."""
        numeros = [e.numero for fase in run_pipeline.FASES for e in fase.etapas]
        self.assertEqual(
            numeros,
            ['1-2', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14',
             '15', '7', '16', '17', '17b', '18'])
        self.assertEqual([f.numero for f in run_pipeline.FASES], [1, 2, 3, 4, 5, 6])

    def test_secuencia_por_defecto(self):
        """Sin flags: todas las etapas salvo las dos de transferencia (15 y 17b)."""
        grabador = self._correr(['--config', 'config.yaml'])
        self.assertEqual(grabador.scripts, [script for _, script in SECUENCIA_COMPLETA])

    def test_postgis_corre_despues_de_las_figuras_y_del_analisis(self):
        """El único reordenamiento acordado: la etapa 7 pasó al bloque de persistencia."""
        grabador = self._correr(['--config', 'config.yaml'])
        orden = grabador.scripts
        self.assertGreater(orden.index('validate_and_load_postgis.py'),
                           orden.index('run_metricas_topk.py'))
        # Sigue dependiendo del mapa RF, que se genera mucho antes.
        self.assertLess(orden.index('generate_suitability_map.py'),
                        orden.index('validate_and_load_postgis.py'))

    def test_sin_mapas_corta_tras_el_modelamiento(self):
        """--sin-mapas equivale a 'hasta la fase 2', como el return de la versión anterior."""
        grabador = self._correr(['--config', 'config.yaml', '--sin-mapas'])
        self.assertEqual(grabador.scripts,
                         ['run_preprocesamiento.py', 'run_entrenamiento.py',
                          'run_spatial_validation.py'])

    def test_transferencia_es_opt_in(self):
        """La fase 5 y los assets de la zona solo aparecen con --con-transferencia."""
        sin = self._correr(['--config', 'config.yaml'])
        self.assertNotIn('run_zona.py', sin.scripts)
        self.assertEqual(sin.scripts.count('generate_web_assets.py'), 1)

    def test_solo_fase_ejecuta_una_fase(self):
        grabador = self._correr(['--config', 'config.yaml', '--solo-fase', '3'])
        self.assertEqual(grabador.scripts,
                         ['generate_suitability_map.py', 'profiles.py', 'run_solar_yield.py'])

    def test_desde_fase_reanuda(self):
        grabador = self._correr(['--config', 'config.yaml', '--desde-fase', '4'])
        self.assertNotIn('run_entrenamiento.py', grabador.scripts)
        self.assertEqual(grabador.scripts[0], 'run_cruce.py')

    def test_etapas_del_motor_son_best_effort(self):
        """Un motor Rust ausente no debe abortar el pipeline (etapas 8 y 10)."""
        best_effort = {script: be for _, script, _, be in
                       self._correr(['--config', 'config.yaml']).llamadas}
        self.assertTrue(best_effort['run_solar_yield.py'])
        self.assertTrue(best_effort['run_comparacion_montaje.py'])
        self.assertFalse(best_effort['generate_suitability_map.py'])

    def test_pipeline_sobrevive_sin_motor_rust(self):
        """Si las etapas del motor fallan, las demás igual se ejecutan hasta el final."""
        grabador = self._correr(
            ['--config', 'config.yaml'],
            fallan=('run_solar_yield.py', 'run_comparacion_montaje.py'))
        self.assertIn('generate_web_assets.py', grabador.scripts)
        self.assertIn('generar_figuras_informe.py', grabador.scripts)

    def test_scripts_sin_argparse_se_invocan_sin_config(self):
        """Tres scripts abren 'config.yaml' literal y no aceptan --config: no hay que pasárselo."""
        grabador = self._correr(['--config', 'config.yaml'])
        for script in ('run_spatial_validation.py', 'generate_suitability_map.py',
                       'generar_figuras_informe.py'):
            self.assertEqual(grabador.args_de(script), [], f"{script} no acepta --config")

    def test_el_resto_recibe_la_ruta_del_config(self):
        """El --config elegido se propaga tal cual a las etapas que sí lo aceptan."""
        with tempfile.TemporaryDirectory() as tmp:
            otro = os.path.join(tmp, 'otro.yaml')
            shutil.copy(os.path.join(directorio_raiz, 'config.yaml'), otro)
            grabador = self._correr(['--config', otro])
        for script in ('run_preprocesamiento.py', 'run_entrenamiento.py', 'run_cruce.py',
                       'run_solar_yield.py', 'run_consenso.py', 'run_shap.py'):
            self.assertEqual(grabador.args_de(script), ['--config', otro])

    def test_el_orquestador_no_carga_datos_geoespaciales(self):
        """La razón de ser del refactor: el proceso padre no retiene capas en memoria.

        Si alguien vuelve a importar geopandas o las funciones de carga en el orquestador,
        el problema de memoria regresa silenciosamente.
        """
        fuente = open(os.path.join(directorio_raiz, 'scripts', 'run_pipeline.py'),
                      encoding='utf-8').read()
        for prohibido in ('cargar_capas_vectoriales', 'generar_dataset_muestras',
                          'entrenar_modelo_rf', 'import geopandas', 'procesar_dem'):
            self.assertNotIn(prohibido, fuente,
                             f"run_pipeline.py no debe usar '{prohibido}': cargaría datos "
                             "en el proceso padre y los mantendría residentes.")


if __name__ == '__main__':
    unittest.main()
