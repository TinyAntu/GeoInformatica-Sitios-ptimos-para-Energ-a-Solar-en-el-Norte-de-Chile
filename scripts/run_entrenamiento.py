"""Fase 2 — Etapa 3: muestreo espacial y entrenamiento del Random Forest.

Genera el dataset de muestras (positivas = plantas existentes; negativas = candidatos
elegibles según `criterios`) y entrena el modelo con optimización Optuna sobre validación
espacial por bloques.

Este script existe como proceso aparte por una razón de memoria: carga las 8 capas
vectoriales (`cargar_capas_vectoriales`) y sostiene `positivas` y `pool_negativos` en
memoria durante el entrenamiento. Al vivir en su propio proceso, todo eso se devuelve al
sistema cuando termina, en vez de quedar residente en el orquestador durante las etapas
siguientes —que es lo que ocurría cuando esta etapa corría dentro de run_pipeline.py.

Uso:
    python scripts/run_entrenamiento.py --config config.yaml
"""

import os
import sys
import argparse

import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.preprocessing import cargar_capas_vectoriales
from src.sampling import generar_dataset_muestras
from src.modeling import entrenar_modelo_rf
from src.utils import _esta_actualizado, _resolver_rutas, _ruta_abs, _shapefile_paths


def entradas_esperadas(config, ruta_config: str) -> list:
    """Rutas cuya fecha decide si hay que reentrenar. Compartida con el orquestador."""
    processed = config['paths']['processed']
    entradas = [
        ruta_config,
        config['paths']['raw']['rasters']['ghi'],
        processed['ghi_32719'],
        processed['slope'],
        processed['aspect'],
        processed['dem_32719'],
    ]
    entradas.extend(config['paths']['raw']['vectores'].values())
    return entradas


def salidas_esperadas(config) -> list:
    """El shapefile del dataset (5 archivos ESRI) más el modelo serializado."""
    results = config['paths']['results']
    salidas = _shapefile_paths(results['dataset_ml'])
    if results.get('model_rf'):
        salidas.append(results['model_rf'])
    return salidas


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza reentrenar aunque el dataset y el modelo estén actualizados')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])

    results = config['paths']['results']
    dataset_out = results['dataset_ml']
    model_out = results.get('model_rf')

    if not args.regenerar and _esta_actualizado(entradas_esperadas(config, ruta_config),
                                                salidas_esperadas(config)):
        print("El dataset y modelo ya están actualizados. No se requiere reprocesar entrenamiento.")
        return 0

    os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)

    vectores = cargar_capas_vectoriales(config['paths']['raw']['vectores'])
    print(f"Se cargaron {len(vectores)} capas vectoriales.")

    processed = config['paths']['processed']
    rutas_rasters = {
        'ghi_32719': processed['ghi_32719'],
        'slope':     processed['slope'],
        'aspect':    processed['aspect'],
        'dem_32719': processed['dem_32719'],
    }

    ml = config['ml_params']
    positivas, pool_negativos = generar_dataset_muestras(
        vectores, rutas_rasters, config['criterios'],
        ratio=ml['ratio_negativos'],
        random_state=ml['random_state'],
    )
    entrenar_modelo_rf(
        positivas=positivas,
        pool_negativos=pool_negativos,
        ratio=ml['ratio_negativos'],
        out_shp=dataset_out,
        optuna_config=config.get('optuna_params', {}),
        random_state=ml['random_state'],
        out_model_path=model_out,
        n_estimators=ml['n_estimators'],
        ratio_alt=ml.get('ratio_alt'),
        tamano_bloque_km=config.get('validacion', {}).get('tamano_bloque_km', 15),
        # Filtrado a Antofagasta+Atacama antes de pasarlo: el shapefile completo de
        # regiones incluye la costa patagónica (miles de vértices) y reproyectar/hacer
        # sjoin contra Chile completo es innecesariamente caro en memoria/tiempo.
        regiones_gdf=vectores['regiones'][
            vectores['regiones']['REGION'].isin(['Antofagasta', 'Atacama'])
        ],
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
