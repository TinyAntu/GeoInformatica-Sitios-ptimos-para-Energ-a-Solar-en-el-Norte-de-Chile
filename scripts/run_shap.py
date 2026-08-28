"""CLI: explicabilidad SHAP del modelo de aptitud (T6, deuda de PEP1).

Genera el summary plot SHAP (global) y, si existe el mapa de rendimiento, el cruce
aptitud–rendimiento. No re-entrena: usa data/results/model_rf.pkl.

Uso:
    python scripts/run_shap.py --config config.yaml
"""

import os
import sys
import argparse
import warnings
import yaml

warnings.filterwarnings("ignore")  # silencia avisos de shap/sklearn sin ocultar errores reales

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.explainability import explicar


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Explicabilidad SHAP del modelo de aptitud (T6)')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(_ruta_abs(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    results = config['paths']['results']
    model_path = _ruta_abs(results.get('model_rf', 'data/results/model_rf.pkl'))
    dataset_path = _ruta_abs(results.get('dataset_ml', 'data/results/dataset_entrenamiento_rf.shp'))
    figures_dir = _ruta_abs('figures')
    out_json = _ruta_abs('data/results/shap_importancias.json')

    # Rendimiento del montaje fijo (T1) para el cruce, si está disponible.
    cfg_solar = config.get('solarpv', {})
    rendimiento_path = _ruta_abs(cfg_solar.get('out_prefix', 'data/results/rendimiento')
                                 + '_fijo_specific_yield.tif')

    if not os.path.exists(model_path):
        print(f"  [ERROR] No existe el modelo ({model_path}). Corre el pipeline primero.")
        return 1

    # Mismo umbral que postgis_validation.prob_min: el cruce aptitud-rendimiento se acota
    # también a los sitios que el modelo efectivamente aprobaría (ver src/explainability.py).
    prob_min = config.get('postgis_validation', {}).get('prob_min', 0.70)

    print("Calculando valores SHAP (TreeExplainer, sin re-entrenar)...")
    res = explicar(model_path, dataset_path, figures_dir, out_json,
                   rendimiento_path=rendimiento_path if os.path.exists(rendimiento_path) else None,
                   prob_min=prob_min)

    print(f"\n  Muestras explicadas: {res['n_muestras']}")
    print("  Importancia SHAP (global):")
    for it in res['importancias_shap']:
        signo = '+' if it['shap_medio'] >= 0 else '-'
        print(f"    {it['feature']:<20} {it['importancia_pct']:>5.1f}%  (dirección {signo})")
    if 'cruce_rendimiento' in res:
        c = res['cruce_rendimiento']
        if 'spearman_aptitud_vs_rendimiento' in c:
            print(f"\n  Cruce, muestra completa (n={c['n_puntos']}): "
                  f"Spearman aptitud–rendimiento = "
                  f"{c['spearman_aptitud_vs_rendimiento']} (p={c['pvalue']})")
            c_apt = c.get('cruce_solo_sitios_aptos', {})
            if 'spearman_aptitud_vs_rendimiento' in c_apt:
                print(f"  Cruce, solo sitios con prob >= {c['prob_min_apto']} "
                      f"(n={c_apt['n_puntos']}): Spearman aptitud–rendimiento = "
                      f"{c_apt['spearman_aptitud_vs_rendimiento']} (p={c_apt['pvalue']})")
            else:
                print(f"  [AVISO] {c_apt.get('nota', 'sin datos suficientes para el cruce '
                                                        'restringido a sitios aptos')}")
    print(f"\n  Figuras: {os.path.basename(res['figuras']['bar'])}, "
          f"{os.path.basename(res['figuras']['beeswarm'])}")
    print(f"  JSON: {os.path.basename(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
