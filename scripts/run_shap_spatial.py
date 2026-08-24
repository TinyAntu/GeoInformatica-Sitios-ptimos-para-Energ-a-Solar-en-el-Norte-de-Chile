"""CLI: explicabilidad espacial SHAP (Brecha 8). Mapas de contribución por variable +
waterfall de los Top-N sitios. No re-entrena: usa model_rf.pkl.

Uso:
    python scripts/run_shap_spatial.py --config config.yaml
"""

import os
import sys
import argparse
import warnings
import yaml

warnings.filterwarnings("ignore")

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.explainability_spatial import generar_mapas_shap
from src.utils import _esta_actualizado


def _ruta_abs(ruta: str) -> str:
    return ruta if os.path.isabs(ruta) else os.path.join(directorio_raiz, ruta)


def main():
    parser = argparse.ArgumentParser(description='Explicabilidad espacial SHAP (Brecha 8)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza recalcular aunque el resultado ya esté actualizado')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_path = _ruta_abs(config['paths']['results'].get('model_rf', 'data/results/model_rf.pkl'))
    if not os.path.exists(model_path):
        print(f"  [ERROR] No existe el modelo ({model_path}). Corre el pipeline primero.")
        return 1

    cfg = config.get('shap_espacial', {})
    resolucion_m = cfg.get('resolucion_m', 500)
    top_n = cfg.get('top_sitios', 5)
    out_dir = _ruta_abs('data/results')
    figures_dir = _ruta_abs('figures')

    # Esta etapa es la más cara del pipeline (millones de píxeles vía TreeExplainer): se
    # salta si el resultado ya es más nuevo que el modelo y la config, igual que cuando
    # corre dentro de run_pipeline.py (que usa el mismo _esta_actualizado).
    out_json = os.path.join(out_dir, 'shap_espacial.json')
    if not args.regenerar and _esta_actualizado([ruta_config, model_path], [out_json]):
        print(f"  [OK] Ya existe y está actualizado: {out_json} (usa --regenerar para forzar)")
        return 0

    print(f"Generando mapas SHAP espaciales (resolución {resolucion_m} m, Top-{top_n} sitios)...")
    res = generar_mapas_shap(config, model_path, resolucion_m, out_dir, figures_dir, top_n=top_n)

    print(f"\n  Píxeles explicados: {res['n_pixeles']:,}")
    print("  Contribución media SHAP por variable:")
    for f_, v in sorted(res['contribucion_media_shap'].items(), key=lambda x: -abs(x[1])):
        print(f"    {f_:<20} {v:+.5f}")
    print("  Píxeles por variable dominante:")
    for f_, n in sorted(res['pixeles_por_variable_dominante'].items(), key=lambda x: -x[1]):
        print(f"    {f_:<20} {n:,}")
    print(f"  Rasters: {len(res['rasters'])} | Waterfalls: {len(res['waterfalls'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
