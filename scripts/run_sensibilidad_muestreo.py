"""Fase 2 — Análisis de Sensibilidad al Diseño de Pseudo-Ausencias.

Compara cuantitativamente los 3 diseños de muestreo de negativos:
  1) ahp_filtrado: línea base con filtros técnicos AHP.
  2) fondo_aleatorio: fondo regional sin filtros técnicos.
  3) fondo_objetivo: condicionado a infraestructura sin filtros geofísicos.

Calcula métricas SBCV, importancias SHAP globales, dominancia de macro-familias
(acceso a red vs recurso) y correlación de rangos (Kendall y Spearman).

Uso:
    python scripts/run_sensibilidad_muestreo.py --config config.yaml
"""

import os
import sys
import argparse
import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _resolver_rutas, _ruta_abs
from src.sensibilidad_muestreo import analizar_sensibilidad_muestreo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo config.yaml')
    parser.add_argument('--out-json', default=None,
                        help='Ruta de salida del JSON (por defecto data/results/sensibilidad_muestreo.json)')
    parser.add_argument('--figures-dir', default=None,
                        help='Directorio de salida de figuras (por defecto figures/)')
    args = parser.parse_args()

    ruta_config = _ruta_abs(args.config)
    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['paths'] = _resolver_rutas(config['paths'])

    out_json = _ruta_abs(args.out_json) if args.out_json else os.path.join(
        directorio_raiz, 'data', 'results', 'sensibilidad_muestreo.json'
    )
    figures_dir = _ruta_abs(args.figures_dir) if args.figures_dir else os.path.join(
        directorio_raiz, 'figures'
    )

    print("=" * 75)
    print("ANÁLISIS DE SENSIBILIDAD AL DISEÑO DE PSEUDO-AUSENCIAS (PEP2 -> PAPER)")
    print("=" * 75)

    resultados = analizar_sensibilidad_muestreo(
        config=config,
        directorio_raiz=directorio_raiz,
        out_json=out_json,
        figures_dir=figures_dir,
    )

    comp = resultados['resumen_comparativo']
    print("\n" + "=" * 75)
    print("RESUMEN DE ESTABILIDAD Y JERARQUÍA SHAP")
    print("=" * 75)
    print(f"Variable Dominante (#1 SHAP):")
    for est, top1 in comp['variable_dominante'].items():
        if est != 'es_invariante':
            print(f"  - {est:18s}: {top1}")
    print(f"  -> ¿Invariante en todos los diseños?: {'SÍ' if comp['variable_dominante']['es_invariante'] else 'NO'}")

    print(f"\nCorrelación de Rangos de Importancia:")
    print(f"  - Kendall's tau (AHP vs Aleatorio): {comp['correlacion_kendall_tau']['ahp_vs_aleatorio']:.4f}")
    print(f"  - Kendall's tau (AHP vs Objetivo):  {comp['correlacion_kendall_tau']['ahp_vs_objetivo']:.4f}")
    print(f"  - Spearman rho  (AHP vs Aleatorio): {comp['correlacion_spearman_rho']['ahp_vs_aleatorio']:.4f}")
    print(f"  - Spearman rho  (AHP vs Objetivo):  {comp['correlacion_spearman_rho']['ahp_vs_objetivo']:.4f}")

    print(f"\nPeso de Macro-Familias SHAP (%):")
    for est, fam in comp['comparacion_macro_familias'].items():
        print(f"  - {est:18s}: Red = {fam['acceso_red_pct']:.1f}% | Recurso = {fam['recurso_topografia_pct']:.1f}% | Ratio = {fam.get('ratio_red_vs_recurso', 0):.2f}x")

    print("\n[OK] Análisis de sensibilidad completado exitosamente.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
