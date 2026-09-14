"""Fase 1, tarea T2 — Análisis de Sensibilidad al Diseño de Pseudo-Ausencias.

Compara cuantitativamente los 4 diseños de muestreo de negativos:
  1) ahp_filtrado: línea base con filtros técnicos AHP.
  2) fondo_aleatorio: fondo regional sin filtros técnicos.
  3) sin_filtros_geofisicos: ablación de la línea base, solo distancia a red.
  4) grupo_objetivo: target-group background en torno a la infraestructura existente.

Calcula métricas SBCV, importancias SHAP globales, dominancia de macro-familias
(acceso a red vs recurso) y correlación de rangos (Kendall y Spearman, con p-valor).

Necesita el modelo ya entrenado: los hiperparámetros salen de model_rf_metrics.json.

Uso:
    python scripts/run_sensibilidad_muestreo.py --config config.yaml
    python scripts/run_sensibilidad_muestreo.py --regenerar   # fuerza recalcular
"""

import os
import sys
import argparse
import yaml

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.utils import _resolver_rutas, _ruta_abs, _esta_actualizado
from src.sensibilidad_muestreo import analizar_sensibilidad_muestreo


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo config.yaml')
    parser.add_argument('--out-json', default=None,
                        help='Ruta de salida del JSON (por defecto data/results/sensibilidad_muestreo.json)')
    parser.add_argument('--figures-dir', default=None,
                        help='Directorio de salida de figuras (por defecto figures/)')
    parser.add_argument('--regenerar', action='store_true',
                        help='Fuerza recalcular aunque el resultado ya esté actualizado')
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

    # Son cuatro entrenamientos completos más cuatro pasadas de SHAP: se omite si el
    # resultado ya es más nuevo que el modelo y la configuración que lo determinan
    # (mismo criterio que run_solar_yield.py y run_metricas_topk.py).
    resultados_dir = config['paths']['results']
    entradas = [ruta_config]
    if resultados_dir.get('model_rf'):
        entradas.append(resultados_dir['model_rf'])
    if not args.regenerar and _esta_actualizado(entradas, [out_json]):
        print(f"  [OK] Ya existe y está actualizado: {out_json} (usa --regenerar para forzar)")
        return 0

    resultados = analizar_sensibilidad_muestreo(
        config=config,
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

    print(f"\nHiperparámetros del RF usados: {resultados['hiperparametros_rf']}")
    print(f"  (origen: {resultados['origen_hiperparametros']})")

    n_vars = comp.get('n_variables_correlacionadas', '?')
    print(f"\nCorrelación de Rangos de Importancia (sobre {n_vars} variables):")
    # Se recorren las claves que el módulo generó (un par por combinación de diseños), en vez
    # de una lista fija: así agregar un diseño no obliga a tocar también este reporte.
    kendall = comp['correlacion_kendall_tau']
    spearman = comp['correlacion_spearman_rho']
    ancho = max((len(k) for k in kendall), default=0)
    for clave, t in kendall.items():
        r = spearman.get(clave, {})
        print(f"  - {clave:<{ancho}} : tau = {t['tau']:+.4f} (p = {t['p_valor']:.4f})"
              + (f" | rho = {r['rho']:+.4f} (p = {r['p_valor']:.4f})" if r else ""))

    print(f"\nPeso de Macro-Familias SHAP (%):")
    for est, fam in comp['comparacion_macro_familias'].items():
        # El ratio es None cuando la familia de recurso concentra 0% de la importancia: un
        # `.get(clave, 0)` NO protege, porque la clave existe con valor None y el formato falla.
        ratio = fam.get('ratio_red_vs_recurso')
        ratio_txt = f"{ratio:.2f}x" if ratio is not None else "n/d (recurso = 0%)"
        print(f"  - {est:18s}: Red = {fam['acceso_red_pct']:.1f}% | "
              f"Recurso = {fam['recurso_topografia_pct']:.1f}% | Ratio = {ratio_txt}")

    print("\n[OK] Análisis de sensibilidad completado exitosamente.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
