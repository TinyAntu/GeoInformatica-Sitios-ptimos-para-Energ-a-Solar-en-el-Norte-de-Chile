"""CLI de Análisis de Desacople Aptitud–Rendimiento (Modelo completo y sin dedicadas).

Calcula y genera:
  1. Preparación de capas de infraestructura sin elementos dedicados (cuando sea necesario).
  2. Entrenamiento de Random Forest y mapa ráster continuo si no existe.
  3. Alineación espacial y cruce celda a celda contra el rendimiento físico (solarpv-rs).
  4. Rendimiento medio y mediano regional vs. zonas aptas (P >= 0.70).
  5. Correlación de Spearman por celda y agregada en bloques de 30 km del SBCV.
  6. Control espacial 1: Prueba t modificada de Dutilleul (rangos/Spearman y medias crudas/Pearson).
  7. Control espacial 2: Bootstrap espacial con superbloques de 2 x 2 bloques (60 km).
  8. Exportación de resultados a tabla CSV consolidada y JSON de metadatos.

Uso:
    python scripts/run_desacople_aptitud_rendimiento.py --config config.yaml
    python scripts/run_desacople_aptitud_rendimiento.py --modo ambos --n-bootstrap 10000
"""

import os
import sys
import yaml
import json
import argparse

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(directorio_raiz)

from src.desacople_aptitud_rendimiento import (
    preparar_capas_sin_dedicadas,
    entrenar_modelo_sin_dedicadas,
    generar_mapa_aptitud_sin_dedicadas,
    alinear_y_cruzar,
    calcular_estadisticas_desacople,
    exportar_tabla_csv,
)


def _resolver_ruta_existente(candidatas: list[str]) -> str | None:
    """Devuelve la primera ruta existente de la lista de candidatas."""
    for ruta in candidatas:
        if ruta and os.path.exists(ruta):
            return ruta
    return None


def _asegurar_mapa_sin_dedicadas(config: dict, dir_auditoria: str, dir_results: str,
                                 random_state: int) -> str:
    """Localiza o genera el ráster de aptitud del modelo sin obras dedicadas."""
    existente = _resolver_ruta_existente([
        os.path.join(dir_results, 'mapa_probabilidad_aptitud_sin_dedicadas.tif'),
        os.path.join(dir_auditoria, 'mapa_probabilidad_aptitud_sin_dedicadas.tif'),
    ])
    if existente:
        print(f"  [OK] Ráster sin dedicadas encontrado: {existente}")
        return existente

    os.makedirs(dir_auditoria, exist_ok=True)
    out_model_path = os.path.join(dir_auditoria, 'model_rf_sin_dedicadas.pkl')
    out_tif_path = os.path.join(dir_auditoria, 'mapa_probabilidad_aptitud_sin_dedicadas.tif')

    rutas_gpkg = {
        'subestaciones': os.path.join(dir_auditoria, 'subestaciones_sin_dedicadas.gpkg'),
        'lineas': os.path.join(dir_auditoria, 'lineas_sin_dedicadas.gpkg'),
        'almacenamiento': os.path.join(dir_auditoria, 'almacenamiento_sin_dedicadas.gpkg'),
    }
    capas_zonales = None
    if not all(os.path.exists(p) for p in rutas_gpkg.values()):
        print("\n[1/3] Preparación de infraestructura sin obras dedicadas...")
        rutas_gpkg, capas_zonales = preparar_capas_sin_dedicadas(config, directorio_raiz, dir_auditoria)

    if not os.path.exists(out_model_path):
        if capas_zonales is None:
            rutas_gpkg, capas_zonales = preparar_capas_sin_dedicadas(config, directorio_raiz, dir_auditoria)
        print("\n[2/3] Entrenando Random Forest sin dedicadas...")
        modelo, _ = entrenar_modelo_sin_dedicadas(
            config, directorio_raiz, capas_zonales, out_model_path, random_state=random_state
        )
    else:
        import joblib
        modelo = joblib.load(out_model_path)

    print("\n[3/3] Generando ráster de aptitud espacial sin dedicadas...")
    return generar_mapa_aptitud_sin_dedicadas(config, directorio_raiz, modelo, rutas_gpkg, out_tif_path)


def _imprimir_resumen(nombre_modelo: str, resultados: dict, umbral: float,
                      out_csv: str, out_json: str) -> None:
    """Imprime un resumen estructurado en consola incluyendo los controles espaciales."""
    r = resultados['resumen_rendimiento']
    reg = resultados['region']
    apt = resultados['aptas']

    print("\n" + "=" * 92)
    print(f"RESULTADOS DEL DESACOPLE APTITUD-RENDIMIENTO ({nombre_modelo.upper()})")
    print("=" * 92)
    print(f"Celdas válidas evaluadas:         {r['celdas_validas']:,}")
    print(f"Celdas aptas (P >= {umbral:.2f}):           {r['celdas_aptas']:,} ({r['pct_aptas']}%)")
    print(
        f"Bloques 30 km / Superbloques 60:  "
        f"{reg['n_bloques_30km']} bloques ({reg['n_superbloques_60km']} superbloques) región | "
        f"{apt['n_bloques_30km']} bloques ({apt['n_superbloques_60km']} superbloques) aptas"
    )
    print(f"Mediana celdas/bloque (región):   {reg['mediana_celdas_por_bloque']:,}")
    print("-" * 92)
    print(f"Rendimiento regional (kWh/kWp):   media = {r['rendimiento_region']['media']} | mediana = {r['rendimiento_region']['mediana']}")
    print(f"Rendimiento en aptas (kWh/kWp):   media = {r['rendimiento_aptas']['media']} | mediana = {r['rendimiento_aptas']['mediana']}")
    print(f"Ganancia relativa en aptas:       media = {r['ganancia_media_pct']:+}% | mediana = {r['ganancia_mediana_pct']:+}%")
    print("-" * 92)

    for etiqueta_amb, amb in [('Región completa', reg), ('Subconjunto aptas', apt)]:
        sc = amb['spearman_celda']
        sb = amb['spearman_bloque']
        ds = sb['dutilleul_spearman']
        dp = sb['dutilleul_pearson']
        bs = sb['bootstrap_superbloques_60km']
        print(f"[{etiqueta_amb}]")
        print(f"  Spearman celda:                 rho = {sc['rho']:.4f} (IC 95%: [{sc['ic_95_inferior']:.4f}, {sc['ic_95_superior']:.4f}])")
        print(f"  Spearman bloque 30 km:          rho = {sb['rho']:.4f} (IC 95% 30km: [{sb['ic_95_inferior']:.4f}, {sb['ic_95_superior']:.4f}]) | p clásico = {sb['p_valor']}")
        print(f"  Dutilleul (rangos/Spearman):    rho = {ds['rho']:.4f} | gl: {ds['gl_clasicos']} -> {ds['gl_efectivos']:.2f} | p = {ds['p_valor']}")
        print(f"  Dutilleul (medias/Pearson):     r   = {dp['r']:.4f} | gl: {dp['gl_clasicos']} -> {dp['gl_efectivos']:.2f} | p = {dp['p_valor']}")
        print(f"  Bootstrap superbloques 2x2:     IC 95% 60km = [{bs['ic_95_inferior']:.4f}, {bs['ic_95_superior']:.4f}] (G={bs['n_superbloques']}, B={resultados['n_bootstrap']})")

    print("-" * 92)
    print(f"Tabla CSV generada en:            {out_csv}")
    print(f"JSON con metadatos en:            {out_json}")
    print("=" * 92)


def _ejecutar_modelo(nombre: str, aptitud_path: str, rendimiento_path: str, umbral: float,
                     bloque_km: float, n_bootstrap: int, random_state: int,
                     out_csv_path: str, out_json_path: str,
                     control_sin_dedicadas: bool) -> dict:
    """Ejecuta cruce, estadísticas (con Dutilleul y superbloques 2x2) y exportación."""
    print(f"\n>>> Evaluando modelo '{nombre}' sobre: {aptitud_path}")
    cruce = alinear_y_cruzar(aptitud_path, rendimiento_path, umbral=umbral)
    resultados = calcular_estadisticas_desacople(
        cruce,
        tamano_bloque_km=bloque_km,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    resultados['entradas'] = {
        'mapa_aptitud': aptitud_path,
        'rendimiento': rendimiento_path,
        'modelo': nombre,
        'control_fuga_excluir_dedicadas': control_sin_dedicadas,
    }

    exportar_tabla_csv(resultados, out_csv_path)
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    _imprimir_resumen(nombre, resultados, umbral, out_csv_path, out_json_path)
    return resultados


def main():
    parser = argparse.ArgumentParser(description='Análisis de Desacople Aptitud-Rendimiento')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo config.yaml')
    parser.add_argument('--bloque-km', type=float, default=30.0, help='Tamaño del bloque espacial en km (default: 30)')
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Número de réplicas bootstrap (default: 10000)')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla aleatoria (default: 42)')
    parser.add_argument(
        '--modo',
        choices=['sin_dedicadas', 'completo', 'ambos'],
        default='ambos',
        help='Mapa(s) de aptitud a evaluar (default: ambos)',
    )
    parser.add_argument('--aptitud-tif', default=None, help='Ruta explícita a un ráster de aptitud específico')
    parser.add_argument('--rendimiento-tif', default=None, help='Ruta explícita al ráster de rendimiento')
    args = parser.parse_args()

    print("=" * 92)
    print("ANÁLISIS DE DESACOPLE APTITUD-RENDIMIENTO CON CONTROL ESPACIAL")
    print("=" * 92)

    config_path = args.config if os.path.isabs(args.config) else os.path.join(directorio_raiz, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dir_auditoria = os.path.join(directorio_raiz, 'data/results/auditoria_infraestructura')
    dir_results = os.path.join(directorio_raiz, 'data/results')

    rendimiento_path = args.rendimiento_tif or os.path.join(dir_results, 'rendimiento_fijo_specific_yield.tif')
    if not os.path.exists(rendimiento_path):
        print(f"[ERROR] No existe el mapa de rendimiento en {rendimiento_path}.")
        return 1

    umbral = config.get('postgis_validation', {}).get('prob_min', 0.70)

    if args.aptitud_tif:
        if not os.path.exists(args.aptitud_tif):
            print(f"[ERROR] No existe el mapa de aptitud en {args.aptitud_tif}.")
            return 1
        _ejecutar_modelo(
            nombre='personalizado',
            aptitud_path=args.aptitud_tif,
            rendimiento_path=rendimiento_path,
            umbral=umbral,
            bloque_km=args.bloque_km,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            out_csv_path=os.path.join(dir_results, 'desacople_aptitud_rendimiento.csv'),
            out_json_path=os.path.join(dir_results, 'desacople_aptitud_rendimiento.json'),
            control_sin_dedicadas='sin_dedicadas' in args.aptitud_tif,
        )
        return 0

    if args.modo in ('completo', 'ambos'):
        apt_completo = os.path.join(dir_results, 'mapa_probabilidad_aptitud.tif')
        if not os.path.exists(apt_completo):
            print(f"[ERROR] No existe el mapa de aptitud completo en {apt_completo}.")
            return 1
        _ejecutar_modelo(
            nombre='modelo_completo',
            aptitud_path=apt_completo,
            rendimiento_path=rendimiento_path,
            umbral=umbral,
            bloque_km=args.bloque_km,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            out_csv_path=os.path.join(dir_results, 'desacople_aptitud_rendimiento_completo.csv'),
            out_json_path=os.path.join(dir_results, 'desacople_aptitud_rendimiento_completo.json'),
            control_sin_dedicadas=False,
        )

    if args.modo in ('sin_dedicadas', 'ambos'):
        apt_sd = _asegurar_mapa_sin_dedicadas(config, dir_auditoria, dir_results, args.random_state)
        _ejecutar_modelo(
            nombre='sin_dedicadas',
            aptitud_path=apt_sd,
            rendimiento_path=rendimiento_path,
            umbral=umbral,
            bloque_km=args.bloque_km,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
            out_csv_path=os.path.join(dir_results, 'desacople_aptitud_rendimiento_sin_dedicadas.csv'),
            out_json_path=os.path.join(dir_results, 'desacople_aptitud_rendimiento_sin_dedicadas.json'),
            control_sin_dedicadas=True,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
