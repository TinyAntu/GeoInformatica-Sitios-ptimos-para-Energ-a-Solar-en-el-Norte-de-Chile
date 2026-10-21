"""CLI de Análisis de Desacople Aptitud–Rendimiento (Modelo sin infraestructura dedicada).

Calcula y genera:
  1. Preparación de capas de infraestructura sin elementos dedicados (TIPO != 'DEDICADO').
  2. Entrenamiento de Random Forest y mapa ráster continuo (mapa_probabilidad_aptitud_sin_dedicadas.tif).
  3. Alineación espacial y cruce celda a celda contra el rendimiento físico (solarpv-rs v0.4.0).
  4. Rendimiento medio y mediano regional vs. zonas aptas (P >= 0.70).
  5. Correlación de Spearman por celda y agregada en bloques de 30 km del SBCV.
  6. Intervalo de confianza al 95% por Bootstrap espacial de bloques.
  7. Exportación de resultados a tabla CSV consolidada y JSON de metadatos.

Uso:
    python scripts/run_desacople_aptitud_rendimiento.py --config config.yaml
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


def main():
    parser = argparse.ArgumentParser(description='Análisis de Desacople Aptitud-Rendimiento')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo config.yaml')
    parser.add_argument('--bloque-km', type=float, default=30.0, help='Tamaño del bloque espacial en km (default: 30)')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Número de réplicas bootstrap (default: 1000)')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla aleatoria (default: 42)')
    args = parser.parse_args()

    print("=" * 80)
    print("ANÁLISIS DE DESACOPLE APTITUD-RENDIMIENTO (MODELO SIN DEDICADAS)")
    print("=" * 80)

    config_path = args.config if os.path.isabs(args.config) else os.path.join(directorio_raiz, args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dir_auditoria = os.path.join(directorio_raiz, 'data/results/auditoria_infraestructura')
    dir_results = os.path.join(directorio_raiz, 'data/results')
    os.makedirs(dir_auditoria, exist_ok=True)

    out_model_path = os.path.join(dir_auditoria, 'model_rf_sin_dedicadas.pkl')
    out_tif_path = os.path.join(dir_auditoria, 'mapa_probabilidad_aptitud_sin_dedicadas.tif')
    rendimiento_path = os.path.join(directorio_raiz, 'data/results/rendimiento_fijo_specific_yield.tif')

    out_csv_auditoria = os.path.join(dir_auditoria, 'desacople_aptitud_rendimiento.csv')
    out_csv_results = os.path.join(dir_results, 'desacople_aptitud_rendimiento.csv')
    out_json_path = os.path.join(dir_results, 'desacople_aptitud_rendimiento.json')

    if not os.path.exists(rendimiento_path):
        print(f"[ERROR] No existe el mapa de rendimiento en {rendimiento_path}.")
        return 1

    # 1. Capas sin dedicadas
    rutas_gpkg = {
        'subestaciones': os.path.join(dir_auditoria, 'subestaciones_sin_dedicadas.gpkg'),
        'lineas': os.path.join(dir_auditoria, 'lineas_sin_dedicadas.gpkg'),
        'almacenamiento': os.path.join(dir_auditoria, 'almacenamiento_sin_dedicadas.gpkg'),
    }
    capas_zonales = None
    if not all(os.path.exists(p) for p in rutas_gpkg.values()):
        print("\n[1/6] Preparación de infraestructura sin obras dedicadas...")
        rutas_gpkg, capas_zonales = preparar_capas_sin_dedicadas(config, directorio_raiz, dir_auditoria)
    else:
        print(f"\n[1/6] Capas no dedicadas ya existen en: {dir_auditoria}")

    # 2. Modelo sin dedicadas
    if not os.path.exists(out_model_path):
        if capas_zonales is None:
            rutas_gpkg, capas_zonales = preparar_capas_sin_dedicadas(config, directorio_raiz, dir_auditoria)
        print("\n[2/6] Entrenando Random Forest sin dedicadas...")
        modelo, _ = entrenar_modelo_sin_dedicadas(
            config, directorio_raiz, capas_zonales, out_model_path, random_state=args.random_state
        )
    else:
        print(f"\n[2/6] Modelo ya existente en: {out_model_path}")
        import joblib
        modelo = joblib.load(out_model_path)

    # 3. Mapa continuo de aptitud sin dedicadas
    if not os.path.exists(out_tif_path):
        print("\n[3/6] Generando ráster de aptitud espacial sin dedicadas...")
        generar_mapa_aptitud_sin_dedicadas(config, directorio_raiz, modelo, rutas_gpkg, out_tif_path)
    else:
        print(f"\n[3/6] Ráster de aptitud sin dedicadas ya existe en: {out_tif_path}")

    # 3. Alineación y cruce
    print("\n[4/6] Alineando rásters y cruzando aptitud vs. rendimiento físico...")
    umbral = config.get('postgis_validation', {}).get('prob_min', 0.70)
    cruce = alinear_y_cruzar(out_tif_path, rendimiento_path, umbral=umbral)

    # 4. Cálculo estadístico y bootstrap
    print(f"\n[5/6] Calculando métricas, correlaciones de Spearman y Bootstrap de bloques ({args.bloque_km} km, B={args.n_bootstrap})...")
    resultados = calcular_estadisticas_desacople(
        cruce, tamano_bloque_km=args.bloque_km,
        n_bootstrap=args.n_bootstrap, random_state=args.random_state
    )

    # 5. Exportar CSV y JSON
    print("\n[6/6] Exportando tabla CSV consolidada y metadatos...")
    df_csv = exportar_tabla_csv(resultados, out_csv_results)
    exportar_tabla_csv(resultados, out_csv_auditoria)

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("RESULTADOS DEL DESACOPLE APTITUD-RENDIMIENTO")
    print("=" * 90)
    r = resultados['resumen_rendimiento']
    reg = resultados['region']
    apt = resultados['aptas']

    print(f"Celdas válidas evaluadas:         {r['celdas_validas']:,}")
    print(f"Celdas aptas (P >= {umbral:.2f}):           {r['celdas_aptas']:,} ({r['pct_aptas']}%)")
    print(f"Bloques espaciales de 30 km:      {reg['n_bloques_30km']} en región | {apt['n_bloques_30km']} con celdas aptas")
    print("-" * 90)
    print(f"Rendimiento regional (kWh/kWp):    media = {r['rendimiento_region']['media']} | mediana = {r['rendimiento_region']['mediana']}")
    print(f"Rendimiento en aptas (kWh/kWp):    media = {r['rendimiento_aptas']['media']} | mediana = {r['rendimiento_aptas']['mediana']}")
    print(f"Ganancia relativa en aptas:        media = {r['ganancia_media_pct']:+}% | mediana = {r['ganancia_mediana_pct']:+}%")
    print("-" * 90)
    sc = reg['spearman_celda']
    sb = reg['spearman_bloque']
    sca = apt['spearman_celda']
    sba = apt['spearman_bloque']
    print(f"Spearman por celda (regional):    rho = {sc['rho']:.4f} (IC 95%: [{sc['ic_95_inferior']:.4f}, {sc['ic_95_superior']:.4f}])")
    print(f"Spearman por bloque (regional):   rho = {sb['rho']:.4f} (IC 95%: [{sb['ic_95_inferior']:.4f}, {sb['ic_95_superior']:.4f}]) | p = {sb['p_valor']}")
    print(f"Spearman por celda (aptas):       rho = {sca['rho']:.4f} (IC 95%: [{sca['ic_95_inferior']:.4f}, {sca['ic_95_superior']:.4f}])")
    print(f"Spearman por bloque (aptas):      rho = {sba['rho']:.4f} (IC 95%: [{sba['ic_95_inferior']:.4f}, {sba['ic_95_superior']:.4f}]) | p = {sba['p_valor']}")
    print("-" * 90)
    print(f"Tabla CSV generada en:            {out_csv_results}")
    print(f"JSON con metadatos en:            {out_json_path}")
    print("=" * 90)
    return 0


if __name__ == '__main__':
    sys.exit(main())
