"""
Script principal para la validacion de coordenadas de instalaciones solares detectadas
y su posterior generacion de scripts SQL / carga en PostGIS.
"""

from __future__ import annotations
import os
import sys
import argparse
import yaml
import json
import geopandas as gpd

# Permite importar módulos desde la raíz del proyecto
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(directorio_raiz)

from src.preprocessing import cargar_capas_vectoriales
from src.postgis_validation import (
    CoordinateValidator,
    extraer_instalaciones_detectadas,
    export_postgis_sql,
    load_to_postgis_db,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validacion de Coordenadas de Instalaciones Solares e Ingestion en PostGIS"
    )
    parser.add_argument("--config", default="config.yaml", help="Ruta al archivo de configuracion")
    parser.add_argument(
        "--input-vector",
        default=None,
        help="Ruta a un Shapefile/GeoPackage/GeoJSON de instalaciones detectadas a validar",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=None,
        help="Umbral de probabilidad de aptitud para filtrar instalaciones",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta la validacion y genera reportes/SQL sin intentar conectarse a la BD PostGIS live",
    )
    args = parser.parse_args()

    print("=== INICIANDO VALIDACIÓN DE COORDENADAS PARA POSTGIS ===")

    # 1. Cargar configuraciones
    ruta_config = os.path.join(directorio_raiz, args.config)
    with open(ruta_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths_raw = config["paths"]["raw"]
    paths_processed = config["paths"]["processed"]
    paths_results = config["paths"]["results"]
    pg_config = config.get("postgis_validation", {})

    prob_min = args.prob_threshold if args.prob_threshold is not None else pg_config.get("prob_min", 0.70)
    target_srid = pg_config.get("target_srid", 32719)

    # 2. Cargar capas de exclusion y regiones de estudio
    print("Cargando capas geograficas de exclusion y limites de estudio...")
    vectores = cargar_capas_vectoriales(paths_raw["vectores"])

    regiones_gdf = vectores["regiones"]
    regiones_norte = regiones_gdf[regiones_gdf["REGION"].isin(["Antofagasta", "Atacama"])]

    exclusion_gdfs = {
        "snap": vectores.get("snap"),
        "masas_lacustres": vectores.get("masas_lacustres"),
        "areas_pobladas": vectores.get("areas_pobladas"),
    }

    # 3. Obtener GeoDataFrame de instalaciones a validar
    if args.input_vector and os.path.exists(args.input_vector):
        print(f"Cargando instalaciones detectadas desde archivo vectorial: {args.input_vector}")
        gdf_detected = gpd.read_file(args.input_vector)
    else:
        raster_prob = os.path.join(directorio_raiz, "data/results/mapa_probabilidad_aptitud.tif")
        if not os.path.exists(raster_prob):
            print(
                f"[AVISO] No se encontró {raster_prob}. Intentando usar el dataset de entrenamiento..."
            )
            dataset_ml = paths_results.get("dataset_ml")
            if dataset_ml and os.path.exists(dataset_ml):
                gdf_detected = gpd.read_file(dataset_ml)
                gdf_detected = gdf_detected[gdf_detected["clase"] == 1].copy()
            else:
                raise FileNotFoundError(
                    "No hay datos de entrada para validar. Ejecute `run_pipeline.py` primero "
                    "o especifique `--input-vector`."
                )
        else:
            gdf_detected = extraer_instalaciones_detectadas(
                raster_prob_path=raster_prob,
                raster_ghi_path=os.path.join(directorio_raiz, paths_processed["ghi_32719"]),
                raster_slope_path=os.path.join(directorio_raiz, paths_processed["slope"]),
                raster_dem_path=os.path.join(directorio_raiz, paths_processed["dem_32719"]),
                prob_min=prob_min,
                min_area_m2=pg_config.get("min_area_m2", 10000.0),
            )

    print(f"Total de candidatos a instalaciones a validar: {len(gdf_detected)}")

    # 4. Inicializar validador
    validator = CoordinateValidator(
        target_srid=target_srid,
        bounds_utm=pg_config.get("bounds_utm"),
        bounds_geo=pg_config.get("bounds_geo"),
        criterios=config.get("criterios"),
        exclusion_gdfs=exclusion_gdfs,
        region_boundary_gdf=regiones_norte,
        duplicate_tolerance_m=pg_config.get("duplicate_tolerance_m", 50.0),
        prob_min=prob_min,
    )

    # 5. Ejecutar validacion
    gdf_valid, gdf_invalid, report = validator.run_full_validation(gdf_detected)

    # 6. Guardar resultados
    out_valid_shp = os.path.join(
        directorio_raiz, paths_results.get("valid_sites_shp", "data/results/instalaciones_validadas.shp")
    )
    out_report_json = os.path.join(
        directorio_raiz, paths_results.get("postgis_report", "data/results/reporte_validacion_postgis.json")
    )
    out_sql = os.path.join(
        directorio_raiz, paths_results.get("postgis_sql", "data/results/cargar_instalaciones_postgis.sql")
    )

    os.makedirs(os.path.dirname(out_valid_shp), exist_ok=True)

    if len(gdf_valid) > 0:
        # Guardar SHP de instalciones validadas (sin columnas complejas de listas/dicts)
        cols_to_save = [c for c in gdf_valid.columns if c not in ["geom_poligono", "motivos_rechazo"]]
        gdf_valid[cols_to_save].to_file(out_valid_shp)
        print(f"  [ÉXITO] Shapefile de instalaciones validadas guardado en: {out_valid_shp}")

    # Guardar reporte JSON
    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [ÉXITO] Reporte de auditoría de validación guardado en: {out_report_json}")

    # Generar script SQL PostGIS
    table_name = pg_config.get("table_name", "instalaciones_solares_optimas")
    schema = pg_config.get("schema", "public")
    export_postgis_sql(gdf_valid, table_name=table_name, schema=schema, srid=target_srid, output_sql_path=out_sql)

    # 7. Ingestion opcional en PostGIS DB
    if not args.dry_run:
        print("\nIntentando conexión e inserción directa en PostGIS...")
        db_loaded = load_to_postgis_db(
            gdf=gdf_valid,
            connection_config=pg_config.get("connection", {}),
            table_name=table_name,
            schema=schema,
            if_exists="replace",
        )
        if not db_loaded:
            print(
                f"  [NOTA] No se realizó inserción en vivo. Puedes ejecutar manualmente el script SQL: {out_sql}"
            )
    else:
        print("\nModo --dry-run activo: Se omite la conexión directa a PostGIS.")

    print("\n=== VALIDACIÓN Y PREPARACIÓN POSTGIS COMPLETADA CON ÉXITO ===")


if __name__ == "__main__":
    main()
