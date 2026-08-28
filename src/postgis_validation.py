"""
Modulo de validacion de coordenadas e instalaciones solares detectadas
previo a su carga en PostGIS.

Este modulo proporciona funciones y clases para:
  1. Extraer instalaciones/zonas optimas detectadas a partir de rasters de aptitud ML.
  2. Validar el Sistema de Referencia Espacial (CRS) y corregir o reproyectar.
  3. Validar limites geograficos y rangos de coordenadas (UTM y WGS84).
  4. Validar integridad de geometrias y detectar duplicados espaciales con indice R-Tree.
  5. Verificar que las coordenadas NO intersecten zonas de exclusion (SNAP, masas lacustres, areas pobladas).
  6. Enforzar criterios tecnicos (pendiente, elevacion, GHI, probabilidad de aptitud).
  7. Generar reportes detallados de auditoria de validacion.
  8. Exportar scripts SQL / DDL para PostGIS y realizar carga segura en base de datos.
"""

from __future__ import annotations
import os
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.validation import explain_validity, make_valid
from shapely.strtree import STRtree
import rasterio
from rasterio.features import shapes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PostGISValidator")


class CoordinateValidator:
    """Clase principal para la validacion rigurosa de coordenadas e instalaciones solares
    previas al almacenamiento en PostGIS.
    """

    def __init__(
        self,
        target_srid: int = 32719,
        bounds_utm: Optional[Dict[str, float]] = None,
        bounds_geo: Optional[Dict[str, float]] = None,
        criterios: Optional[Dict[str, float]] = None,
        exclusion_gdfs: Optional[Dict[str, gpd.GeoDataFrame]] = None,
        region_boundary_gdf: Optional[gpd.GeoDataFrame] = None,
        duplicate_tolerance_m: float = 50.0,
        prob_min: float = 0.70,
    ):
        self.target_srid = target_srid
        self.target_crs = f"EPSG:{target_srid}"
        self.duplicate_tolerance_m = duplicate_tolerance_m
        self.prob_min = prob_min

        # Limites por defecto para el Norte de Chile (UTM 19S EPSG:32719)
        self.bounds_utm = bounds_utm or {
            "min_x": 100000.0,
            "max_x": 900000.0,
            "min_y": 6000000.0,
            "max_y": 8500000.0,
        }

        # Limites geograficos (WGS84 EPSG:4326) para Chile Norte
        self.bounds_geo = bounds_geo or {
            "min_lon": -72.5,
            "max_lon": -66.5,
            "min_lat": -30.0,
            "max_lat": -17.0,
        }

        self.criterios = criterios or {
            "ghi_min": 160.0,
            "slope_max": 15.0,
            "elev_max": 3500.0,
        }

        self.exclusion_gdfs = exclusion_gdfs or {}
        self.region_boundary_gdf = region_boundary_gdf

    def validate_crs(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Verifica y reproyecta el GeoDataFrame al CRS objetivo si es necesario."""
        if gdf.crs is None:
            logger.warning(f"GeoDataFrame no tiene CRS definido. Asumiendo {self.target_crs}.")
            gdf = gdf.set_crs(self.target_crs)
        elif gdf.crs.to_string().upper() != self.target_crs.upper():
            logger.info(f"Reproyectando de {gdf.crs.to_string()} a {self.target_crs}...")
            gdf = gdf.to_crs(self.target_crs)
        return gdf

    def validate_geometries_and_coordinates(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, List[List[str]]]:
        """Valida que las geometrias sean no nulas, validas y contengan coordenadas razonables."""
        reasons_list: List[List[str]] = [[] for _ in range(len(gdf))]
        gdf_clean = gdf.copy()

        for idx, (i, row) in enumerate(gdf_clean.iterrows()):
            geom = row.geometry

            # 1. Geometria nula o vacia
            if geom is None or geom.is_empty:
                reasons_list[idx].append("GEOMETRIA_NULA_O_VACIA")
                continue

            # 2. Reparacion o comprobacion de validez
            if not geom.is_valid:
                explanation = explain_validity(geom)
                logger.warning(f"Fila {i}: Geometria invalida ({explanation}). Intentando make_valid()...")
                try:
                    repaired = make_valid(geom)
                    if repaired.is_valid and not repaired.is_empty:
                        gdf_clean.at[i, "geometry"] = repaired
                        geom = repaired
                    else:
                        reasons_list[idx].append(f"GEOMETRIA_INVALIDA: {explanation}")
                        continue
                except Exception as e:
                    reasons_list[idx].append(f"GEOMETRIA_CORRUPTA: {str(e)}")
                    continue

            # 3. Comprobar coordenadas NaN o Inf
            coords = np.array(geom.centroid.coords)
            if np.isnan(coords).any() or np.isinf(coords).any():
                reasons_list[idx].append("COORDENADAS_NAN_O_INF")
                continue

            # 4. Rango de coordenadas segun CRS
            if gdf_clean.crs and gdf_clean.crs.is_geographic:
                lon, lat = geom.centroid.x, geom.centroid.y
                if not (self.bounds_geo["min_lon"] <= lon <= self.bounds_geo["max_lon"]):
                    reasons_list[idx].append(
                        f"LONGITUD_FUERA_DE_RANGO ({lon:.4f} no en [{self.bounds_geo['min_lon']}, {self.bounds_geo['max_lon']}])"
                    )
                if not (self.bounds_geo["min_lat"] <= lat <= self.bounds_geo["max_lat"]):
                    reasons_list[idx].append(
                        f"LATITUD_FUERA_DE_RANGO ({lat:.4f} no en [{self.bounds_geo['min_lat']}, {self.bounds_geo['max_lat']}])"
                    )
            else:
                x, y = geom.centroid.x, geom.centroid.y
                if not (self.bounds_utm["min_x"] <= x <= self.bounds_utm["max_x"]):
                    reasons_list[idx].append(
                        f"EASTING_UTM_FUERA_DE_RANGO ({x:.1f} no en [{self.bounds_utm['min_x']}, {self.bounds_utm['max_x']}])"
                    )
                if not (self.bounds_utm["min_y"] <= y <= self.bounds_utm["max_y"]):
                    reasons_list[idx].append(
                        f"NORTHING_UTM_FUERA_DE_RANGO ({y:.1f} no en [{self.bounds_utm['min_y']}, {self.bounds_utm['max_y']}])"
                    )

        return gdf_clean, reasons_list

    def validate_region_boundary(
        self, gdf: gpd.GeoDataFrame, reasons_list: List[List[str]]
    ) -> List[List[str]]:
        """Verifica mediante Spatial Index que la instalacion este dentro de la region geografica de estudio."""
        if self.region_boundary_gdf is None or len(self.region_boundary_gdf) == 0:
            return reasons_list

        regiones_utm = self.region_boundary_gdf
        if regiones_utm.crs != gdf.crs:
            regiones_utm = regiones_utm.to_crs(gdf.crs)

        # Usar sindex para evaluacion espacial ultrarrapida
        sindex_reg = regiones_utm.sindex

        for idx, (_, row) in enumerate(gdf.iterrows()):
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                candidates_idx = list(sindex_reg.intersection(geom.bounds))
                if not candidates_idx:
                    reasons_list[idx].append("FUERA_DE_REGION_DE_ESTUDIO")
                else:
                    candidates = regiones_utm.iloc[candidates_idx]
                    if not candidates.intersects(geom).any():
                        reasons_list[idx].append("FUERA_DE_REGION_DE_ESTUDIO")

        return reasons_list

    def validate_exclusion_zones(
        self, gdf: gpd.GeoDataFrame, reasons_list: List[List[str]]
    ) -> List[List[str]]:
        """Verifica con R-Tree Spatial Index que la instalacion NO intersecte zonas de exclusion."""
        for clave, excl_gdf in self.exclusion_gdfs.items():
            if excl_gdf is None or len(excl_gdf) == 0:
                continue

            excl_utm = excl_gdf
            if excl_utm.crs != gdf.crs:
                excl_utm = excl_utm.to_crs(gdf.crs)

            sindex_excl = excl_utm.sindex

            for idx, (_, row) in enumerate(gdf.iterrows()):
                geom = row.geometry
                if geom is not None and not geom.is_empty:
                    candidates_idx = list(sindex_excl.intersection(geom.bounds))
                    if candidates_idx:
                        candidates = excl_utm.iloc[candidates_idx]
                        if candidates.intersects(geom).any():
                            reasons_list[idx].append(f"INTERSECTA_ZONA_EXCLUSION_{clave.upper()}")
        return reasons_list

    def validate_technical_criteria(
        self, gdf: gpd.GeoDataFrame, reasons_list: List[List[str]]
    ) -> List[List[str]]:
        """Aplica validaciones sobre las variables tecnicas (GHI, Slope, Elev, Aptitud)."""
        for idx, (_, row) in enumerate(gdf.iterrows()):
            # Pendiente maxima
            if "slope" in row and not pd.isna(row["slope"]):
                if row["slope"] > self.criterios["slope_max"]:
                    reasons_list[idx].append(
                        f"PENDIENTE_EXCEDE_MAXIMA ({row['slope']:.1f}° > {self.criterios['slope_max']}°)"
                    )

            # Elevacion maxima
            if "elev" in row and not pd.isna(row["elev"]):
                if row["elev"] > self.criterios["elev_max"]:
                    reasons_list[idx].append(
                        f"ELEVACION_EXCEDE_MAXIMA ({row['elev']:.0f}m > {self.criterios['elev_max']}m)"
                    )

            # GHI minimo
            if "ghi" in row and not pd.isna(row["ghi"]):
                if row["ghi"] < self.criterios["ghi_min"]:
                    reasons_list[idx].append(
                        f"GHI_BAJO_MINIMO ({row['ghi']:.1f} < {self.criterios['ghi_min']})"
                    )

            # Probabilidad / Score de Aptitud solar
            if "probabilidad" in row and not pd.isna(row["probabilidad"]):
                if row["probabilidad"] < self.prob_min:
                    reasons_list[idx].append(
                        f"APTITUD_BAJO_UMBRAL ({row['probabilidad']:.2f} < {self.prob_min:.2f})"
                    )
            elif "aptitud" in row and not pd.isna(row["aptitud"]):
                if row["aptitud"] < self.prob_min:
                    reasons_list[idx].append(
                        f"APTITUD_BAJO_UMBRAL ({row['aptitud']:.2f} < {self.prob_min:.2f})"
                    )

        return reasons_list

    def validate_duplicates(
        self, gdf: gpd.GeoDataFrame, reasons_list: List[List[str]]
    ) -> List[List[str]]:
        """Identifica duplicados espaciales mediante STRtree a una distancia menor a `duplicate_tolerance_m`."""
        if len(gdf) <= 1:
            return reasons_list

        centroids = [row.geometry.centroid if row.geometry is not None else None for _, row in gdf.iterrows()]
        valid_indices = [i for i, p in enumerate(centroids) if p is not None and not p.is_empty]
        valid_geoms = [centroids[i] for i in valid_indices]

        if len(valid_geoms) <= 1:
            return reasons_list

        tree = STRtree(valid_geoms)
        vistos = set()

        for idx_in_valid, p1 in enumerate(valid_geoms):
            orig_i = valid_indices[idx_in_valid]
            if orig_i in vistos:
                continue

            # Buscar vecinos dentro de la distancia de tolerancia
            near_indices = tree.query(p1.buffer(self.duplicate_tolerance_m))
            for near_idx in near_indices:
                orig_j = valid_indices[near_idx]
                if orig_i != orig_j and orig_j not in vistos:
                    p2 = valid_geoms[near_idx]
                    dist = p1.distance(p2)
                    if dist < self.duplicate_tolerance_m:
                        reasons_list[orig_j].append(
                            f"DUPLICADO_ESPACIAL (a {dist:.1f}m del objeto index={orig_i})"
                        )
                        vistos.add(orig_j)

        return reasons_list

    def run_full_validation(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
        """Ejecuta el pipeline completo de validacion.

        Retorna:
          - gdf_valid: Registro de instalaciones aprobadas
          - gdf_invalid: Registro de instalaciones rechazadas con motivo explicito
          - audit_report: Diccionario con metricas completas del proceso de auditoria
        """
        logger.info(f"Iniciando validacion completa de {len(gdf)} registros...")
        gdf_prep = self.validate_crs(gdf.copy())

        # 1. Geometrias y coordenadas
        gdf_prep, reasons = self.validate_geometries_and_coordinates(gdf_prep)

        # 2. Region de estudio
        reasons = self.validate_region_boundary(gdf_prep, reasons)

        # 3. Zonas de exclusion
        reasons = self.validate_exclusion_zones(gdf_prep, reasons)

        # 4. Criterios tecnicos
        reasons = self.validate_technical_criteria(gdf_prep, reasons)

        # 5. Duplicados espaciales
        reasons = self.validate_duplicates(gdf_prep, reasons)

        # Asignar resultados
        gdf_prep["es_valido"] = [len(r) == 0 for r in reasons]
        gdf_prep["motivos_rechazo"] = ["; ".join(r) if len(r) > 0 else "OK" for r in reasons]
        gdf_prep["fecha_validacion"] = datetime.now().isoformat()

        gdf_valid = gdf_prep[gdf_prep["es_valido"]].copy().reset_index(drop=True)
        gdf_invalid = gdf_prep[~gdf_prep["es_valido"]].copy().reset_index(drop=True)

        # Contar tipos de rechazo
        rejection_summary: Dict[str, int] = {}
        for r_list in reasons:
            for reason in r_list:
                clean_reason = reason.split(" (")[0]
                rejection_summary[clean_reason] = rejection_summary.get(clean_reason, 0) + 1

        audit_report = {
            "total_evaluados": len(gdf),
            "total_validos": len(gdf_valid),
            "total_rechazados": len(gdf_invalid),
            "tasa_aprobacion_pct": round((len(gdf_valid) / len(gdf)) * 100, 2) if len(gdf) > 0 else 0.0,
            "target_srid": self.target_srid,
            "conteo_motivos_rechazo": rejection_summary,
            "fecha_ejecucion": datetime.now().isoformat(),
        }

        logger.info(
            f"Validacion finalizada. Validos: {len(gdf_valid)} | Rechazados: {len(gdf_invalid)} "
            f"({audit_report['tasa_aprobacion_pct']}%)"
        )
        return gdf_valid, gdf_invalid, audit_report


def extraer_instalaciones_detectadas(
    raster_prob_path: str,
    raster_ghi_path: str,
    raster_slope_path: str,
    raster_dem_path: str,
    prob_min: float = 0.70,
    min_area_m2: float = 10000.0,
) -> gpd.GeoDataFrame:
    """Extrae polígonos/centroides de sitios óptimos detectados por el modelo de ML
    a partir del GeoTIFF de aptitud, agregando sus variables ambientales.
    """
    logger.info(f"Extrayendo zonas optimas desde {raster_prob_path} (prob >= {prob_min})...")
    if not os.path.exists(raster_prob_path):
        raise FileNotFoundError(f"No se encuentra el mapa de probabilidad en {raster_prob_path}")

    with rasterio.open(raster_prob_path) as src:
        prob_array = src.read(1)
        transform = src.transform
        crs = src.crs

        # Mascara binaria de pixeles con alta aptitud
        binary_mask = (prob_array >= prob_min) & (prob_array != src.nodata) & (~np.isnan(prob_array))
        mask_uint8 = binary_mask.astype(np.uint8)

        poly_geoms = []
        poly_probs = []

        # connectivity=8: sin esto (default 4), dos píxeles aptos que solo se tocan en
        # diagonal cuentan como polígonos separados, fragmentando artificialmente una zona
        # contigua de alta aptitud en múltiples "candidatos" distintos.
        for geom, val in shapes(mask_uint8, mask=binary_mask, transform=transform, connectivity=8):
            if val == 1:
                shp = shape(geom)
                if shp.area >= min_area_m2:
                    poly_geoms.append(shp)
                    # Muestrear la probabilidad en el centroide de la zona contigua
                    col, row = ~transform * (shp.centroid.x, shp.centroid.y)
                    r, c = int(row), int(col)
                    if 0 <= r < src.height and 0 <= c < src.width:
                        p_val = float(prob_array[r, c])
                    else:
                        p_val = float(prob_min)
                    poly_probs.append(p_val)

    if not poly_geoms:
        logger.warning("No se detectaron zonas que cumplan con el umbral de aptitud y area minima.")
        return gpd.GeoDataFrame(columns=["geometry", "probabilidad"], crs=crs)

    gdf_zones = gpd.GeoDataFrame(
        {
            "id_sitio": [f"SOLAR_SITE_{i+1:04d}" for i in range(len(poly_geoms))],
            "probabilidad": poly_probs,
            "area_m2": [g.area for g in poly_geoms],
            "geometry": [g.centroid for g in poly_geoms],  # Centroides para representacion puntual
            "geom_poligono": poly_geoms,
        },
        crs=crs,
    )

    # Muestrear variables asociadas
    from src.sampling import extraer_valores_puntos

    if os.path.exists(raster_ghi_path):
        gdf_zones = extraer_valores_puntos(gdf_zones, raster_ghi_path, "ghi")
    if os.path.exists(raster_slope_path):
        gdf_zones = extraer_valores_puntos(gdf_zones, raster_slope_path, "slope")
    if os.path.exists(raster_dem_path):
        gdf_zones = extraer_valores_puntos(gdf_zones, raster_dem_path, "elev")

    logger.info(f"Se extrajeron {len(gdf_zones)} candidatos a instalaciones solares.")
    return gdf_zones


def export_postgis_sql(
    gdf: gpd.GeoDataFrame,
    table_name: str = "instalaciones_solares_optimas",
    schema: str = "public",
    srid: int = 32719,
    output_sql_path: Optional[str] = None,
) -> str:
    """Genera instrucciones SQL (DDL + DML) de PostGIS para crear la tabla
    e insertar las instalaciones validadas.
    """
    sql_lines = [
        f"-- Script de Carga PostGIS generado el {datetime.now().isoformat()}",
        "CREATE EXTENSION IF NOT EXISTS postgis;",
        f"CREATE SCHEMA IF NOT EXISTS {schema};",
        f"DROP TABLE IF EXISTS {schema}.{table_name};",
        f"CREATE TABLE {schema}.{table_name} (",
        "    id SERIAL PRIMARY KEY,",
        "    site_id VARCHAR(50),",
        "    probabilidad FLOAT,",
        "    area_m2 FLOAT,",
        "    ghi FLOAT,",
        "    slope FLOAT,",
        "    elev FLOAT,",
        "    es_valido BOOLEAN,",
        "    fecha_validacion TIMESTAMP,",
        f"    geom geometry(Point, {srid})",
        ");",
        "",
        f"CREATE INDEX idx_{table_name}_geom ON {schema}.{table_name} USING GIST (geom);",
        "",
    ]

    for _, row in gdf.iterrows():
        site_id = row.get("id_sitio", "SITE_N/A")
        prob = row.get("probabilidad", 0.0)
        area = row.get("area_m2", 0.0)
        ghi = row.get("ghi", "NULL")
        slope = row.get("slope", "NULL")
        elev = row.get("elev", "NULL")
        valido = "TRUE" if row.get("es_valido", True) else "FALSE"
        fecha = f"'{row.get('fecha_validacion', datetime.now().isoformat())}'"

        geom = row.geometry
        if geom is not None and not geom.is_empty:
            wkt_geom = f"ST_SetSRID(ST_GeomFromText('{geom.centroid.wkt}'), {srid})"
        else:
            wkt_geom = "NULL"

        ghi_val = f"{ghi:.2f}" if isinstance(ghi, (int, float)) and not np.isnan(ghi) else "NULL"
        slope_val = f"{slope:.2f}" if isinstance(slope, (int, float)) and not np.isnan(slope) else "NULL"
        elev_val = f"{elev:.2f}" if isinstance(elev, (int, float)) and not np.isnan(elev) else "NULL"

        insert_stmt = (
            f"INSERT INTO {schema}.{table_name} "
            f"(site_id, probabilidad, area_m2, ghi, slope, elev, es_valido, fecha_validacion, geom) "
            f"VALUES ('{site_id}', {prob:.4f}, {area:.2f}, {ghi_val}, {slope_val}, {elev_val}, {valido}, {fecha}, {wkt_geom});"
        )
        sql_lines.append(insert_stmt)

    sql_content = "\n".join(sql_lines)

    if output_sql_path:
        os.makedirs(os.path.dirname(output_sql_path), exist_ok=True)
        with open(output_sql_path, "w", encoding="utf-8") as f:
            f.write(sql_content)
        logger.info(f"Script SQL de PostGIS guardado en: {output_sql_path}")

    return sql_content


def load_to_postgis_db(
    gdf: gpd.GeoDataFrame,
    connection_config: Dict[str, Any],
    table_name: str = "instalaciones_solares_optimas",
    schema: str = "public",
    if_exists: str = "replace",
) -> bool:
    """Intenta cargar el GeoDataFrame validado a una base de datos PostGIS activa
    utilizando SQLAlchemy y GeoPandas.
    """
    try:
        import os
        from urllib.parse import quote_plus
        from sqlalchemy import create_engine

        user = connection_config.get("user", "postgres")
        # La contraseña puede venir del entorno (PGPASSWORD) para no versionarla en config.yaml.
        pwd = os.environ.get("PGPASSWORD") or connection_config.get("password", "")
        host = connection_config.get("host", "localhost")
        port = connection_config.get("port", 5432)
        dbname = connection_config.get("dbname", "geoinformatica_db")

        # quote_plus escapa caracteres especiales de la contraseña en la URL de conexión.
        connection_url = f"postgresql://{user}:{quote_plus(pwd)}@{host}:{port}/{dbname}"
        logger.info(f"Conectando a PostGIS en {host}:{port}/{dbname}...")
        engine = create_engine(connection_url)

        gdf.to_postgis(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
        )
        logger.info(f"Éxito: {len(gdf)} registros cargados exitosamente en PostGIS ({schema}.{table_name}).")
        return True

    except Exception as e:
        logger.error(f"No se pudo realizar la conexion o carga directa a PostGIS: {e}")
        logger.info("Puedes usar el script SQL generado para cargar los datos manualmente.")
        return False
