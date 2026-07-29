"""
Pruebas unitarias para el modulo de validacion de coordenadas e ingesta a PostGIS (src/postgis_validation.py).
"""

import sys
import os
import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.postgis_validation import CoordinateValidator, export_postgis_sql


class TestPostGISValidation(unittest.TestCase):

    def test_crs_and_geometry_validation(self):
        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3],
                "geometry": [
                    Point(350000, 7350000),  # Valido
                    Point(9999999, 100000), # Fuera de rango Northing/Easting
                    Point(),                # Vacia
                ],
            },
            crs="EPSG:32719",
        )

        validator = CoordinateValidator(target_srid=32719)
        gdf_valid, gdf_invalid, report = validator.run_full_validation(gdf)

        self.assertEqual(report["total_evaluados"], 3)
        self.assertEqual(len(gdf_valid), 1)
        self.assertEqual(len(gdf_invalid), 2)
        self.assertIn("GEOMETRIA_NULA_O_VACIA", gdf_invalid.iloc[1]["motivos_rechazo"])

    def test_technical_criteria_validation(self):
        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3],
                "slope": [10.0, 25.0, 5.0],   # max 15.0
                "elev": [2000.0, 1500.0, 4000.0], # max 3500
                "ghi": [220.0, 200.0, 100.0],    # min 160
                "probabilidad": [0.85, 0.90, 0.50], # min 0.70
                "geometry": [
                    Point(350000, 7350000),
                    Point(351000, 7351000),
                    Point(352000, 7352000),
                ],
            },
            crs="EPSG:32719",
        )

        validator = CoordinateValidator(
            target_srid=32719,
            criterios={"ghi_min": 160.0, "slope_max": 15.0, "elev_max": 3500.0},
            prob_min=0.70,
        )
        gdf_valid, gdf_invalid, report = validator.run_full_validation(gdf)

        self.assertEqual(len(gdf_valid), 1)
        self.assertEqual(gdf_valid.iloc[0]["id"], 1)
        self.assertEqual(len(gdf_invalid), 2)

    def test_exclusion_zones_validation(self):
        snap_poly = Polygon([(349000, 7349000), (350500, 7349000), (350500, 7350500), (349000, 7350500)])
        snap_gdf = gpd.GeoDataFrame({"id": [101]}, geometry=[snap_poly], crs="EPSG:32719")

        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [Point(350000, 7350000), Point(360000, 7360000)],
            },
            crs="EPSG:32719",
        )

        validator = CoordinateValidator(
            target_srid=32719,
            exclusion_gdfs={"snap": snap_gdf},
        )
        gdf_valid, gdf_invalid, report = validator.run_full_validation(gdf)

        self.assertEqual(len(gdf_valid), 1)
        self.assertEqual(gdf_valid.iloc[0]["id"], 2)
        self.assertEqual(len(gdf_invalid), 1)
        self.assertIn("INTERSECTA_ZONA_EXCLUSION_SNAP", gdf_invalid.iloc[0]["motivos_rechazo"])

    def test_duplicate_detection(self):
        gdf = gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "geometry": [Point(350000, 7350000), Point(350005, 7350005)],
            },
            crs="EPSG:32719",
        )

        validator = CoordinateValidator(target_srid=32719, duplicate_tolerance_m=50.0)
        gdf_valid, gdf_invalid, report = validator.run_full_validation(gdf)

        self.assertEqual(len(gdf_valid), 1)
        self.assertEqual(len(gdf_invalid), 1)
        self.assertIn("DUPLICADO_ESPACIAL", gdf_invalid.iloc[0]["motivos_rechazo"])

    def test_sql_generation(self):
        gdf = gpd.GeoDataFrame(
            {
                "id_sitio": ["SOLAR_001"],
                "probabilidad": [0.88],
                "area_m2": [15000.0],
                "ghi": [230.5],
                "slope": [4.2],
                "elev": [1850.0],
                "es_valido": [True],
                "geometry": [Point(350000, 7350000)],
            },
            crs="EPSG:32719",
        )

        sql = export_postgis_sql(gdf, table_name="test_instalaciones", srid=32719)

        self.assertIn("CREATE EXTENSION IF NOT EXISTS postgis;", sql)
        self.assertIn("CREATE TABLE public.test_instalaciones", sql)
        self.assertIn("INSERT INTO public.test_instalaciones", sql)
        self.assertIn("ST_SetSRID(ST_GeomFromText('POINT (350000 7350000)'), 32719)", sql)


if __name__ == "__main__":
    unittest.main()
