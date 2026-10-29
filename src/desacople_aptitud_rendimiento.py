"""Análisis de desacople aptitud–rendimiento (punto 2 de la retroalimentación del profesor).

Pregunta: ¿las celdas que el modelo considera aptas producen más energía que la región?

Es un ANÁLISIS sobre productos del pipeline, no un entrenamiento: usa el mapa oficial de
aptitud (`data/results/aptitud/mapa_probabilidad_aptitud.tif`), que con `control_fuga.excluir_dedicadas`
ya proviene del modelo sin infraestructura dedicada y con hiperparámetros de Optuna (ver
src/infraestructura.py), y el rendimiento físico de solarpv-rs v0.4.0 (EPSG:32719).

Qué celdas entran al análisis:
  - con aptitud válida y rendimiento > 0: el motor deja en cero las celdas sin dato de GHI
    observado, y promediarlas como si produjeran cero sesgaría la región hacia abajo;
  - fuera de las zonas de exclusión (SNASPE, masas lacustres, áreas pobladas): no son
    terreno construible, y su aptitud forzada a 0 llenaría el Spearman de empates.

Calcula rendimiento medio y mediano de aptas vs. región, y la correlación de Spearman
aptitud–rendimiento por celda y agregada en bloques de 30 km del SBCV, con intervalos por
bootstrap de bloques (los bloques, no las celdas, son el tamaño de muestra efectivo).
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.stats import spearmanr

from src.cruce_aptitud_rendimiento import _alinear_rendimiento, NODATA


def mascara_zonas_exclusion(rutas_exclusion: list, aptitud_path: str) -> np.ndarray:
    """Máscara booleana (True = zona de exclusión) sobre la grilla del mapa de aptitud.

    Rasteriza con el mismo criterio que el mapa de aptitud (`get_rasterized_mask`,
    `all_touched=True`), para excluir exactamente las celdas que el pipeline forzó a 0.
    """
    from scripts.generate_suitability_map import get_rasterized_mask

    with rasterio.open(aptitud_path) as apt:
        forma, transform, crs = (apt.height, apt.width), apt.transform, apt.crs
    mascara = np.zeros(forma, dtype=bool)
    for ruta in rutas_exclusion:
        if not ruta or not os.path.exists(ruta):
            print(f"  [AVISO] Capa de exclusión no encontrada, se omite: {ruta}")
            continue
        capa = gpd.read_file(ruta).to_crs(crs)
        mascara |= get_rasterized_mask(capa, forma, transform, fill=0, default_value=1) == 1
    return mascara


def alinear_y_cruzar(aptitud_path: str, rendimiento_path: str, umbral: float = 0.70,
                     mascara_exclusion: np.ndarray | None = None) -> dict:
    """Alinea celda a celda el mapa de aptitud y el mapa de rendimiento físico.

    `mascara_exclusion` (True = excluir) quita las zonas de exclusión territorial. También se
    excluyen las celdas con rendimiento <= 0 (sin dato de GHI observado en el motor).

    Devuelve un diccionario con los arrays alineados, máscaras y estadísticas descriptivas base.
    """
    with rasterio.open(aptitud_path) as apt:
        aptitud = apt.read(1).astype(np.float32)
        transform = apt.transform
        crs = apt.crs
        rendimiento = _alinear_rendimiento(rendimiento_path, apt)

    con_aptitud = (aptitud != NODATA) & np.isfinite(aptitud) & np.isfinite(rendimiento)
    sin_ghi = con_aptitud & (rendimiento <= 0)
    en_exclusion = (con_aptitud & mascara_exclusion if mascara_exclusion is not None
                    else np.zeros_like(con_aptitud))
    base_valida = con_aptitud & ~sin_ghi & ~en_exclusion
    if not base_valida.any():
        raise ValueError("No hay celdas válidas coincidentes entre aptitud y rendimiento.")

    aptas = base_valida & (aptitud >= umbral)

    y_region = rendimiento[base_valida]
    y_aptas = rendimiento[aptas]

    stats_descriptivas = {
        'umbral_probabilidad': float(umbral),
        'celdas_excluidas_sin_ghi': int(sin_ghi.sum()),
        'celdas_excluidas_zonas_exclusion': int((en_exclusion & ~sin_ghi).sum()),
        'celdas_validas': int(base_valida.sum()),
        'celdas_aptas': int(aptas.sum()),
        'pct_aptas': round(100.0 * float(aptas.sum()) / float(base_valida.sum()), 2),
        'rendimiento_region': {
            'media': round(float(y_region.mean()), 2),
            'mediana': round(float(np.median(y_region)), 2),
            'min': round(float(y_region.min()), 2),
            'max': round(float(y_region.max()), 2),
        },
        'rendimiento_aptas': {
            'media': round(float(y_aptas.mean()), 2) if len(y_aptas) else None,
            'mediana': round(float(np.median(y_aptas)), 2) if len(y_aptas) else None,
            'min': round(float(y_aptas.min()), 2) if len(y_aptas) else None,
            'max': round(float(y_aptas.max()), 2) if len(y_aptas) else None,
        },
        'ganancia_media_pct': (
            round(float(100.0 * (y_aptas.mean() / y_region.mean() - 1.0)), 2)
            if len(y_aptas) else None
        ),
        'ganancia_mediana_pct': (
            round(float(100.0 * (np.median(y_aptas) / np.median(y_region) - 1.0)), 2)
            if len(y_aptas) else None
        ),
    }

    return {
        'aptitud': aptitud,
        'rendimiento': rendimiento,
        'base_valida': base_valida,
        'aptas': aptas,
        'transform': transform,
        'crs': crs,
        'stats': stats_descriptivas,
    }


def asignar_bloques_a_celdas(transform, mascara: np.ndarray, tamano_bloque_km: float = 30.0) -> np.ndarray:
    """Calcula el identificador único de bloque espacial de 30 km para cada celda en la máscara."""
    rows, cols = np.where(mascara)
    xs = transform[2] + cols * transform[0] + transform[0] / 2.0
    ys = transform[5] + rows * transform[4] + transform[4] / 2.0
    bloque_m = tamano_bloque_km * 1000.0
    bx = np.floor(xs / bloque_m).astype(np.int64)
    by = np.floor(ys / bloque_m).astype(np.int64)
    block_ids = bx * 1_000_003 + by
    return block_ids


def calcular_estadisticas_desacople(cruce: dict, tamano_bloque_km: float = 30.0,
                                   n_bootstrap: int = 1000, random_state: int = 42) -> dict:
    """Calcula todas las métricas de desacople requeridas:

    - Rendimiento medio y mediano regional vs. celdas aptas.
    - Correlación de Spearman por celda y por bloque de 30 km.
    - Intervalo de confianza por Bootstrap espacial de bloques al 95%.
    """
    aptitud = cruce['aptitud']
    rendimiento = cruce['rendimiento']
    base_valida = cruce['base_valida']
    aptas = cruce['aptas']
    transform = cruce['transform']
    stats_base = cruce['stats']

    # Spearman y el bootstrap de bloques necesitan al menos un par de bloques con celdas
    # aptas; sin ellos scipy/numpy fallarían con un error que no dice la causa.
    n_aptas = int(aptas.sum())
    if n_aptas < 3:
        raise ValueError(f"Solo hay {n_aptas} celdas aptas con rendimiento válido: no se puede "
                         "medir el desacople. Revisa el umbral, la zona o el ráster de rendimiento.")

    # 1. Vectores de celdas válidas a nivel regional
    apt_reg = aptitud[base_valida]
    rend_reg = rendimiento[base_valida]
    bloques_reg = asignar_bloques_a_celdas(transform, base_valida, tamano_bloque_km)

    # 2. Correlación de Spearman celda a celda (Región)
    rho_celda_reg = spearmanr(apt_reg, rend_reg).statistic

    # 3. Agregación vectorizada por bloques espaciales (30 km) a nivel regional
    unique_b, block_indices_reg = np.unique(bloques_reg, return_inverse=True)
    n_bloques_reg = len(unique_b)
    counts_reg = np.bincount(block_indices_reg)
    apt_b_reg = np.bincount(block_indices_reg, weights=apt_reg) / counts_reg
    rend_b_reg = np.bincount(block_indices_reg, weights=rend_reg) / counts_reg

    rho_bloque_reg, pval_bloque_reg = spearmanr(apt_b_reg, rend_b_reg)

    # 4. Bootstrap espacial de bloques al 95% para Spearman regional
    rng = np.random.default_rng(random_state)
    boot_rhos_bloque_reg = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx_b = rng.choice(n_bloques_reg, size=n_bloques_reg, replace=True)
        boot_rhos_bloque_reg[i] = spearmanr(apt_b_reg[idx_b], rend_b_reg[idx_b]).statistic

    ic_bloque_reg = np.percentile(boot_rhos_bloque_reg, [2.5, 97.5])
    se_bloque_reg = float(boot_rhos_bloque_reg.std())

    # Cluster/Block bootstrap para celda regional (submuestreo representativo 1:100 para eficiencia extrema)
    sub_paso = max(1, len(apt_reg) // 200_000)
    sub_idx = np.arange(0, len(apt_reg), sub_paso)
    sub_apt = apt_reg[sub_idx]
    sub_rend = rend_reg[sub_idx]
    sub_bi = block_indices_reg[sub_idx]

    # Pre-indexar arrays por bloque (0..n_bloques_reg - 1)
    block_cell_data = [
        (sub_apt[sub_bi == bi], sub_rend[sub_bi == bi])
        for bi in range(n_bloques_reg)
    ]

    boot_rhos_celda_reg = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_bi = rng.choice(n_bloques_reg, size=n_bloques_reg, replace=True)
        a_parts = [block_cell_data[bi][0] for bi in sampled_bi if len(block_cell_data[bi][0]) > 0]
        r_parts = [block_cell_data[bi][1] for bi in sampled_bi if len(block_cell_data[bi][1]) > 0]
        if a_parts:
            boot_rhos_celda_reg[i] = spearmanr(np.concatenate(a_parts), np.concatenate(r_parts)).statistic
        else:
            boot_rhos_celda_reg[i] = rho_celda_reg

    ic_celda_reg = np.percentile(boot_rhos_celda_reg, [2.5, 97.5])
    se_celda_reg = float(boot_rhos_celda_reg.std())

    # 5. Evaluación análoga en el subconjunto de celdas aptas (P >= 0.70)
    apt_aptas = aptitud[aptas]
    rend_aptas = rendimiento[aptas]
    bloques_aptas = asignar_bloques_a_celdas(transform, aptas, tamano_bloque_km)

    rho_celda_aptas = spearmanr(apt_aptas, rend_aptas).statistic

    unique_ba, block_indices_aptas = np.unique(bloques_aptas, return_inverse=True)
    n_bloques_aptas = len(unique_ba)
    counts_aptas = np.bincount(block_indices_aptas)
    apt_b_aptas = np.bincount(block_indices_aptas, weights=apt_aptas) / counts_aptas
    rend_b_aptas = np.bincount(block_indices_aptas, weights=rend_aptas) / counts_aptas

    rho_bloque_aptas, pval_bloque_aptas = spearmanr(apt_b_aptas, rend_b_aptas)

    boot_rhos_bloque_aptas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx_ba = rng.choice(n_bloques_aptas, size=n_bloques_aptas, replace=True)
        boot_rhos_bloque_aptas[i] = spearmanr(apt_b_aptas[idx_ba], rend_b_aptas[idx_ba]).statistic

    ic_bloque_aptas = np.percentile(boot_rhos_bloque_aptas, [2.5, 97.5])
    se_bloque_aptas = float(boot_rhos_bloque_aptas.std())

    # Cluster/Block bootstrap para celda en subconjunto de celdas aptas
    sub_paso_a = max(1, len(apt_aptas) // 100_000)
    sub_idx_a = np.arange(0, len(apt_aptas), sub_paso_a)
    sub_apt_a = apt_aptas[sub_idx_a]
    sub_rend_a = rend_aptas[sub_idx_a]
    sub_bi_a = block_indices_aptas[sub_idx_a]

    block_cell_data_aptas = [
        (sub_apt_a[sub_bi_a == bi], sub_rend_a[sub_bi_a == bi])
        for bi in range(n_bloques_aptas)
    ]

    boot_rhos_celda_aptas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_bia = rng.choice(n_bloques_aptas, size=n_bloques_aptas, replace=True)
        a_parts = [block_cell_data_aptas[bi][0] for bi in sampled_bia if len(block_cell_data_aptas[bi][0]) > 0]
        r_parts = [block_cell_data_aptas[bi][1] for bi in sampled_bia if len(block_cell_data_aptas[bi][1]) > 0]
        if a_parts:
            boot_rhos_celda_aptas[i] = spearmanr(np.concatenate(a_parts), np.concatenate(r_parts)).statistic
        else:
            boot_rhos_celda_aptas[i] = rho_celda_aptas

    ic_celda_aptas = np.percentile(boot_rhos_celda_aptas, [2.5, 97.5])
    se_celda_aptas = float(boot_rhos_celda_aptas.std())

    resultados = {
        'resumen_rendimiento': stats_base,
        'tamano_bloque_km': tamano_bloque_km,
        'n_bootstrap': n_bootstrap,
        'random_state': random_state,
        'region': {
            'n_celdas': int(base_valida.sum()),
            'n_bloques_30km': n_bloques_reg,
            'spearman_celda': {
                'rho': round(float(rho_celda_reg), 4),
                'ic_95_inferior': round(float(ic_celda_reg[0]), 4),
                'ic_95_superior': round(float(ic_celda_reg[1]), 4),
                'std_bootstrap': round(se_celda_reg, 4),
            },
            'spearman_bloque': {
                'rho': round(float(rho_bloque_reg), 4),
                'p_valor': round(float(pval_bloque_reg), 6),
                'ic_95_inferior': round(float(ic_bloque_reg[0]), 4),
                'ic_95_superior': round(float(ic_bloque_reg[1]), 4),
                'std_bootstrap': round(se_bloque_reg, 4),
            },
        },
        'aptas': {
            'n_celdas': int(aptas.sum()),
            'n_bloques_30km': n_bloques_aptas,
            'spearman_celda': {
                'rho': round(float(rho_celda_aptas), 4),
                'ic_95_inferior': round(float(ic_celda_aptas[0]), 4),
                'ic_95_superior': round(float(ic_celda_aptas[1]), 4),
                'std_bootstrap': round(se_celda_aptas, 4),
            },
            'spearman_bloque': {
                'rho': round(float(rho_bloque_aptas), 4),
                'p_valor': round(float(pval_bloque_aptas), 4),
                'ic_95_inferior': round(float(ic_bloque_aptas[0]), 4),
                'ic_95_superior': round(float(ic_bloque_aptas[1]), 4),
                'std_bootstrap': round(se_bloque_aptas, 4),
            },
        },
    }

    return resultados


def exportar_tabla_csv(resultados: dict, out_csv_path: str) -> pd.DataFrame:
    """Construye y guarda la tabla CSV con todas las métricas del análisis de desacople."""
    r = resultados['resumen_rendimiento']
    reg = resultados['region']
    apt = resultados['aptas']

    filas = [
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_medio_regional',
            'ambito': 'Region Completa (Antofagasta + Atacama)',
            'valor': r['rendimiento_region']['media'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Media de rendimiento físico anual en todas las celdas válidas de la región',
        },
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_mediano_regional',
            'ambito': 'Region Completa (Antofagasta + Atacama)',
            'valor': r['rendimiento_region']['mediana'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Mediana de rendimiento físico anual en la región',
        },
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_medio_aptas',
            'ambito': 'Celdas Aptas (Probabilidad RF >= 0.70)',
            'valor': r['rendimiento_aptas']['media'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Media de rendimiento físico anual en celdas clasificadas como aptas',
        },
        {
            'categoria': 'Rendimiento Físico',
            'metrica': 'rendimiento_mediano_aptas',
            'ambito': 'Celdas Aptas (Probabilidad RF >= 0.70)',
            'valor': r['rendimiento_aptas']['mediana'],
            'unidad': 'kWh/kWp/año',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Mediana de rendimiento físico anual en celdas aptas',
        },
        {
            'categoria': 'Comparación Rendimiento',
            'metrica': 'ganancia_media_aptas_vs_region',
            'ambito': 'Aptas vs Región',
            'valor': f"{r['ganancia_media_pct']:+}%",
            'unidad': '%',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Diferencia porcentual del rendimiento medio en zonas aptas respecto al regional',
        },
        {
            'categoria': 'Comparación Rendimiento',
            'metrica': 'ganancia_mediana_aptas_vs_region',
            'ambito': 'Aptas vs Región',
            'valor': f"{r['ganancia_mediana_pct']:+}%",
            'unidad': '%',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Diferencia porcentual de la mediana en zonas aptas respecto al regional',
        },
        {
            'categoria': 'Cobertura Territorial',
            'metrica': 'celdas_validas_totales',
            'ambito': 'Region Completa',
            'valor': r['celdas_validas'],
            'unidad': 'celdas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número total de celdas válidas evaluadas en la grilla de 100 m',
        },
        {
            'categoria': 'Cobertura Territorial',
            'metrica': 'celdas_excluidas_sin_ghi',
            'ambito': 'Region Completa',
            'valor': r.get('celdas_excluidas_sin_ghi', ''),
            'unidad': 'celdas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Celdas con rendimiento <= 0 (sin dato de GHI observado en el motor), excluidas',
        },
        {
            'categoria': 'Cobertura Territorial',
            'metrica': 'celdas_excluidas_zonas_exclusion',
            'ambito': 'Region Completa',
            'valor': r.get('celdas_excluidas_zonas_exclusion', ''),
            'unidad': 'celdas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Celdas en SNASPE, masas lacustres o áreas pobladas (no construibles), excluidas',
        },
        {
            'categoria': 'Cobertura Territorial',
            'metrica': 'celdas_aptas_totales',
            'ambito': 'Zonas Aptas (P >= 0.70)',
            'valor': r['celdas_aptas'],
            'unidad': 'celdas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': f"Número de celdas con aptitud >= 0.70 ({r['pct_aptas']}% del territorio)",
        },
        {
            'categoria': 'Estructura Espacial',
            'metrica': 'n_bloques_espaciales_30km_region',
            'ambito': 'Region Completa',
            'valor': reg['n_bloques_30km'],
            'unidad': 'bloques',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número de bloques regulares de 30 km con celdas válidas (SBCV)',
        },
        {
            'categoria': 'Estructura Espacial',
            'metrica': 'n_bloques_espaciales_30km_aptas',
            'ambito': 'Zonas Aptas',
            'valor': apt['n_bloques_30km'],
            'unidad': 'bloques',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': 'Número de bloques de 30 km que contienen al menos una celda apta',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_celda_regional',
            'ambito': 'Por Celda (Región Completa)',
            'valor': reg['spearman_celda']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': reg['spearman_celda']['ic_95_inferior'],
            'ic_95_superior': reg['spearman_celda']['ic_95_superior'],
            'p_valor': '',
            'descripcion': ('Correlación de Spearman entre aptitud y rendimiento celda a celda regional. '
                            'Sin p-valor: las celdas están autocorrelacionadas; usar el IC por bloques'),
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_bloques_30km_regional',
            'ambito': 'Agregada por Bloques de 30 km (Región)',
            'valor': reg['spearman_bloque']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': reg['spearman_bloque']['ic_95_inferior'],
            'ic_95_superior': reg['spearman_bloque']['ic_95_superior'],
            'p_valor': reg['spearman_bloque']['p_valor'],
            'descripcion': 'Correlación de Spearman agregada a nivel de bloque (SBCV 30 km)',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_celda_aptas',
            'ambito': 'Por Celda (Subconjunto Aptas)',
            'valor': apt['spearman_celda']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': apt['spearman_celda']['ic_95_inferior'],
            'ic_95_superior': apt['spearman_celda']['ic_95_superior'],
            'p_valor': '',
            'descripcion': 'Correlación de Spearman dentro de las celdas clasificadas como aptas',
        },
        {
            'categoria': 'Correlación de Spearman',
            'metrica': 'spearman_rho_bloques_30km_aptas',
            'ambito': 'Agregada por Bloques de 30 km (Aptas)',
            'valor': apt['spearman_bloque']['rho'],
            'unidad': 'rho',
            'ic_95_inferior': apt['spearman_bloque']['ic_95_inferior'],
            'ic_95_superior': apt['spearman_bloque']['ic_95_superior'],
            'p_valor': apt['spearman_bloque']['p_valor'],
            'descripcion': 'Correlación de Spearman agregada por bloque en celdas aptas',
        },
        {
            'categoria': 'Metodología Bootstrap',
            'metrica': 'replicas_bootstrap_bloques',
            'ambito': 'Bloques Espaciales 30 km',
            'valor': resultados['n_bootstrap'],
            'unidad': 'réplicas',
            'ic_95_inferior': '',
            'ic_95_superior': '',
            'p_valor': '',
            'descripcion': f"Remuestreo con reemplazo de bloques espaciales (random_state={resultados['random_state']})",
        },
    ]

    df_out = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False, encoding='utf-8')
    print(f"  [TABLA CSV] Guardada exitosamente en: {out_csv_path}")
    return df_out
