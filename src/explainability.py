"""Explicabilidad del modelo de aptitud con SHAP (deuda pendiente de PEP1).

Usa `TreeExplainer` (exacto para modelos de árboles) sobre el Random Forest ya entrenado
(`model_rf.pkl`), SIN re-entrenar. Responde *por qué* el modelo considera apto un sitio:
qué variables empujan la probabilidad hacia arriba o abajo, y con qué magnitud.

Cierre de narrativa con T2/T3: si SHAP confirma que el `ghi` casi no contribuye a la
decisión de aptitud, eso explica mecánicamente el hallazgo de que las zonas aptas no
tienen mayor rendimiento físico (aptitud y producción están desacopladas: el modelo
decide por logística, no por energía).
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend no interactivo: solo guardamos figuras
import matplotlib.pyplot as plt
import geopandas as gpd
import joblib
import shap

from src.features import FEATURES

# ESRI trunca los nombres de columna del shapefile a 10 caracteres (ver AGENTS.md).
RENAME_ESRI = {
    "dist_trans": "dist_transmision",
    "dist_almac": "dist_almacen",
    "dist_subs": "dist_subestaciones",
}


def _cargar_dataset(dataset_path):
    """Carga el dataset de entrenamiento con los nombres de columna restaurados."""
    g = gpd.read_file(dataset_path)
    g = g.rename(columns={k: v for k, v in RENAME_ESRI.items() if k in g.columns})
    return g


def explicar(model_path, dataset_path, figures_dir, out_json, rendimiento_path=None):
    """Calcula SHAP global, guarda figuras y (si hay rendimiento) el cruce. Devuelve dict."""
    model = joblib.load(model_path)
    g = _cargar_dataset(dataset_path)
    X = g[FEATURES]  # orden canónico (coincide con model.feature_names_in_)

    # SHAP para la clase 1 (apto). shap_values devuelve (n, n_features, n_clases).
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv1 = sv[:, :, 1] if sv.ndim == 3 else sv  # contribuciones hacia "apto"

    # --- Importancia global: media(|shap|) por feature, en % ---
    imp = np.abs(sv1).mean(axis=0)
    imp_pct = 100.0 * imp / imp.sum()
    direccion = sv1.mean(axis=0)  # signo: + empuja a apto, - aleja
    importancias = sorted(
        [{"feature": f, "importancia_pct": round(float(p), 2),
          "shap_medio": round(float(d), 5)}
         for f, p, d in zip(FEATURES, imp_pct, direccion)],
        key=lambda x: x["importancia_pct"], reverse=True,
    )

    # --- Figuras ---
    os.makedirs(figures_dir, exist_ok=True)
    bar_path = os.path.join(figures_dir, "shap_summary_bar.png")
    bees_path = os.path.join(figures_dir, "shap_summary_beeswarm.png")

    shap.summary_plot(sv1, X, plot_type="bar", show=False)
    plt.title("Importancia SHAP (media |valor|) — aptitud solar")
    plt.tight_layout(); plt.savefig(bar_path, dpi=200, bbox_inches="tight"); plt.close()

    shap.summary_plot(sv1, X, show=False)
    plt.title("SHAP por variable — aptitud solar")
    plt.tight_layout(); plt.savefig(bees_path, dpi=200, bbox_inches="tight"); plt.close()

    resultado = {
        "n_muestras": int(len(X)),
        "importancias_shap": importancias,
        "figuras": {"bar": bar_path, "beeswarm": bees_path},
    }

    # --- Cruce con el rendimiento físico (opcional) ---
    if rendimiento_path and os.path.exists(rendimiento_path):
        resultado["cruce_rendimiento"] = _cruzar_con_rendimiento(model, g, X, rendimiento_path)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    return resultado


def _cruzar_con_rendimiento(model, g, X, rendimiento_path):
    """¿La aptitud del modelo se relaciona con el rendimiento físico? (cruce de la guía).

    Muestrea el rendimiento (kWh/kWp/año) en los puntos de entrenamiento y mide la
    correlación de Spearman entre la probabilidad de aptitud y el rendimiento. Una
    correlación cercana a 0 confirma, a nivel de punto, el desacople hallado en T2.
    """
    import rasterio
    from scipy.stats import spearmanr

    with rasterio.open(rendimiento_path) as ren:
        pts = g.geometry.to_crs(ren.crs)
        coords = [(p.x, p.y) for p in pts]
        rend = np.array([v[0] for v in ren.sample(coords)], dtype=float)
        nodata = ren.nodata

    prob = model.predict_proba(X)[:, 1]
    valido = np.isfinite(rend) & (rend != nodata) & (rend > 0)
    if valido.sum() < 10:
        return {"nota": "muy pocos puntos con rendimiento válido para correlacionar"}

    rho, pval = spearmanr(prob[valido], rend[valido])
    ghi_rho, _ = spearmanr(X["ghi"].values[valido], rend[valido])
    return {
        "n_puntos": int(valido.sum()),
        "spearman_aptitud_vs_rendimiento": round(float(rho), 4),
        "pvalue": round(float(pval), 4),
        "spearman_ghi_vs_rendimiento": round(float(ghi_rho), 4),
        "interpretacion": (
            "Correlación baja aptitud–rendimiento confirma el desacople de T2: el modelo "
            "no persigue la producción física. "
            if abs(rho) < 0.3 else
            "Correlación no despreciable aptitud–rendimiento: revisar respecto a T2. "
        ),
    }
