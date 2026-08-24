"""Visor interactivo de resultados (T7) — Sitios óptimos para plantas solares.

Mapa web (folium/Leaflet) con las capas de aptitud (RF), rendimiento físico (solarpv-rs)
y su cruce superpuestas sobre un basemap, con control de capas, más paneles de
estadísticas (T2/T3/T6). Lee únicamente `app/assets/` (PNG livianos + manifest + stats),
por lo que funciona idéntico en local y en Streamlit Community Cloud, sin depender de los
.tif pesados ni de un computador encendido.

Ejecutar local:
    streamlit run app/visor.py
Regenerar los assets (tras recomputar los mapas):
    python scripts/generate_web_assets.py --config config.yaml
"""

import os
import json
import base64

import streamlit as st
import folium
from streamlit_folium import st_folium
from branca.element import MacroElement, Element
from jinja2 import Template

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(APP_DIR, "assets")

st.set_page_config(page_title="Sitios óptimos para energía solar — Norte de Chile",
                   layout="wide")

st.markdown("""
<style>
    [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: inherit !important;
    }
    .pie-cartografico { color: var(--text-color); font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def _cargar_json(nombre):
    ruta = os.path.join(ASSETS, nombre)
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _data_uri(png_rel):
    """Codifica un PNG de assets como data URI (para que folium lo embeba sin servidor)."""
    ruta = _ruta_asset(png_rel)
    with open(ruta, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _ruta_asset(ruta_rel):
    """Resuelve rutas del manifest tanto si incluyen ``assets/`` como si no."""
    ruta = os.path.join(APP_DIR, ruta_rel)
    if os.path.exists(ruta):
        return ruta
    return os.path.join(ASSETS, os.path.basename(ruta_rel))


ETIQUETAS = {
    "balanceado": "Aptitud balanceado ML",
    "conservador": "Aptitud perfil conservador",
    "agresivo": "Aptitud perfil agresivo",
    "rendimiento": "Rendimiento (kWh/kWp/año)",
    "cruce": "Cruce aptitud × rendimiento",
    "consenso": "Consenso de perfiles",
    "dominante": "Variable dominante (SHAP)",
}

# Agrupación de capas en el panel lateral (solo se muestran las presentes en el manifest).
# Los perfiles de aptitud y el mapa de consenso van en secciones separadas pero adyacentes
# (ambos responden "dónde", son comparables); la producción física va aparte.
GRUPOS = [
    ("Perfiles de aptitud", ["balanceado", "conservador", "agresivo"]),
    ("Consenso / divergencia", ["consenso"]),
    ("Producción física", ["rendimiento", "cruce"]),
    ("Explicabilidad espacial (SHAP)", ["dominante"]),
]


class ControlCartografico(MacroElement):
        """Añade escala métrica y una flecha de norte al mapa Leaflet."""

        _template = Template("""
                {% macro script(this, kwargs) %}
                        L.control.scale({imperial: false, metric: true, maxWidth: 150}).addTo({{this._parent.get_name()}});
                    var controlNorte = L.control({position: 'topright'});
                    controlNorte.onAdd = function() {
                                var control = L.DomUtil.create('div', 'control-norte');
                                control.innerHTML = '<div class="flecha-norte">&#8593;</div><div>N</div>';
                                return control;
                    };
                    controlNorte.addTo({{this._parent.get_name()}});
                {% endmacro %}
        """)


def _agregar_estilos_mapa(mapa):
        estilos = """
        <style>
            .control-norte { background: rgba(255,255,255,.88); color: #17202a;
                padding: 5px 8px; text-align: center; font-weight: 700; border-radius: 3px;
                box-shadow: 0 1px 5px rgba(0,0,0,.35); }
            .flecha-norte { font-size: 24px; line-height: 20px; }
            .leaflet-control-scale-line { background: rgba(255,255,255,.82); color: #17202a;
                border-color: #17202a; }
            @media (prefers-color-scheme: dark) {
                .control-norte, .leaflet-control-scale-line { background: rgba(35,40,45,.9);
                    color: #f4f6f7; border-color: #f4f6f7; }
            }
        </style>
        """
        mapa.get_root().html.add_child(Element(estilos))
        mapa.add_child(ControlCartografico())


def main():
    st.title("☀️ Sitios óptimos para plantas solares — Norte de Chile")
    st.caption("Antofagasta y Atacama · Random Forest + AHP (dónde es apto) × solarpv-rs "
               "(cuánto produce). CRS EPSG:32719.")

    manifest = _cargar_json("manifest.json")
    stats = _cargar_json("stats.json")

    if not manifest:
        st.error("No hay assets. Genera con: python scripts/generate_web_assets.py --config config.yaml")
        return

    # --- Controles ---
    with st.sidebar:
        st.header("Capas")
        capas_activas = []
        for titulo, ids in GRUPOS:
            presentes = [cid for cid in ids if cid in manifest]
            if not presentes:
                continue
            st.caption(titulo)
            for cid in presentes:
                if st.checkbox(ETIQUETAS.get(cid, cid), value=(cid == "balanceado")):
                    capas_activas.append(cid)
        opacidad = st.slider("Opacidad", 0.0, 1.0, 0.75, 0.05)
        st.divider()
        st.caption("Fuente: proyecto Geoinformática USACH · datos: DEM SRTM, Explorador "
                   "Solar, infraestructura eléctrica (Antofagasta/Atacama).")

    # --- Mapa ---
    col_mapa, col_info = st.columns([3, 1])
    with col_mapa:
        # Centro a partir de los bounds de cualquier capa.
        (s, w), (n, e) = list(manifest.values())[0]["bounds"]
        m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=6,
                       tiles="CartoDB positron")
        _agregar_estilos_mapa(m)
        for cid in capas_activas:
            capa = manifest[cid]
            folium.raster_layers.ImageOverlay(
                image=_data_uri(capa["png"]),
                bounds=capa["bounds"],
                opacity=opacidad,
                name=ETIQUETAS.get(cid, cid),
                interactive=False, cross_origin=False, zindex=1,
            ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=None, height=620, returned_objects=[])
        st.markdown(
            '<div class="pie-cartografico">Sistema de referencia: EPSG:32719 (WGS 84 / UTM 19S) · '
            'Norte arriba · Escala métrica dinámica</div>',
            unsafe_allow_html=True,
        )

    with col_info:
        st.subheader("Leyenda")
        for cid in capas_activas:
            unidad = manifest[cid].get("unidad", "")
            vmin, vmax = manifest[cid].get("vmin"), manifest[cid].get("vmax")
            rango = f"Rango: {vmin:g}–{vmax:g}" if vmin is not None and vmax is not None else ""
            st.markdown(f"**{ETIQUETAS.get(cid, cid)}**  \n{unidad}  \n{rango}")
            st.image(_ruta_asset(manifest[cid]["colorbar"]))
            categorias = manifest[cid].get("categorias") or []
            if categorias:
                st.markdown(" · ".join(categorias))

    # --- Estadísticas (T2/T3/T6) ---
    st.divider()
    st.header("Resultados")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Cruce aptitud × rendimiento (T2)")
        cr = stats.get("cruce", {})
        if cr:
            st.metric("Celdas aptas", f"{cr.get('celdas_aptas', 0):,}",
                      f"{cr.get('pct_aptas', 0)}% de la región")
            r, a = cr.get("rendimiento_region", {}), cr.get("rendimiento_aptas", {})
            st.write(f"Rendimiento medio región: **{r.get('media','–')}** kWh/kWp/año")
            st.write(f"En zonas aptas: **{a.get('media','–')}** "
                     f"(ganancia {cr.get('ganancia_aptas_pct','–')}%)")
            st.caption("Aptitud y rendimiento casi desacoplados: el modelo decide por "
                       "logística, no por energía.")

    with c2:
        st.subheader("Fijo vs. seguidor (T3)")
        cm = stats.get("comparacion_montaje", {})
        if cm:
            f_, s_ = cm.get("rendimiento_fijo_aptas", {}), cm.get("rendimiento_seguidor_aptas", {})
            st.metric("Ganancia del seguidor",
                      f"{cm.get('ganancia_seguidor_media_pct','–')}%",
                      f"ref. autor +{cm.get('referencia_autor_pct','–')}%")
            st.write(f"Fijo: **{f_.get('media','–')}** · Seguidor: **{s_.get('media','–')}** kWh/kWp/año")
            if "n_bloques_espaciales" in cm:
                st.caption(f"Mediana por bloque espacial ({cm['n_bloques_espaciales']} bloques "
                           f"de {cm.get('tamano_bloque_km','–')} km, independientes de la "
                           f"autocorrelación entre píxeles vecinos): "
                           f"{cm.get('ganancia_seguidor_mediana_bloque_pct','–')}%")

    with c3:
        st.subheader("Explicabilidad SHAP (T6)")
        sh = stats.get("shap", {})
        imps = sh.get("importancias_shap", [])
        if imps:
            for it in imps[:4]:
                st.write(f"{it['feature']}: **{it['importancia_pct']}%**")
            cruce = sh.get("cruce_rendimiento", {})
            if "spearman_aptitud_vs_rendimiento" in cruce:
                st.caption(f"Correlación aptitud–rendimiento (muestra completa): "
                           f"{cruce['spearman_aptitud_vs_rendimiento']}")
                c_apt = cruce.get("cruce_solo_sitios_aptos", {})
                if "spearman_aptitud_vs_rendimiento" in c_apt:
                    st.caption(f"Solo sitios con prob ≥ {cruce.get('prob_min_apto', '–')}: "
                               f"{c_apt['spearman_aptitud_vs_rendimiento']} (≈0 → desacople)")

    # --- Consenso entre perfiles (Brecha 6) ---
    co = stats.get("consenso", {})
    if co:
        st.subheader("Consenso vs. divergencia entre perfiles (Brecha 6)")
        k1, k2, k3 = st.columns(3)
        k1.metric("Consenso (3 perfiles)", f"{co['consenso_3_perfiles']['pct']}%",
                  "aptas en los 3 → robustas")
        k2.metric("Divergencia (2 perfiles)", f"{co['divergencia_2_perfiles']['pct']}%")
        k3.metric("Divergencia (1 perfil)", f"{co['divergencia_1_perfil']['pct']}%")
        st.caption(f"Definición: 'apto' = top {100 - co['percentil_apto']}% de cada perfil "
                   f"(misma proporción de área en los 3, por construcción). Solo una fracción "
                   "es consenso: dónde conviene depende del criterio priorizado (conservador "
                   "vs. agresivo).")
        corr = co.get("correlacion_rango_entre_perfiles", {})
        if corr:
            st.caption(f"Correlación de rango (Spearman) entre perfiles: "
                       f"conservador–balanceado={corr['conservador_vs_balanceado']} · "
                       f"conservador–agresivo={corr['conservador_vs_agresivo']} · "
                       f"balanceado–agresivo={corr['balanceado_vs_agresivo']}")

    # --- Explicabilidad espacial SHAP (Brecha 8) ---
    se = stats.get("shap_espacial", {})
    dom = se.get("pixeles_por_variable_dominante", {})
    if dom:
        st.subheader("Variable dominante por píxel — SHAP espacial (Brecha 8)")
        total = sum(dom.values()) or 1
        top = sorted(dom.items(), key=lambda x: -x[1])[:3]
        cols = st.columns(len(top))
        for col, (feat, n) in zip(cols, top):
            col.metric(feat, f"{100 * n / total:.1f}%", f"{n:,} px")
        st.caption("Variable que más empuja la aptitud en cada píxel. La cercanía a "
                   "infraestructura domina casi todo el territorio; el GHI casi nunca — "
                   "explicación auditable pixel a pixel (argumento SEIA).")


# Streamlit ejecuta este script en cada interacción; llamamos a main() directamente.
main()
