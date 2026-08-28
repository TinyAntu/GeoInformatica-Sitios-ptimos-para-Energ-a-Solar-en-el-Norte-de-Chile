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


def _selector_capas(manifest, prefijo, por_defecto="balanceado"):
    """Checkbox de capas + opacidad. `prefijo` da claves únicas por pestaña.

    Los controles viven dentro de cada pestaña y no en el sidebar: Streamlit renderiza los
    widgets de TODAS las pestañas en cada pasada, así que un sidebar compartido mostraría dos
    juegos de checkbox con las mismas etiquetas (y `DuplicateWidgetID` si no llevaran clave).
    """
    activas = []
    columnas = st.columns(len(GRUPOS) + 1)
    for col, (titulo, ids) in zip(columnas, GRUPOS):
        presentes = [cid for cid in ids if cid in manifest]
        if not presentes:
            continue
        with col:
            st.caption(titulo)
            for cid in presentes:
                if st.checkbox(ETIQUETAS.get(cid, cid), value=(cid == por_defecto),
                               key=f"{prefijo}_capa_{cid}"):
                    activas.append(cid)
    with columnas[-1]:
        st.caption("Opacidad")
        opacidad = st.slider("Opacidad", 0.0, 1.0, 0.75, 0.05,
                             key=f"{prefijo}_opacidad", label_visibility="collapsed")
    return activas, opacidad


def _render_mapa(manifest, capas_activas, opacidad, key):
    """Dibuja el mapa Leaflet con las capas activas. `key` debe ser único por pestaña."""
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
    # `key` distinto por pestaña: sin él, st_folium reutiliza el estado del primer mapa y el
    # segundo no llega a renderizarse.
    st_folium(m, width=None, height=620, returned_objects=[], key=key)
    st.markdown(
        '<div class="pie-cartografico">Sistema de referencia: EPSG:32719 (WGS 84 / UTM 19S) · '
        'Norte arriba · Escala métrica dinámica</div>',
        unsafe_allow_html=True,
    )


def _render_leyenda(manifest, capas_activas):
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


def _pestana_zona_estudio(manifest, stats):
    capas_activas, opacidad = _selector_capas(manifest, "norte")
    col_mapa, col_info = st.columns([3, 1])
    with col_mapa:
        _render_mapa(manifest, capas_activas, opacidad, key="mapa_norte")
    with col_info:
        _render_leyenda(manifest, capas_activas)

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


def _recall_en(bloque, region, k):
    """Extrae recall@k% de un bloque por región de metricas_topk.json."""
    try:
        return bloque["por_region"][region]["por_k"][k]["recall"]
    except (KeyError, TypeError):
        return None


def _pestana_transferencia(manifest, stats, stats_norte):
    """Generalización del modelo: aplicarlo a una región que no participó del entrenamiento."""
    tr = stats.get("transferibilidad", {})
    regiones = " + ".join(tr.get("regiones", [])) or "otra región"

    st.warning(
        f"**El modelo no se reentrenó.** Se cargó `model_rf.pkl`, entrenado con Antofagasta y "
        f"Atacama, y se aplicó tal cual sobre {regiones}. Lo que se mide aquí es si la "
        "capacidad predictiva sobrevive fuera del territorio donde se ajustó.",
        icon="🧪",
    )

    # --- 1. El gradiente A -> B -> C: el resultado del ejercicio ---
    metricas = tr.get("metricas", {})
    por_k = metricas.get("por_k", {})
    topk = stats_norte.get("metricas_topk", {})
    k = "1"
    a = _recall_en(topk.get("in_sample_por_region", {}), "Atacama", k)
    b = _recall_en(topk.get("out_of_sample_loro", {}), "Atacama", k)
    c = (por_k.get(k) or {}).get("recall")

    st.subheader(f"Generalización: recall@{k} % en tres condiciones")
    g1, g2, g3 = st.columns(3)
    if a is not None:
        g1.metric("A · Atacama in-sample", f"{100 * a:.1f}%", "cota optimista",
                  delta_color="off")
    if b is not None:
        g2.metric("B · Atacama hold-out (LORO)", f"{100 * b:.1f}%",
                  f"{100 * (b - a):+.1f} pp vs A" if a is not None else None)
    if c is not None:
        g3.metric(f"C · {regiones} transferido", f"{100 * c:.1f}%",
                  f"{100 * (c - b):+.1f} pp vs B" if b is not None else None)
    st.caption(
        "A → B es la caída esperable por validación espacial: el modelo evalúa una región que "
        "no entrenó. B → C es de otra magnitud, y es el hallazgo: fuera del dominio de "
        "entrenamiento el desempeño se desploma."
    )

    # --- 2. Mapa ---
    st.divider()
    capas_activas, opacidad = _selector_capas(manifest, "transf")
    col_mapa, col_info = st.columns([3, 1])
    with col_mapa:
        _render_mapa(manifest, capas_activas, opacidad, key="mapa_transferencia")
    with col_info:
        _render_leyenda(manifest, capas_activas)

    # --- 3. Por qué: desplazamiento de covariables ---
    shift = tr.get("covariate_shift", {})
    if shift:
        st.divider()
        st.subheader("Por qué falla: desplazamiento de covariables")
        st.caption(
            "Un Random Forest **no extrapola**: fuera del rango de valores que vio al "
            "entrenar devuelve el valor de la hoja más cercana. No lanza error, simplemente "
            "satura. Esta tabla mide cuánto territorio cae fuera de ese rango."
        )
        filas = sorted(shift.items(), key=lambda kv: -kv[1]["pct_fuera_de_rango"])
        st.dataframe(
            [{
                "Variable": f_,
                "% fuera del rango de entrenamiento": d["pct_fuera_de_rango"],
                "Desplazamiento (sd)": d["desplazamiento_medias_sd"],
                "Rango entrenamiento": f"{d['entrenamiento']['min']:g} – {d['entrenamiento']['max']:g}",
                "Rango zona": f"{d['zona']['min']:g} – {d['zona']['max']:g}",
            } for f_, d in filas],
            hide_index=True, width='stretch',
        )
        peor = filas[0]
        st.caption(
            f"La variable más desplazada es **{peor[0]}**: "
            f"{peor[1]['pct_fuera_de_rango']:.1f} % de los píxeles quedan fuera del rango de "
            f"entrenamiento ({peor[1]['desplazamiento_medias_sd']} desviaciones estándar de "
            "diferencia entre medias)."
        )

    # --- 4. Lo que sí transfiere: la estructura de la decisión ---
    dom_z = (stats.get("shap_espacial", {}) or {}).get("pixeles_por_variable_dominante", {})
    dom_n = (stats_norte.get("shap_espacial", {}) or {}).get("pixeles_por_variable_dominante", {})
    if dom_z and dom_n:
        st.divider()
        st.subheader("Lo que sí transfiere: la estructura de la decisión")
        tot_z, tot_n = sum(dom_z.values()) or 1, sum(dom_n.values()) or 1
        st.dataframe(
            [{
                "Variable dominante": f_,
                "Zona de estudio (%)": round(100 * dom_n.get(f_, 0) / tot_n, 2),
                f"{regiones} (%)": round(100 * dom_z.get(f_, 0) / tot_z, 2),
            } for f_ in sorted(dom_n, key=lambda x: -dom_n[x])],
            hide_index=True, width='stretch',
        )
        st.caption(
            "El desempeño cae, pero el *criterio* es el mismo: sigue mandando la conectividad "
            "eléctrica y la irradiancia sigue siendo casi irrelevante. El modelo no se "
            "desorienta, se queda sin rango donde discriminar."
        )

    # --- 5. Consenso y límite conocido del dato ---
    co = stats.get("consenso", {})
    co_n = stats_norte.get("consenso", {})
    if co and co_n:
        st.divider()
        st.subheader("Consenso entre perfiles")
        k1, k2, k3 = st.columns(3)
        k1.metric("Consenso (3 perfiles)", f"{co['consenso_3_perfiles']['pct']}%",
                  f"{co['consenso_3_perfiles']['pct'] - co_n['consenso_3_perfiles']['pct']:+.2f} pp "
                  "vs zona de estudio")
        k2.metric("Divergencia (2 perfiles)", f"{co['divergencia_2_perfiles']['pct']}%")
        k3.metric("Divergencia (1 perfil)", f"{co['divergencia_1_perfil']['pct']}%")

    sens = tr.get("sensibilidad_filtro_lineas", {})
    if sens:
        s1 = (sens.get("metricas", {}).get("por_k", {}) or {}).get(k, {})
        p1 = por_k.get(k, {})
        if s1 and p1:
            st.info(
                "**Límite conocido del dato.** La capa de transmisión codifica las líneas que "
                "cruzan fronteras regionales como cadenas compuestas "
                "(`'ATACAMA;COQUIMBO'`), y el filtro por región —el mismo que usó el "
                f"entrenamiento— las descarta. Rescatándolas, el recall@{k} % sube de "
                f"{100 * p1['recall']:.1f} % a {100 * s1['recall']:.1f} % y la precisión de "
                f"{100 * p1['precision_area']['precision']:.3f} % a "
                f"{100 * s1['precision_area']['precision']:.3f} %. Parte de la caída es "
                "artefacto de datos, no geografía; el resultado principal usa el filtro "
                "original por consistencia con el modelo entrenado.",
                icon="⚠️",
            )


def main():
    st.title("☀️ Sitios óptimos para plantas solares — Norte de Chile")
    st.caption("Antofagasta y Atacama · Random Forest + AHP (dónde es apto) × solarpv-rs "
               "(cuánto produce) · más una prueba de generalización sobre una región no "
               "entrenada. CRS EPSG:32719.")

    manifest = _cargar_json("manifest.json")
    stats = _cargar_json("stats.json")

    if not manifest:
        st.error("No hay assets. Genera con: python scripts/generate_web_assets.py --config config.yaml")
        return

    manifest_tr = _cargar_json("manifest_transferibilidad.json")
    stats_tr = _cargar_json("stats_transferibilidad.json")

    with st.sidebar:
        st.header("Acerca de")
        st.caption("Los controles de capas están dentro de cada pestaña, porque cada una "
                   "tiene su propio mapa.")
        st.divider()
        st.caption("Fuente: proyecto Geoinformática USACH · datos: DEM SRTM (NASA), "
                   "Explorador Solar (Min. Energía), infraestructura eléctrica.")

    if manifest_tr:
        tab_norte, tab_transf = st.tabs(
            ["Zona de estudio (Antofagasta + Atacama)", "🧪 Transferencia a otra región"])
        with tab_norte:
            _pestana_zona_estudio(manifest, stats)
        with tab_transf:
            _pestana_transferencia(manifest_tr, stats_tr, stats)
    else:
        # Sin assets de transferencia el visor funciona igual que antes, sin pestañas.
        _pestana_zona_estudio(manifest, stats)


# Streamlit ejecuta este script en cada interacción; llamamos a main() directamente.
main()
