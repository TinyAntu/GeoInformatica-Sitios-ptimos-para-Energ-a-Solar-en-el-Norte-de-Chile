# GeoInformatica-Sitios-ptimos-para-Energ-a-Solar-en-el-Norte-de-Chile
Problema. Identificar sitios óptimos para nuevas plantas fotovoltaicas en una región del norte de Chile mediante análisis multicriterio GIS, considerando irradiación, topografía, accesibilidad y restricciones.

```
git clone https://github.com/TinyAntu/GeoInformatica-Sitios-ptimos-para-Energ-a-Solar-en-el-Norte-de-Chile.git
pip install -r requirements.txt
python scripts/run_pipeline.py --config config.yaml
```

Para descargar los datos por favor acceda a: https://drive.google.com/drive/folders/1ujkfnfOVDNQYJluZw06fFZAsKIZ-7T0z?usp=sharing

## Reproducibilidad

El pipeline es idempotente (chequeo incremental por etapa) y usa `random_state=42`. Para
reproducir de cero:

1. **Datos**: descargar desde el Drive de arriba y dejarlos en `data/` (ver rutas en
   `config.yaml`; `data/` está en `.gitignore`).
2. **Entorno Python**: `pip install -r requirements.txt` (versiones fijadas).
3. **Pipeline base** (etapas 1–10): `python scripts/run_pipeline.py --config config.yaml`.
   Incluye la Etapa 9 (cruce con rendimiento, se omite sola si aún no generaste el mapa
   del motor) y la Etapa 10 (consenso entre perfiles).
4. **Motor Rust** (opcional, para rendimiento): ver sección siguiente. Commits exactos
   usados para estos resultados:
   - `solarpv-rs` @ `41cdaaf` — https://github.com/franciscoparrao/solarpv-rs
   - `surtgis` @ `v1.2.2` (`4c60871`) — https://github.com/franciscoparrao/surtgis
5. **Tests**: `python -m unittest discover -s tests`.

Orden recomendado end-to-end (con motor): pipeline base → comparación de montaje →
cruce → SHAP global → SHAP espacial → assets → visor. Los pasos de SHAP deben correr antes de generar los assets (el visor consume sus salidas).

| Componente | Comando |
|---|---|
| Pipeline base (aptitud, validación, cruce, consenso) | `python scripts/run_pipeline.py --config config.yaml` |
| Rendimiento PV fijo / seguidor             | `python scripts/run_solar_yield.py [--mount tracker]` |
| Cruce aptitud × rendimiento (= Etapa 9)    | `python scripts/run_cruce.py --config config.yaml` |
| Comparación fijo vs. seguidor              | `python scripts/run_comparacion_montaje.py [--regenerar]` |
| Consenso entre perfiles (= Etapa 10)       | `python scripts/run_consenso.py --config config.yaml` |
| Explicabilidad SHAP global                 | `python scripts/run_shap.py --config config.yaml` |
| Explicabilidad SHAP espacial (Brecha 8)    | `python scripts/run_shap_spatial.py --config config.yaml` |
| Recall@K / Precisión@K vs. umbrales PEP1   | `python scripts/run_metricas_topk.py --config config.yaml` |
| Assets del visor                           | `python scripts/generate_web_assets.py --config config.yaml` |
| Visor web                                  | `streamlit run app/visor.py` |

**Resoluciones configurables** (cuidado de editar la clave correcta):
`solarpv.resolucion_m` (motor de rendimiento, 300 m; `null` = DEM nativo ~90 m, ~1 h por
montaje considerando 12 núcleos — requiere `--regenerar`), `shap_espacial.resolucion_m` (500 m),
`salida_mapa.resolucion_m` (mapa RF, 100 m), `preprocesamiento.dem_resolucion_m` (DEM base).

## Motor de rendimiento fotovoltaico (solarpv-rs)

La aptitud (RF + AHP) responde *dónde* es apto instalar; el motor Rust `solarpv-rs`
agrega *cuánto produce* cada sitio (kWh/kWp/año). Se compila aparte y no se versiona en
este repo (está en `.gitignore`). El paso en grilla del motor (terrain) reutiliza la
librería `surtgis`, así que hay que clonar **ambos** repos como hermanos, y necesitas
tener Rust/cargo instalado (`https://rustup.rs`):

```
# Ambos repos deben quedar en la raíz del proyecto (surtgis es dependencia de path de solarpv-rs)
git clone https://github.com/franciscoparrao/solarpv-rs.git
git clone https://github.com/franciscoparrao/surtgis.git
cd solarpv-rs
cargo build --release -p solarpv-cli   # el feature 'terrain' ya viene activo por defecto en el CLI
cd ..
```

Esto genera el binario `solarpv-rs/target/release/solarpv` (nombre del `[[bin]]`, no
`solarpv-cli`). Ajusta `solarpv.binario` en `config.yaml` si queda en otra ruta. Luego
genera el mapa de rendimiento:

```
python scripts/run_solar_yield.py --config config.yaml            # montaje fijo
python scripts/run_solar_yield.py --mount tracker                 # seguidor de un eje
python scripts/run_solar_yield.py --dry-run                       # solo imprime el comando
```

Salida: `data/results/rendimiento_fijo_specific_yield.tif` (EPSG:32719), directamente
comparable celda a celda con `data/results/mapa_probabilidad_aptitud.tif`.

## Visor interactivo (demo)

Mapa web con las capas de aptitud, rendimiento y su cruce superpuestas sobre un basemap.
Los assets livianos (PNG + JSON) se generan a partir de los rasters de resultados:

```
python scripts/generate_web_assets.py --config config.yaml   # crea app/assets/
streamlit run app/visor.py                                    # visor local
```

**Deploy en la nube (Streamlit Community Cloud)**
1. Los assets de `app/assets/` (pocos MB) se versionan en el repo; los `.tif` pesados no
   hacen falta en entorno cloud.
2. En share.streamlit.io: repo del proyecto, archivo principal `app/visor.py`.
3. Usa `app/requirements.txt` (mínimo: streamlit/folium/streamlit-folium) para un build
   liviano — el visor no necesita las librerías geoespaciales del pipeline.
