# GeoInformatica-Sitios-ptimos-para-Energ-a-Solar-en-el-Norte-de-Chile
Problema. Identificar sitios óptimos para nuevas plantas fotovoltaicas en una región del norte de Chile mediante análisis multicriterio GIS, considerando irradiación, topografía, accesibilidad y restricciones.

git clone https://github.com/TinyAntu/GeoInformatica-Sitios-ptimos-para-Energ-a-Solar-en-el-Norte-de-Chile.git

pip install -r requirements.txt

python scripts/run pipeline.py --config config.yaml.

Para descargar los datos por favor acceda a: https://drive.google.com/drive/folders/1ujkfnfOVDNQYJluZw06fFZAsKIZ-7T0z?usp=sharing

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

**Deploy en la nube (Streamlit Community Cloud), sin depender de localhost:**
1. Los assets de `app/assets/` (pocos MB) se versionan en el repo; los `.tif` pesados no
   hacen falta en la nube.
2. En share.streamlit.io: repo del proyecto, archivo principal `app/visor.py`.
3. Usa `app/requirements.txt` (mínimo: streamlit/folium/streamlit-folium) para un build
   liviano — el visor no necesita las librerías geoespaciales del pipeline.
