# Sitios Óptimos para Plantas Solares Fotovoltaicas en el Norte de Chile

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Rust Motor](https://img.shields.io/badge/Rust-solarpv--rs-orange.svg)
![CRS](https://img.shields.io/badge/CRS-EPSG%3A32719-green.svg)
![Pipeline](https://img.shields.io/badge/Pipeline-Idempotente-brightgreen.svg)


**Problema:** Identificar sitios óptimos para nuevas plantas fotovoltaicas en las regiones de Antofagasta y Atacama (Norte de Chile) mediante análisis multicriterio GIS y aprendizaje automático (Random Forest + AHP), integrando físicas reales de producción fotovoltaica y explicabilidad espacial.

```bash
git clone https://github.com/TinyAntu/GeoInformatica-Sitios-ptimos-para-Energ-a-Solar-en-el-Norte-de-Chile.git
cd GeoInformatica-Sitios-ptimos-para-Energ-a-Solar-en-el-Norte-de-Chile
pip install -r requirements.txt

# Ejecución base (Antofagasta y Atacama):
python scripts/run_pipeline.py --config config.yaml

# Evaluación de transferencia (incluyendo Coquimbo):
python scripts/run_pipeline.py --config config.yaml --con-transferencia
```

> **Descarga de Datos Crudos:** Accede a la carpeta oficial en Google Drive: [Descargar datos del proyecto](https://drive.google.com/drive/folders/1ujkfnfOVDNQYJluZw06fFZAsKIZ-7T0z?usp=sharing).

---

## Reproducibilidad y Arquitectura

El pipeline es **idempotente** (chequeo incremental de frescura por marcas de tiempo en cada etapa) y propaga de forma estricta `random_state=42` para garantizar la reproducibilidad de todas las métricas.

Las 18 etapas están organizadas en **6 fases**, siguiendo el flujo canónico de un pipeline
geoespacial (datos crudos → limpieza → transformación → análisis → visualización):

| Fase | Nombre | Qué produce |
|---|---|---|
| 1 | Preprocesamiento | `data/processed/`: DEM, slope, aspect y GHI en EPSG:32719 |
| 2 | Features y modelamiento | Dataset de muestras, `model_rf.pkl` y validación espacial (SBCV + LOROCV) |
| 3 | Inferencia y superficies | Mapa de aptitud RF, perfiles AHP/WLC y rendimiento físico (motor Rust) |
| 4 | Análisis integrado | Cruce aptitud×rendimiento, consenso, SHAP global/espacial y métricas Top-K |
| 5 | Transferibilidad | Evaluación sobre Coquimbo sin reentrenar (opt-in) |
| 6 | Persistencia y difusión | PostGIS, figuras del informe y assets del visor |

Cada etapa corre en **su propio proceso**, de modo que su memoria se devuelve al sistema al
terminar. Si una corrida se corta por falta de memoria, `--desde-fase N` la retoma sin rehacer
lo anterior.

1. **Datos:** Descargar datos desde Google Drive y colocarlos en la carpeta `data/` respetando la estructura declarada en `config.yaml`.
2. **Entorno Python:** `pip install -r requirements.txt`. *(Nota en Windows: si `rasterio` o `pyproj` no detectan automáticamente `proj.db`, define `$env:PROJ_LIB=".../rasterio/proj_data"`)*.
3. **Pipeline Base (Etapas 1–18):** `python scripts/run_pipeline.py --config config.yaml`.
4. **Motor Rust (`solarpv-rs`):** Opcional pero recomendado para simulación física de rendimiento (`kWh/kWp/año`). Commits exactos:
   - `solarpv-rs` @ `41cdaaf` — [https://github.com/franciscoparrao/solarpv-rs](https://github.com/franciscoparrao/solarpv-rs)
   - `surtgis` @ `v1.2.2` (`4c60871`) — [https://github.com/franciscoparrao/surtgis](https://github.com/franciscoparrao/surtgis)
5. **Pruebas Unitarias:** `python -m unittest discover -s tests`.

---

## Tabla de Comandos del Proyecto

| Componente / Etapa | Comando | Descripción |
|---|---|---|
| **Pipeline base completo** | `python scripts/run_pipeline.py --config config.yaml` | Ejecuta las 6 fases / 18 etapas (aptitud RF, AHP, cruce, consenso, SHAP y assets) |
| **Pipeline (solo modelo)** | `python scripts/run_pipeline.py --config config.yaml --sin-mapas` | Corre solo hasta la fase 2; omite rasters pesados para iteración rápida en ML |
| **Ver el plan de fases** | `python scripts/run_pipeline.py --listar-fases` | Imprime las 6 fases con sus etapas, sin ejecutar nada |
| **Simulación** | `python scripts/run_pipeline.py --dry-run` | Muestra qué etapas correrían y cuáles están al día |
| **Reanudar desde una fase** | `python scripts/run_pipeline.py --desde-fase 3` | Retoma tras un fallo o un corte por memoria |
| **Correr una sola fase** | `python scripts/run_pipeline.py --solo-fase 4` | Ejecuta únicamente esa fase |
| **Preprocesamiento (fase 1)** | `python scripts/run_preprocesamiento.py --config config.yaml` | DEM, slope/aspect y reproyección del GHI a EPSG:32719 |
| **Entrenamiento (etapa 3)** | `python scripts/run_entrenamiento.py --config config.yaml` | Muestreo espacial y Random Forest con Optuna |
| **Rendimiento Fijo (23°)** | `python scripts/run_solar_yield.py --mount tilt [--tilt 23]` | Genera mapa de rendimiento fijo a inclinación óptima por latitud (`~23°`) |
| **Rendimiento Plano (0°)** | `python scripts/run_solar_yield.py --mount tilt --tilt 0` | Genera mapa de rendimiento fijo en superficie horizontal (`tilt = 0°`) |
| **Rendimiento Seguidor** | `python scripts/run_solar_yield.py --mount tracker` | Genera mapa de rendimiento para seguidor de un eje (`gcr = 0.3`) |
| **Cruce Aptitud × Rendimiento** | `python scripts/run_cruce.py --config config.yaml` | Cruza celda a celda la aptitud probabilística con la producción física |
| **Comparación de Montaje (0°, 23°, Tracker)** | `python scripts/run_comparacion_montaje.py [--incluir-tilt0] [--regenerar]` | Cuantifica ganancias entre horizontal (`0°`), latitud (`23°`) y seguidor (`tracker`) |
| **Consenso de Perfiles** | `python scripts/run_consenso.py --config config.yaml` | Evalúa consenso vs. divergencia entre perfiles Conservador, Agresivo y RF |
| **Explicabilidad SHAP Global** | `python scripts/run_shap.py --config config.yaml` | Importancia global de características por autovector de Shapley |
| **Explicabilidad SHAP Espacial** | `python scripts/run_shap_spatial.py --config config.yaml` | Grilla espacial de variable dominante por píxel |
| **Métricas Top-K (PEP1)** | `python scripts/run_metricas_topk.py --config config.yaml` | Métricas Recall@K y Precision@K vs. umbrales del informe |
| **Assets del Visor** | `python scripts/generate_web_assets.py --config config.yaml` | Prepara imágenes y JSON livianos para el visor web |
| **Visor Web Interactivo** | `streamlit run app/visor.py` | Lanza la aplicación interactiva de exploración local |

---

## Motor de Rendimiento Fotovoltaico (`solarpv-rs`)

Mientras que la aptitud (RF + AHP) responde **dónde** es conveniente instalar, el motor Rust `solarpv-rs` calcula **cuánto produce** cada sitio en unidades energéticas reales (`kWh/kWp/año`), considerando sombreado por relieve (DEM) y trayectoria solar celda a celda en `EPSG:32719`.

### Configuración y Compilación
Clonar ambos repositorios como hermanos en la raíz del proyecto y compilar el binario Rust:

```bash
git clone https://github.com/franciscoparrao/solarpv-rs.git
git clone https://github.com/franciscoparrao/surtgis.git
cd solarpv-rs
cargo build --release -p solarpv-cli
```

### Opciones de Montaje y Tilt
- **Inclinación Óptima a Latitud (`tilt = 23°`):** Inclinación fija estándar para Antofagasta/Atacama (`data/results/rendimiento_fijo_specific_yield.tif`).
- **Montaje Plano Horizontal (`tilt = 0°`):** Módulos sin inclinación horizontal (`data/results/rendimiento_fijo_tilt0_specific_yield.tif`).
- **Seguidor Solar de Un Eje (`tracker`):** Seguidor eje norte-sur con backtracking (`data/results/rendimiento_seguidor_specific_yield.tif`).

Para evaluar y comparar las 3 tecnologías sobre las zonas aptas:
```bash
python scripts/run_comparacion_montaje.py --incluir-tilt0
```

---

## Visor Web Interactivo (Streamlit)

La aplicación web permite inspeccionar interactivamente las capas de aptitud, rendimiento físico y explicabilidad SHAP superpuestas en un mapa basemap.

```bash
python scripts/generate_web_assets.py --config config.yaml   # Genera app/assets/
streamlit run app/visor.py                                    # Inicia visor local
```

### Despliegue en Streamlit Community Cloud
1. Los assets en `app/assets/` están versionados en el repositorio (no se requieren los `.tif` pesados en la nube).
2. Configurar en [share.streamlit.io](https://share.streamlit.io) apuntando a `app/visor.py`.
3. El despliegue usa `app/requirements.txt` para mantener una construcción ligera de paquetes.

