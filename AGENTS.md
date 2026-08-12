# AGENTS.md 
Proyecto: **Sitios óptimos para plantas solares fotovoltaicas en el norte de Chile** (Geoinformática). Modelo Random Forest + AHP

---

## 1. Idioma y estilo

- **Todo en español**: código nuevo, nombres de variables/funciones, comentarios, mensajes de `print`, docstrings, mensajes de commit y descripciones de PR.
- Sigue el estilo del código existente: nombres descriptivos en español (`generar_dataset_muestras`, `calcular_distancia_a_capa`), comentarios que explican el *porqué* de una decisión, no el *qué*.
- No introduzcas dependencias nuevas sin justificarlo. El stack es numpy/pandas/geopandas/shapely/rasterio/scipy/scikit-learn/optuna/matplotlib/surtgis.

## 2. Alcance del dominio

- **Solo las regiones de Antofagasta y Atacama.** Cualquier filtro geográfico debe respetar esto (ver `region_mask` en `scripts/generate_suitability_map.py`).
- **CRS objetivo del proyecto: EPSG:32719 (UTM 19S).** Todo raster/vector procesado se reproyecta a este CRS. `solarpv-rs` emite en el mismo CRS: los cruces son celda a celda.
- Los datos crudos viven en `data/`.

## 3. Invariantes que NO se pueden romper

1. **Orden canónico de features** — `src/features.py` (`FEATURES`) es la fuente de verdad. Su orden debe coincidir exactamente con:
   - el `column_stack` de `X_pred` en `scripts/generate_suitability_map.py`(hay un `assert` que falla en voz alta si se desalinea), y
   - las columnas muestreadas en `src/sampling.py`.
   Si agregas/reordenas una feature, actualiza los tres lugares en el mismo commit.
2. **Truncado ESRI a 10 caracteres** — los shapefiles truncan nombres de columna. El código restaura los nombres con un `rename_map` (`dist_trans`→`dist_transmision`, `dist_almac`→`dist_almacen`, `dist_subs`→`dist_subestaciones`).
   Respeta este patrón; no rompas la lectura de shapefiles existentes.
3. **`random_state = 42`** se propaga por todo el pipeline (muestreo, split, RF, Optuna).
   No lo cambies: la reproducibilidad de las métricas depende de él.
4. **Pipeline idempotente** — `src/utils.py::_esta_actualizado` implementa chequeo incremental por timestamps. Cada etapa nueva debe declarar sus entradas/salidas y omitirse si ya está actualizada, igual que las etapas 1–8 de `run_pipeline.py`.
5. **Matrices AHP de `config.yaml`** (`ahp_perfiles`) están validadas con CR<0.10 y sus valores coinciden con el informe de PEP1 (CR=0.0012 / 0.0205). No las modifiques sin recalcular y verificar el Ratio de Consistencia.

## 4. Cómo correr y verificar

- **Pipeline completo:** `python scripts/run_pipeline.py --config config.yaml`
- **Iterar solo en el modelo (sin mapas/figuras):** agrega `--sin-mapas`.
- **Tests:** `python -m unittest discover -s tests` (framework: `unittest`).
- **Validación PostGIS sin base de datos activa:** `python scripts/validate_and_load_postgis.py --dry-run`
- Python **3.10+**. No asumas librerías fuera de `requirements.txt`.

**Antes de dar por terminada una tarea** debes: correr los tests, si tocaste el pipeline, correrlo con `--sin-mapas` como mínimo, y reportar qué corriste y su resultado. No declares "funciona" sin evidencia de ejecución.

## 5. Qué NO tocar sin permiso explícito

- Los artefactos committeados en `data/results/` (`model_rf.pkl`, shapefiles de entrenamiento, JSONs de métricas).
- Las matrices AHP y los umbrales de `config.yaml` (`criterios`, `postgis_validation`) salvo que la tarea lo pida y lo justifique.
- El orden de `FEATURES` (ver invariante 1).
- Credenciales/`connection` de PostGIS en `config.yaml`.

## 6. Integración con `solarpv-rs`

- Repo del motor: `https://github.com/franciscoparrao/solarpv-rs`. Requiere clonar además `https://github.com/franciscoparrao/surtgis` como repo hermano (dependencia de path del paso terrain). Se compila con `cargo build --release -p solarpv-cli` (el feature `terrain` ya viene activo por defecto en el CLI; NO uses `--features terrain`, no existe en ese paquete). El binario resultante es `solarpv-rs/target/release/solarpv` (el `[[bin]]` se llama `solarpv`, no `solarpv-cli`).
- El motor produce `specific_yield.tif` (kWh/kWp/año por celda) en **EPSG:32719** sobre el mismo DEM: es directamente comparable, celda a celda, con `data/results/mapa_probabilidad_aptitud.tif`.
- Regla de integración: el pipeline Python invoca el binario Rust (subprocess) y consume sus `.tif`; no reimplementes la física PV en Python. Valida siempre que la salida del motor esté en EPSG:32719 y en la extensión de Antofagasta/Atacama antes de cruzarla.
- Fija la versión del motor con `Cargo.lock` para reproducibilidad.

## 7. Tareas

- **Explicabilidad**: prefiere código legible sobre código "listo".
- **No subir nada al repositorio Remoto**: todos los cambios sobre el repositorio Git y GitHub son realizados manualmente.++

## 8. Datos geoespaciales — cuidados

- Verifica CRS antes de cualquier operación entre capas (reproyecta a EPSG:32719).
- `aspect` es circular (0°/360°): resamplea con `Resampling.nearest`, nunca bilinear.
- Distancias euclidianas se calculan sobre la grilla del DEM (~fina), no del GHI (~1 km).
- No inventes rutas de datos: usa siempre las de `config.yaml`.
