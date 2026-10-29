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

### Dónde queda cada resultado

Los resultados y las figuras van en subcarpetas por tema. La misma estructura se repite dentro de `coquimbo/`, y los estudios que corren fuera del pipeline quedan aparte en `analisis/`. La tabla de qué va en cada tema está en `src/rutas.py`, el único lugar del código que lo decide. Las raíces se declaran en `config.yaml > paths.results`.

| Tema | `data/results/<tema>/` | `figures/<tema>/` |
|---|---|---|
| `modelo` | `model_rf.pkl` y sus métricas, dataset de entrenamiento, validación espacial, métricas top-K | importancias, ROC, precisión-recall, folds SBCV, métricas de validación |
| `aptitud` | mapa RF, perfiles AHP (conservador y agresivo), consenso | mapas de aptitud y de consenso |
| `rendimiento` | rendimiento del motor (fijo 23°, fijo 0°, seguidor), comparación de montajes | mapas de rendimiento, fijo vs. seguidor |
| `cruce` | aptitud × rendimiento, rendimiento en aptas | mapa del cruce |
| `explicabilidad` | SHAP global y espacial | resumen y waterfalls SHAP, variable dominante |
| `postgis` | sitios validados, reporte, script SQL | — |
| `transferibilidad` | solo en `coquimbo/`: métricas, AOA, DI | solo en `coquimbo/`: mapas de AOA y DI |

- **`analisis/`:** auditoría de fuga, desacople, comparación de teselado, sensibilidad, AOA LORO y ablación.
- **`historico/`:** respaldos que ya no produce el pipeline, como los rásters de cielo despejado y figuras antiguas.

Para pasar una copia local con la estructura plana anterior: `python scripts/reorganizar_resultados.py --config config.yaml`. Muestra lo que movería; con `--aplicar` mueve, conservando las fechas de los archivos.

1. **Datos:** Descargar datos desde Google Drive y colocarlos en la carpeta `data/` respetando la estructura declarada en `config.yaml`.
2. **Entorno Python:** `pip install -r requirements.txt`. *(Nota en Windows: si `rasterio` o `pyproj` no detectan automáticamente `proj.db`, define `$env:PROJ_LIB=".../rasterio/proj_data"`)*.
3. **Pipeline Base (Etapas 1–18):** `python scripts/run_pipeline.py --config config.yaml`.
4. **Motor Rust (`solarpv-rs`):** Opcional pero recomendado para simulación física de rendimiento (`kWh/kWp/año`). Commits exactos:
   - `solarpv-rs` @ `8f17b9f` (v0.4.0) + `parches/solarpv-rs_tile_ghi_origen.patch` — [https://github.com/franciscoparrao/solarpv-rs](https://github.com/franciscoparrao/solarpv-rs)
   - `surtgis` @ `v1.2.2` (`4c60871`) — [https://github.com/franciscoparrao/surtgis](https://github.com/franciscoparrao/surtgis)
5. **Pruebas Unitarias:** `python -m unittest discover -s tests`.

### Control de fuga en la infraestructura

Las capas del CEN (subestaciones, líneas y almacenamiento) son de 2026 e incluyen infraestructura **dedicada**, construida para las propias plantas. Medir la distancia de una planta a su propia subestación es causalidad inversa ("hay red porque hay planta"), e infla las métricas del modelo. Con `control_fuga.excluir_dedicadas: true` en `config.yaml`, que es la configuración oficial, todo el pipeline usa la red sin infraestructura dedicada. La auditoría que lo justifica está en `data/results/analisis/auditoria_infraestructura/README.md`.

| | Con fuga (original) | Sin fuga |
|---|---|---|
| AUC LORO | 0,967 | ≈ 0,92 |
| Recall LORO en el top 1 % del mapa | 56 % | ≈ 37 % |
| SHAP del acceso a red | 75 % | ≈ 61 % |

Los puntos de entrenamiento no cambian: el criterio de pseudo-ausencias (≤ 20 km de transmisión) se sigue midiendo contra la red completa. `false` reproduce el pipeline original.

### Pendiente: base de datos PostGIS (etapa 7)

La etapa 7 (`scripts/validate_and_load_postgis.py`) valida los sitios aptos y escribe en `data/results/postgis/` la capa `instalaciones_optimas_validadas.shp`, el reporte `reporte_validacion_postgis.json` y el script `cargar_instalaciones_postgis.sql`. Todo eso funciona sin base de datos. Lo que queda **pendiente** es la carga en vivo a PostGIS: no hay una base levantada, así que la etapa (best-effort) deja el `.sql` listo y el pipeline sigue.

Para habilitarla se propone Docker, que calza con `config.yaml` (`postgis/postgis` en `localhost:5433`, base `geoinformatica_db`):

```powershell
docker run -d --name postgis-geoinformatica `
  -e POSTGRES_PASSWORD=<clave> -e POSTGRES_DB=geoinformatica_db `
  -p 5433:5432 -v postgis_datos:/var/lib/postgresql/data postgis/postgis:16-3.4
$env:PGPASSWORD = "<clave>"            # la contraseña no se versiona en config.yaml
python scripts/validate_and_load_postgis.py --config config.yaml
docker exec -it postgis-geoinformatica psql -U postgres -d geoinformatica_db -c "SELECT count(*) FROM public.instalaciones_solares_optimas;"
```

- La conexión usa el driver `psycopg2` explícito (`postgresql+psycopg2://`): desde SQLAlchemy 2.1, `postgresql://` a secas usa `psycopg` (v3), que no está en `requirements.txt`.
- Con 16 GB de RAM, conviene no tener Docker Desktop abierto mientras el motor corre (etapa 8) y levantarlo solo para la etapa 7, o limitar la memoria de WSL2 en `%UserProfile%\.wslconfig`.
- Queda pendiente también un `docker-compose.yml` para que el equipo levante la misma base con `docker compose up -d`.

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
| **Sensibilidad a pseudo-ausencias** | `python scripts/run_sensibilidad_muestreo.py --config config.yaml` | Contrasta 4 diseños de negativos y mide si la jerarquía SHAP se mantiene |
| **Rendimiento Fijo (23°)** | `python scripts/run_solar_yield.py --mount tilt [--tilt 23]` | Genera mapa de rendimiento fijo a inclinación óptima por latitud (`~23°`) |
| **Rendimiento Plano (0°)** | `python scripts/run_solar_yield.py --mount tilt --tilt 0` | Genera mapa de rendimiento fijo en superficie horizontal (`tilt = 0°`) |
| **Rendimiento Seguidor** | `python scripts/run_solar_yield.py --mount tracker` | Genera mapa de rendimiento para seguidor de un eje (`gcr = 0.3`) |
| **Cruce Aptitud × Rendimiento** | `python scripts/run_cruce.py --config config.yaml` | Cruza celda a celda la aptitud probabilística con la producción física |
| **Comparación de Montaje (0°, 23°, Tracker)** | `python scripts/run_comparacion_montaje.py [--incluir-tilt0] [--regenerar]` | Cuantifica ganancias entre horizontal (`0°`), latitud (`23°`) y seguidor (`tracker`) |
| **Consenso de Perfiles** | `python scripts/run_consenso.py --config config.yaml` | Evalúa consenso vs. divergencia entre perfiles Conservador, Agresivo y RF |
| **Explicabilidad SHAP Global** | `python scripts/run_shap.py --config config.yaml` | Importancia global de características por autovector de Shapley |
| **Explicabilidad SHAP Espacial** | `python scripts/run_shap_spatial.py --config config.yaml` | Grilla espacial de variable dominante por píxel |
| **Métricas Top-K (PEP1)** | `python scripts/run_metricas_topk.py --config config.yaml` | Métricas Recall@K y Precision@K vs. umbrales del informe |
| **Auditoría de fuga (aparte)** | `python scripts/run_auditoria_infraestructura.py --config config.yaml [--optimizar]` | Compara 11 variantes de infraestructura (todas / sin dedicadas / previas) para medir la fuga |
| **Comparación de teselado (aparte)** | `python scripts/run_comparar_tiles.py --config config.yaml [--binario-original <v0.4.0>]` | Motor sin `--tile`, con `--tile` y teselado desde Python sobre recortes con GHI observado (`data/results/analisis/comparar_tiles/`) |
| **Reproducción del bug de `--tile` (autónoma)** | `python scripts/reproducir_bug_tile_ghi.py` (opcional: `--original RUTA --corregido RUTA`) | Escena sintética, sin datos del proyecto: muestra el error de v0.4.0 y verifica la corrección, con figura y `resumen.json` (`reproduccion_bug_tile_ghi/`) |
| **Desacople aptitud–rendimiento (aparte)** | `python scripts/run_desacople_aptitud_rendimiento.py --config config.yaml` | Aptas vs. región y Spearman por celda y por bloque de 30 km, sobre el mapa oficial sin fuga |
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
git checkout 8f17b9f                                      # v0.4.0
git apply ../parches/solarpv-rs_tile_ghi_origen.patch     # corrige --tile con --ghi (ver abajo)
cargo build --release -p solarpv-cli
```

**Por qué el parche:** en v0.4.0, `--tile` junto con `--ghi` recorta el DEM por tesela pero no el GHI, y cada tesela lo lee con sus índices locales: todas leen la esquina noroeste del DEM (en nuestro caso, mar sin dato) y el rendimiento sale 0. El parche pasa a cada tesela su origen dentro del DEM y agrega un test de regresión al motor. Con él, `--tile` da resultados idénticos bit a bit a un teselado equivalente hecho desde Python. Antes de cada corrida con GHI y teselas, el pipeline verifica que el binario tenga la corrección (`verificar_tile_con_ghi`) y se detiene con instrucciones si no la tiene. Ya se reportó al autor del motor; cuando publique una versión corregida, basta con compilar esa versión sin el parche.

### GHI observado

El motor no corre a cielo despejado: reescala la irradiancia con el **GHI anual medio diario del Explorador Solar** (`ghi-m00-ll.tif`, kWh/m²/día), que capta la nubosidad costera. Ese raster se guarda en `data/raw/ghi_explorador_anual.tif` y se alinea con remuestreo bilineal a la grilla exacta del DEM en EPSG:32719 antes de cada corrida (`solarpv.ghi_origen` / `ghi` / `ghi_unit: kwh` en `config.yaml`; Coquimbo usa su propio `transferibilidad.ghi`). Las celdas donde el producto no tiene dato (≈ 7,6 % de la región, en la franja andina oriental) quedan con rendimiento 0 y se excluyen del cruce, de la comparación de montajes y del desacople. Los resultados anteriores, de cielo despejado, están en `data/results/historico/cielo_despejado/`.

### Opciones de Montaje y Tilt
- **Inclinación Óptima a Latitud (`tilt = 23°`):** Inclinación fija estándar para Antofagasta/Atacama (`data/results/rendimiento/rendimiento_fijo_specific_yield.tif`).
- **Montaje Plano Horizontal (`tilt = 0°`):** Módulos sin inclinación horizontal (`data/results/rendimiento/rendimiento_fijo_tilt0_specific_yield.tif`).
- **Seguidor Solar de Un Eje (`tracker`):** Seguidor eje norte-sur con backtracking (`data/results/rendimiento/rendimiento_seguidor_specific_yield.tif`).

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

