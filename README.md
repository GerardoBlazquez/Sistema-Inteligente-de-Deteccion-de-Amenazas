# Sistema Inteligente de Detección de Amenazas

Sistema de detección de amenazas basado en Machine Learning para dos dominios:

- **Fraude bancario**
- **Detección de bots web**

El proyecto combina modelado supervisado y no supervisado, optimización de umbral, análisis económico, persistencia de modelos y exposición mediante API REST. En el repositorio, el backend final está implementado con **FastAPI** y se ejecuta con **Uvicorn**; además, se incluye despliegue preparado para **Render**. fileciteturn328542view0turn211881view3turn170577view1

## Resumen del proyecto

La arquitectura separa claramente el entrenamiento de los modelos y la inferencia en producción. El sistema permite seleccionar el dominio, el modelo y el modo de decisión, devolviendo una respuesta en JSON con la probabilidad, la clasificación final, el umbral usado y la importancia de características cuando está disponible. fileciteturn328542view0turn211881view4

### Dominios incluidos

- **Fraude financiero**: modelo supervisado y modelos de anomalías para transacciones con un conjunto de variables tipo `Time`, `V1`...`V28`, `Amount`. fileciteturn451354view0
- **Bots web**: modelo basado en comportamiento agregado por IP con variables `requests_per_ip`, `avg_time_diff`, `error_rate` y `unique_resources`. fileciteturn451354view0turn2file8

## Características principales

- API REST con endpoints para salud, listado de modelos, predicción, métricas y análisis económico por lotes. fileciteturn328542view0turn211881view4
- Registro centralizado de modelos por dominio y por algoritmo. fileciteturn451354view0
- Umbrales configurables por modo:
  - **Modo `f1`**: equilibrio entre precisión y recall.
  - **Modo `auto_cost`** / **`cost`**: optimización orientada al coste.
  - **Modo `balanced`** o equivalente, según el modelo cargado. fileciteturn211881view4turn451354view0
- Métricas de observabilidad con **Prometheus**. fileciteturn211881view2turn170577view0
- Preparado para despliegue cloud mediante `render.yaml`. fileciteturn170577view1

## Modelos disponibles

### Fraude
- `random_forest`
- `isolation_forest`
- `lof`
- `ocsvm` fileciteturn451354view0

### Bots
- `random_forest`
- `xgboost` fileciteturn451354view0turn2file0

## Resultados destacados del TFG

En la memoria del proyecto, el modelo supervisado de fraude con **Random Forest optimizado** alcanzó un **F1-score de 0.79** y un **ROC-AUC de 0.97**, reduciendo aproximadamente un **49%** del coste estimado frente al enfoque no supervisado optimizado. fileciteturn2file11turn2file15

Para bots, el modelo final seleccionado fue **XGBoost optimizado**, con un escenario equilibrado alrededor de **threshold 0.41** y otro modo de seguridad alrededor de **0.08**; el estudio concluye que XGBoost ofrece mejor separación probabilística y buen equilibrio entre precisión y recall. fileciteturn2file1turn2file9turn2file16

## Estructura del repositorio

```text
.
├── main.py
├── registry.py
├── requirements.txt
├── render.yaml
├── models/
│   ├── fraud/
│   │   ├── fraud_random_forest.pkl
│   │   ├── fraud_isolation_forest.pkl
│   │   ├── fraud_lof.pkl
│   │   └── fraud_ocsvm.pkl
│   └── bots/
│       ├── random_forest.pkl
│       └── xgboost.pkl
├── scalers/
│   └── fraud_scaler.pkl
└── README.md
```

Los nombres anteriores reflejan la organización que usa `registry.py` para cargar los modelos de forma segura desde `models/` y, en el caso de fraude, también el escalador desde `scalers/`. fileciteturn451354view0

## Requisitos

El proyecto usa Python y, según `requirements.txt`, depende de:

- `fastapi`
- `uvicorn`
- `gunicorn`
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`
- `prometheus-client` fileciteturn170577view0

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/GerardoBlazquez/Sistema-Inteligente-de-Deteccion-de-Amenazas.git
cd Sistema-Inteligente-de-Deteccion-de-Amenazas
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

#### Windows
```bash
.venv\Scripts\activate
```

#### Linux / macOS
```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución

### Modo desarrollo

```bash
python main.py
```

### Modo producción local

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

El propio `main.py` arranca Uvicorn en el puerto `8000` cuando se ejecuta como script, y `render.yaml` usa `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`. fileciteturn211881view3turn170577view1

## API REST

### `GET /health`

Devuelve el estado general del servicio y qué modelos están disponibles.

### `GET /models`

Lista los modelos cargados por dominio y sus features asociadas.

### `POST /predict`

Realiza una predicción sobre el dominio y modelo seleccionados.

#### Body de ejemplo

```json
{
  "domain": "bots",
  "model_name": "xgboost",
  "mode": "f1",
  "data": {
    "requests_per_ip": 120,
    "avg_time_diff": 0.42,
    "error_rate": 0.18,
    "unique_resources": 4
  }
}
```

#### Respuesta de ejemplo

```json
{
  "score": 0.87,
  "classification": 1,
  "threshold_used": 0.41,
  "domain": "bots",
  "model": "xgboost",
  "mode": "f1",
  "features": [
    "requests_per_ip",
    "avg_time_diff",
    "error_rate",
    "unique_resources"
  ],
  "feature_importance": [0.42, 0.31, 0.18, 0.09]
}
```

El endpoint valida el dominio y el modelo, construye el vector de entrada en el orden esperado y aplica el umbral correspondiente antes de devolver la respuesta. fileciteturn211881view4turn328542view0

### `POST /economic-analysis/batch`

Permite evaluar lotes de scores y etiquetas reales para calcular el mejor umbral por coste y el ROI frente a una línea base. fileciteturn211881view4

### `GET /metrics`

Expone métricas en formato Prometheus para observabilidad. fileciteturn211881view3

## Formatos de entrada esperados

### Fraude
El modelo espera las variables:

`Time`, `V1`...`V28`, `Amount`. fileciteturn451354view0

### Bots
El modelo espera:

`requests_per_ip`, `avg_time_diff`, `error_rate`, `unique_resources`. fileciteturn451354view0turn2file8

## Despliegue en Render

El proyecto incluye `render.yaml` para despliegue como web service en Render, con instalación de dependencias y arranque mediante Uvicorn. fileciteturn170577view1

## Dataset y entrenamiento

La memoria del TFG documenta dos pipelines de entrenamiento:

- **Fraude bancario**: `Credit Card Fraud Detection`, con gran desbalance de clases.
- **Bots web**: logs en JSON transformados a un dataset agregado por IP. fileciteturn2file17turn2file12

El entrenamiento y la experimentación se realizaron en Google Colab, separando claramente la fase offline de entrenamiento y la fase online de inferencia. fileciteturn2file17

## Licencia

MIT. fileciteturn903588view0

## Autor

**Gerardo Blázquez Moreno**
