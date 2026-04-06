# Jenkins Financial Risk Auto

Proyecto de Ciencia de Datos en Produccion para evaluar riesgo crediticio usando:

- Pipeline de entrenamiento y evaluacion de modelos.
- API en FastAPI para inferencia.
- Frontend en Streamlit para consumo de predicciones.
- Validaciones DevOps locales (estructura y secretos).
- Pipeline Jenkins para automatizar chequeos de calidad basicos.

## 1. Resumen

Este repositorio integra un flujo de trabajo end-to-end:

1. Carga de datos desde BigQuery.
2. Feature engineering con transformadores personalizados compatibles con scikit-learn.
3. Entrenamiento y evaluacion de multiples modelos (Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost, LightGBM).
4. Despliegue de inferencia via API (modelo unico o panel multi-modelo).
5. Interfaz visual con Streamlit.
6. Validaciones de proyecto y escaneo de posibles secretos con scripts de pyops.
7. Ejecucion de validaciones en Jenkins ante eventos de push/PR/merge (Webhooks).

## 2. Estructura del proyecto

```text
Jenkins_Financial_Risk_Auto/
|- docker-compose-jenkins.yml
|- Jenkinsfile
|- requirements.txt
|- README.md
|- pyops/
|	|- chequeo_archivos.py
|	|- scanner_secretos.py
|	|- validador_proyecto.py
|- pipeline_MLOps/
|  |- docker-compose.yml
|  |- Dockerfile
|  |- requirements.txt
|  |- src/
|  |  |- cargar_datos.py
|  |  |- ft_engineering.py
|  |  |- heuristic_model.py
|  |  |- model_training_evaluation.py
|  |  |- model_deploy.py
|  |  |- multimodel_deploy.py
|  |  |- model_interface.py
|  |  |- multimodel_interface.py
|  |- outputs/
|     |- artefactos/
|     |- modelo_heuristico/
|     |- reporte_training_evaluation/
```

## 3. Que hace cada parte

### Orquestacion y CI

- `Jenkinsfile`: detecta tipo de evento y ejecuta validaciones de proyecto, escaneo de secretos (gitleaks) y validacion de compose.
- `docker-compose-jenkins.yml`: levanta Jenkins en contenedor y monta el repo dentro de `/workspace`.

### Pipeline ML y despliegue

- `pipeline_MLOps/src/cargar_datos.py`: consulta tabla de scoring en BigQuery y devuelve un DataFrame.
- `pipeline_MLOps/src/ft_engineering.py`: construye y guarda pipeline de preprocesamiento.
- `pipeline_MLOps/src/heuristic_model.py`: modelo heuristico basado en reglas para comparacion.
- `pipeline_MLOps/src/model_training_evaluation.py`: tuning, entrenamiento, evaluacion y guardado de modelos.
- `pipeline_MLOps/src/model_deploy.py`: API FastAPI para prediccion con modelo unico.
- `pipeline_MLOps/src/multimodel_deploy.py`: API FastAPI con panel de expertos (4 modelos).
- `pipeline_MLOps/src/model_interface.py`: interfaz Streamlit para modelo unico.
- `pipeline_MLOps/src/multimodel_interface.py`: interfaz Streamlit para panel multi-modelo.
- `pipeline_MLOps/outputs/`: carpeta de artefactos y reportes generados.

### Operaciones y validacion

- `pyops/chequeo_archivos.py`: valida rutas requeridas y chequeos basicos del Dockerfile.
- `pyops/scanner_secretos.py`: escaneo regex de posibles secretos.
- `pyops/validador_proyecto.py`: orquestador CLI de validaciones. Este además cuenta con 2 parámetros: `--verbose` o `--silent`.

## 4. Variables de entorno

Crear archivo `.env` dentro de `pipeline_MLOps/` con valores como:

```env
PROJECT_GCP=tu_proyecto_gcp
ARTIFACTS=./outputs/artefactos
DATA_FOLDER=./src/data
OUTPUTS=./outputs/reporte_training_evaluation
DATA_FILE=./src/data
API_URL=http://127.0.0.1:8000/predict
```

Notas:

- Para BigQuery, configura credenciales ADC: `gcloud auth application-default login`.
- En Docker, `API_URL` suele apuntar a `http://api:8000/predict` (ya definido en `docker-compose.yml`).

## 5. Paso a paso para correr todo

### A. Preparar entorno local (Windows PowerShell)

Desde la raiz del repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### B. Ejecutar validaciones de proyecto (pyops)

```powershell
python pyops/validador_proyecto.py --verbose
```

### C. Entrenar pipeline y modelos

```powershell
cd pipeline_MLOps
python src/ft_engineering.py
python src/model_training_evaluation.py
```

Esto debe generar artefactos en `pipeline_MLOps/outputs/artefactos`.

### D. Levantar API y frontend (modelo unico)

Terminal 1 (en `pipeline_MLOps`):

```powershell
uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000
```

Terminal 2 (en `pipeline_MLOps`):

```powershell
streamlit run src/model_interface.py
```

Accesos:

- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:8501`

### E. Levantar API y frontend (panel multi-modelo)

Terminal 1 (en `pipeline_MLOps`):

```powershell
uvicorn src.multimodel_deploy:app --host 0.0.0.0 --port 8000
```

Terminal 2 (en `pipeline_MLOps`):

```powershell
streamlit run src/multimodel_interface.py
```

### F. Ejecutar con Docker Compose (pipeline)

En `pipeline_MLOps`:

```powershell
docker compose up --build
```

Servicios:

- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

### G. Levantar Jenkins local con Docker

Desde la raiz del repo:

```powershell
docker compose -f docker-compose-jenkins.yml up -d
```

Luego abrir `http://localhost:8080` y crear un pipeline apuntando a este repositorio.

## 6. Flujo recomendado de trabajo

1. Ejecutar `pyops/validador_proyecto.py` antes de commitear.
2. Entrenar y actualizar artefactos si hubo cambios en datos o features.
3. Probar API (`/docs`) y UI Streamlit.
4. Verificar pipeline Jenkins en push/PR/merge.