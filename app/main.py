from contextlib import asynccontextmanager
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
import pandas as pd

from src import config
from src.features import FinancialRatioTransformer
from app.schemas import CompanyFinancialPayload, PredictionResponse

ml_components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        model_uri = f"models:/{config.MODEL_REGISTRY_NAME}@champion"
        
        ml_components["model"] = mlflow.sklearn.load_model(
            model_uri=model_uri
        )
        ml_components["transformer"] = FinancialRatioTransformer()
        
        ml_components["version"] = "champion"
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar Champion: {e}")
        ml_components["model"] = None
        ml_components["transformer"] = FinancialRatioTransformer()
        ml_components["version"] = "unloaded"
    yield
    
    ml_components.clear()

app = FastAPI(
    title="Financial Distress Inference Service",
    description="Microservicio de inferencia para riesgo financiero empresarial (Porvenir MLOps Pipeline)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    status = "healthy" if ml_components.get("model") is not None else "degraded"
    return {"status": status, "model_version": ml_components.get("version")}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: CompanyFinancialPayload):
    model = ml_components.get("model")
    transformer = ml_components.get("transformer")

    if not model or not transformer:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # Convertir payload a DataFrame
    df_raw = pd.DataFrame([payload.model_dump()])

    # Calcular ratios
    X_features = transformer.transform(df_raw).astype(float)

    # Inferencia
    proba = float(model.predict_proba(X_features)[:, 1][0])
    pred = int(proba >= 0.5)

    if proba >= 0.70:
        alerta = "ALTO"
    elif proba >= 0.40:
        alerta = "MEDIO"
    else:
        alerta = "BAJO"

    return PredictionResponse(
        nit=payload.nit,
        probabilidad_estres=round(proba, 4),
        clasificacion_riesgo=pred,
        nivel_alerta=alerta,
        model_version=ml_components["version"]
    )
    