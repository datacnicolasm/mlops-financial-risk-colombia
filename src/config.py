import os
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

TRIN_DATA_PATH = PROCESSED_DATA_DIR / "train_pymes_2017_2022.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_pymes_2023_2025.csv"

# Configuracion de MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "riesgo_insolvencia_empresarial"
MODEL_REGISTRY_NAME = "modelo_riesgo_pymes"

# Umbral de reglas minimas
MIN_ROC_AUC = 0.85
MIN_RECALL = 0.75
MIN_PR_AUC = 0.70