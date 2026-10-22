import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from src import config

def evaluate_and_promote():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Obtener el experimento
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if not experiment:
        print("Error: experimento no encontrado.")
        sys.exit(1)

    # Buscar la ultima corrida activa del experimento
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1    
    )

    if not runs:
        print("No se encontraron corridas en MLflow.")
        sys.exit(1)

    latest_run = runs[0]
    metrics = latest_run.data.metrics
    run_id = latest_run.info.run_id

    # Validacion contra los umbrales minimos definidos
    passes_quality_gate = (
        metrics.get("roc_auc", 0) >= config.MIN_ROC_AUC and
        metrics.get("recall", 0) >= config.MIN_RECALL and
        metrics.get("pr_auc", 0) >= config.MIN_PR_AUC
    )

    if not passes_quality_gate:
        print("Pruebas de calidad no superadas.")
        sys.exit(1)

    print("El modelo supero las pruebas.")

    # Registrar en el Model Registry y promover a Champion
    try:
        client.get_registered_model(config.MODEL_REGISTRY_NAME)
    except RestException:
        print(f"Creando nuevo contenedor en Model Registry: '{config.MODEL_REGISTRY_NAME}'...")
        client.create_registered_model(config.MODEL_REGISTRY_NAME)

    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=config.MODEL_REGISTRY_NAME
    )

    # Asignar el alias
    client.set_registered_model_alias(
        name=config.MODEL_REGISTRY_NAME,
        alias="champion",
        version=model_version.version
    )

    print("Modelo promovido con exito")

if __name__ == "__main__":
    evaluate_and_promote()