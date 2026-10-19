import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score
)
from src import config

def main():
    # Configuracion MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Carga de datos
    df_train = pd.read_csv(config.TRIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    X_train = df_train.drop(columns=["target"]).astype(float)
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target"]).astype(float)
    y_test = df_test["target"]

    # Hiperparametros del modelo
    rf_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 50,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    with mlflow.start_run(run_name="challenger_random_forest") as run:
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        # Predicciones e inferencia
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculo de metricas sobre test
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
            "recall": float(recall_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred))
        }

        # Logging a MLflow
        mlflow.log_params(rf_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("split", "out-of-time (2017-2022 -> 2023-2025)")

        # Firma del modelo
        signature = infer_signature(X_train, model.predict(X_train))

        # Registrar el artefacto
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
