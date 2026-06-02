from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "model_artifacts"

MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ROOT / "mlflow_artifacts"

ADAPTER_TAR = ARTIFACT_DIR / "qwen_lora_adapter.tar.gz"
MERGED_TAR = ARTIFACT_DIR / "qwen_lora_merged.tar.gz"

EXPERIMENT_NAME = "lambda-llm-artifact-registry"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_LOSS = 2.8668886886702643


def size_mb(path):
    return round(path.stat().st_size / (1024 * 1024), 2)


def ensure_experiment(client):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri(),
        )
        return experiment_id

    return experiment.experiment_id


def ensure_registered_model(client, name):
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def register_artifact(client, experiment_id, run_name, model_name, artifact_path, artifact_type, vllm_ready):
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")

    ensure_registered_model(client, model_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_param("base_model", BASE_MODEL)
        mlflow.log_param("training_method", "LoRA SFT")
        mlflow.log_param("artifact_type", artifact_type)
        mlflow.log_param("artifact_file", artifact_path.name)
        mlflow.log_param("artifact_format", "tar.gz")

        mlflow.log_metric("train_loss", TRAIN_LOSS)
        mlflow.log_metric("artifact_size_mb", size_mb(artifact_path))

        mlflow.set_tag("status", "candidate")
        mlflow.set_tag("vllm_ready", str(vllm_ready).lower())
        mlflow.set_tag("project", "lambda-field-engineering-prep")

        mlflow.log_artifact(str(artifact_path), artifact_path="model_artifact")

        source_uri = mlflow.get_artifact_uri("model_artifact")

        version = client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=run.info.run_id,
        )

        print(f"Registered {model_name} version {version.version}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Source: {source_uri}")


def main():
    tracking_uri = f"sqlite:///{MLFLOW_DB}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()
    experiment_id = ensure_experiment(client)

    register_artifact(
        client=client,
        experiment_id=experiment_id,
        run_name="qwen_lora_adapter_v1",
        model_name="qwen_lora_adapter",
        artifact_path=ADAPTER_TAR,
        artifact_type="lora_adapter",
        vllm_ready=False,
    )

    register_artifact(
        client=client,
        experiment_id=experiment_id,
        run_name="qwen_lora_merged_v1",
        model_name="qwen_lora_merged",
        artifact_path=MERGED_TAR,
        artifact_type="merged_full_model",
        vllm_ready=True,
    )

    print("MLflow registration complete.")
    print(f"Tracking DB: {MLFLOW_DB}")
    print(f"Artifact store: {MLFLOW_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
