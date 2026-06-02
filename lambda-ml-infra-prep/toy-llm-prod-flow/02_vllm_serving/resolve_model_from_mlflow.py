from pathlib import Path
import json
import tarfile
import shutil
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

MODEL_NAME = "qwen_lora_merged"
MODEL_VERSION = "1"

DOWNLOAD_DIR = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "downloaded_artifacts"
SERVING_MODEL_DIR = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "models" / "qwen_lora_merged_v1"
SELECTED_MODEL_FILE = ROOT / "toy-llm-prod-flow" / "02_vllm_serving" / "selected_model.json"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, MODEL_VERSION)

    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    local_artifact_path = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source,
        dst_path=str(DOWNLOAD_DIR),
    )

    local_artifact_path = Path(local_artifact_path)

    tar_files = list(local_artifact_path.rglob("*.tar.gz"))

    if len(tar_files) == 0:
        raise FileNotFoundError(f"No tar.gz artifact found under {local_artifact_path}")

    tar_path = tar_files[0]

    if SERVING_MODEL_DIR.exists():
        shutil.rmtree(SERVING_MODEL_DIR)

    SERVING_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    extract_root = SERVING_MODEL_DIR / "_extract"
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_root)

    extracted_dirs = list(extract_root.rglob("config.json"))

    if len(extracted_dirs) == 0:
        raise FileNotFoundError("Could not find config.json after extracting model artifact.")

    actual_model_dir = extracted_dirs[0].parent

    for item in actual_model_dir.iterdir():
        target = SERVING_MODEL_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    shutil.rmtree(extract_root)

    selected = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "mlflow_source": version.source,
        "serving_model_dir": str(SERVING_MODEL_DIR),
        "vllm_ready": True,
    }

    with open(SELECTED_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    print("Resolved model from MLflow registry.")
    print(f"Model: {MODEL_NAME}")
    print(f"Version: {MODEL_VERSION}")
    print(f"Serving model dir: {SERVING_MODEL_DIR}")
    print(f"Selected model file: {SELECTED_MODEL_FILE}")


if __name__ == "__main__":
    main()
