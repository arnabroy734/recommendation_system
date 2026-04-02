import mlflow
from typing import Optional
from src.artifacts.store import ArtifactStore
from src.tracking.config import DEFAULT_TAGS


class ExperimentTracker:
    """
    Wraps MLflow run lifecycle. Training scripts interact only with this class,
    never with mlflow directly.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        store: ArtifactStore,
        tags: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.store = store
        self.tags = {**DEFAULT_TAGS, **(tags or {})}
        self._run = None

    def __enter__(self) -> "ExperimentTracker":
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")
        return False  # do not suppress exceptions

    @property
    def run_id(self) -> str:
        return self._run.info.run_id if self._run else None

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        self.store.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        self.store.log_artifacts(local_dir, artifact_path)

    def log_model(self, model, artifact_path: str = "model"):
        import mlflow.pytorch
        mlflow.pytorch.log_model(model, artifact_path)

    def set_tag(self, key: str, value: str):
        mlflow.set_tag(key, value)