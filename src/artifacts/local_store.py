import mlflow
from typing import Optional
from src.artifacts.store import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    """
    Stores artifacts in the local mlruns/ directory.
    Use during development and experimentation.
    """

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        mlflow.log_artifacts(local_dir, artifact_path)