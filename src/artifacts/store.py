from abc import ABC, abstractmethod
from typing import Optional


class ArtifactStore(ABC):
    """
    Abstract interface for artifact storage.
    Swap backend (local, S3, GCS etc.) without touching training code.
    """

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        ...

    @abstractmethod
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        ...