from __future__ import annotations

from pathlib import Path

from .config import StorageSettings


def _build_default_storage_client():
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for V3.1 GCS uploads."
        ) from exc
    return storage.Client()


class GCSStorageClient:
    def __init__(self, *, settings: StorageSettings, storage_client=None) -> None:
        self.settings = settings
        self._client = storage_client or _build_default_storage_client()

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        bucket = self._client.bucket(self.settings.gcs_bucket)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.settings.gcs_bucket}/{object_name}"


__all__ = ["GCSStorageClient"]
