from __future__ import annotations

import datetime
import logging
from pathlib import Path

from .config import StorageSettings

logger = logging.getLogger(__name__)


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

    def get_https_url(self, gcs_uri: str, expiry_hours: int = 24) -> str:
        """Return an HTTPS URL for a GCS object suitable for third-party access.

        Tries V4 signed URL first (works for service account credentials).
        Falls back to making the object publicly readable and returning the
        storage.googleapis.com URL (acceptable for smoke-test / dev).
        """
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got: {gcs_uri!r}")
        path = gcs_uri[len("gs://"):]
        bucket_name, _, object_name = path.partition("/")

        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        # Try V4 signed URL (requires service account credentials with signing ability).
        try:
            creds = self._client._credentials
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(hours=expiry_hours),
                method="GET",
                credentials=creds,
            )
            logger.info("[gcs]  signed URL generated for %s", gcs_uri)
            return signed_url
        except Exception as signing_err:
            logger.warning(
                "[gcs]  signed URL failed (%s) — falling back to public URL", signing_err
            )

        # Fallback: make the object public and return the canonical HTTPS URL.
        blob.make_public()
        public_url = blob.public_url
        logger.info("[gcs]  object made public: %s", public_url)
        return public_url


__all__ = ["GCSStorageClient"]
