from __future__ import annotations

import datetime
import logging
from pathlib import Path

from .config import StorageSettings

logger = logging.getLogger(__name__)


class GcsSigningError(RuntimeError):
    """Raised when V4 signed URL generation for a GCS object fails.

    We deliberately do not fall back to `blob.make_public()` — silently
    flipping a bucket object to public ACL on a signing failure (rotated
    creds, quota, transient auth) is a data-exposure risk and violates the
    fail-fast doctrine. Callers must handle this exception explicitly.
    """

    def __init__(self, gcs_uri: str, *, cause: Exception | None = None) -> None:
        super().__init__(
            f"Failed to generate V4 signed URL for {gcs_uri!r}"
            + (f": {cause}" if cause is not None else "")
        )
        self.gcs_uri = gcs_uri


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

    def download_file(self, *, gcs_uri: str, local_path: Path) -> Path:
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got: {gcs_uri!r}")
        path = gcs_uri[len("gs://") :]
        bucket_name, _, object_name = path.partition("/")
        if not bucket_name or not object_name:
            raise ValueError(f"Expected gs://bucket/object URI, got: {gcs_uri!r}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(str(local_path))
        return local_path

    def get_https_url(self, gcs_uri: str, expiry_hours: int = 24) -> str:
        """Return a V4 signed HTTPS URL for a GCS object.

        Requires service-account credentials with signing ability. If
        signing fails (missing/rotated creds, quota, transient auth), we
        raise :class:`GcsSigningError` rather than making the object public
        — silently flipping bucket ACL to public on failure would expose
        customer data and violates fail-fast doctrine.
        """
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got: {gcs_uri!r}")
        path = gcs_uri[len("gs://"):]
        bucket_name, _, object_name = path.partition("/")

        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        creds = self._client._credentials
        try:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(hours=expiry_hours),
                method="GET",
                credentials=creds,
            )
        except Exception as signing_err:
            raise GcsSigningError(gcs_uri, cause=signing_err) from signing_err
        logger.info("[gcs]  signed URL generated for %s", gcs_uri)
        return signed_url


__all__ = ["GCSStorageClient", "GcsSigningError"]
