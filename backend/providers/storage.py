from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import NewType

from .config import StorageSettings

logger = logging.getLogger(__name__)

GcsUri = NewType("GcsUri", str)


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


def parse_gcs_uri(uri: str | GcsUri) -> tuple[str, str]:
    """Return ``(bucket, object_key)`` for a ``gs://`` URI."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri!r}")
    rest = uri[len("gs://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed GCS URI: {uri!r}")
    return bucket, key


def _normalize_node_media_object_prefix(
    bucket: str,
    prefix_or_uri: str,
    *,
    job_id: str | None = None,
) -> str:
    prefix = prefix_or_uri.strip()
    if not prefix:
        if job_id is None:
            raise ValueError("node-media object prefix is required")
        return f"phase14/{job_id}/node_media"
    if prefix.startswith("gs://"):
        uri_bucket, prefix = parse_gcs_uri(prefix)
        if bucket and uri_bucket != bucket:
            raise ValueError(f"Expected gs://{bucket}/... URI, got: {prefix_or_uri!r}")
    prefix = prefix.strip().strip("/")
    if not prefix:
        if job_id is None:
            raise ValueError("node-media object prefix is required")
        return f"phase14/{job_id}/node_media"
    return prefix


class GCSStorageClient:
    def __init__(self, *, settings: StorageSettings, storage_client=None) -> None:
        self.settings = settings
        self._client = storage_client or _build_default_storage_client()

    def upload_file(self, *, local_path: Path, object_name: str) -> str:
        bucket = self._client.bucket(self.settings.gcs_bucket)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.settings.gcs_bucket}/{object_name}"

    def download_file(self, *, gcs_uri: GcsUri | str, local_path: Path) -> Path:
        bucket_name, object_name = parse_gcs_uri(gcs_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.download_to_filename(str(local_path))
        return local_path

    def get_https_url(self, gcs_uri: GcsUri | str, expiry_hours: int = 24) -> str:
        """Return a V4 signed HTTPS URL for a GCS object.

        Requires service-account credentials with signing ability. If
        signing fails (missing/rotated creds, quota, transient auth), we
        raise :class:`GcsSigningError` rather than making the object public
        — silently flipping bucket ACL to public on failure would expose
        customer data and violates fail-fast doctrine.
        """
        bucket_name, object_name = parse_gcs_uri(gcs_uri)

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


__all__ = ["GCSStorageClient", "GcsSigningError", "GcsUri", "parse_gcs_uri"]
