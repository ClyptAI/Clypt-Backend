from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import urllib.error
import urllib.request

from backend.pipeline.semantics.media_prep_contracts import (
    NodeMediaPrepItem,
    NodeMediaPrepRequest,
    NodeMediaPrepResponse,
)

from .config import Phase24MediaPrepSettings


@dataclass(slots=True)
class CloudRunMediaPrepClient:
    settings: Phase24MediaPrepSettings

    def prepare_node_media(
        self,
        *,
        nodes: list[Any],
        paths: Any,
        phase1_outputs: Any,
    ) -> list[dict[str, str]]:
        if not nodes:
            return []
        phase1_audio = dict(getattr(phase1_outputs, "phase1_audio", {}) or {})
        source_video_gcs_uri = str(phase1_audio.get("video_gcs_uri") or "").strip()
        if not source_video_gcs_uri:
            raise ValueError(
                "phase1_outputs.phase1_audio.video_gcs_uri is required for remote node media prep."
            )

        request_payload = NodeMediaPrepRequest(
            run_id=str(paths.run_id),
            source_video_gcs_uri=source_video_gcs_uri,
            object_prefix=f"phase14/{paths.run_id}/node_media",
            items=[
                NodeMediaPrepItem(
                    node_id=str(node.node_id),
                    start_ms=int(node.start_ms),
                    end_ms=int(node.end_ms),
                )
                for node in nodes
            ],
        )
        response = self._post_request(request_payload)
        expected_order = [item.node_id for item in request_payload.items]
        actual_order = [item.node_id for item in response.items]
        if actual_order != expected_order:
            raise RuntimeError(
                "remote node media prep returned descriptors in unexpected order: "
                f"expected={expected_order} actual={actual_order}"
            )
        return [
            item.model_dump(mode="json", exclude_none=True)
            for item in response.items
        ]

    def _endpoint_url(self) -> str:
        base_url = (self.settings.service_url or "").strip()
        if not base_url:
            raise ValueError(
                "CLYPT_PHASE24_MEDIA_PREP_SERVICE_URL is required when remote media prep is enabled."
            )
        return f"{base_url.rstrip('/')}/tasks/node-media-prep"

    def _auth_header(self) -> dict[str, str]:
        auth_mode = (self.settings.auth_mode or "id_token").strip().lower()
        if auth_mode == "none":
            return {}
        if auth_mode != "id_token":
            raise ValueError(
                "Unsupported CLYPT_PHASE24_MEDIA_PREP_AUTH_MODE="
                f"{self.settings.auth_mode!r}; expected none|id_token."
            )

        audience = (self.settings.audience or self.settings.service_url or "").strip()
        if not audience:
            raise ValueError(
                "phase24 media prep auth_mode=id_token requires audience or service_url."
            )

        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import id_token

        token = id_token.fetch_id_token(GoogleAuthRequest(), audience)
        return {"Authorization": f"Bearer {token}"}

    def _post_request(self, payload: NodeMediaPrepRequest) -> NodeMediaPrepResponse:
        if self.settings.timeout_s <= 0:
            raise ValueError("CLYPT_PHASE24_MEDIA_PREP_TIMEOUT_S must be > 0.")
        request_body = json.dumps(
            payload.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint_url(),
            data=request_body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **self._auth_header(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.settings.timeout_s) as resp:
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"remote node media prep failed with HTTP {exc.code}: {detail[:2000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"remote node media prep request failed: {exc.reason}"
            ) from exc

        try:
            payload_dict = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"remote node media prep returned non-JSON response: {raw_body[:2000]}"
            ) from exc
        return NodeMediaPrepResponse.model_validate(payload_dict)


__all__ = ["CloudRunMediaPrepClient"]
