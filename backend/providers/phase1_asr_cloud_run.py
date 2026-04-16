from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from .config import Phase1ASRSettings
from .phase1_asr_contracts import Phase1ASRRequest, Phase1ASRResponse


@dataclass(slots=True)
class CloudRunVibeVoiceProvider:
    settings: Phase1ASRSettings
    hotwords_context: str
    max_new_tokens: int = 32768
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1

    # Remote ASR runs off-box, so Phase 1 may overlap visual extraction safely.
    supports_concurrent_visual: bool = True

    def run(
        self,
        audio_path: str | Path,
        context_info: str | None = None,
        audio_gcs_uri: str | None = None,
    ) -> list[dict[str, Any]]:
        del audio_path
        if not audio_gcs_uri:
            raise ValueError(
                "audio_gcs_uri is required for Cloud Run Phase 1 ASR; local-file fallback is disabled."
            )
        payload = Phase1ASRRequest(
            audio_gcs_uri=audio_gcs_uri,
            context_info=context_info if context_info is not None else self.hotwords_context,
            generation_config={
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "num_beams": self.num_beams,
            },
        )
        request = urllib.request.Request(
            self._endpoint_url(),
            data=json.dumps(
                payload.model_dump(mode="json", exclude_none=True),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                **self._auth_header(),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as resp:
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"remote Phase 1 ASR failed with HTTP {exc.code}: {detail[:2000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"remote Phase 1 ASR request failed: {exc.reason}") from exc

        try:
            payload_dict = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"remote Phase 1 ASR returned non-JSON response: {raw_body[:2000]}"
            ) from exc
        if isinstance(payload_dict, list):
            payload_dict = {"turns": payload_dict}
        response = Phase1ASRResponse.model_validate(payload_dict)
        return [dict(item) for item in response.turns]

    def _endpoint_url(self) -> str:
        base_url = (self.settings.service_url or "").strip()
        if not base_url:
            raise ValueError(
                "CLYPT_PHASE1_ASR_SERVICE_URL is required when CLYPT_PHASE1_ASR_BACKEND=cloud_run_l4."
            )
        return f"{base_url.rstrip('/')}/tasks/asr"

    def _auth_header(self) -> dict[str, str]:
        auth_mode = (self.settings.auth_mode or "id_token").strip().lower()
        if auth_mode == "none":
            return {}
        if auth_mode != "id_token":
            raise ValueError(
                "Unsupported CLYPT_PHASE1_ASR_AUTH_MODE="
                f"{self.settings.auth_mode!r}; expected none|id_token."
            )
        audience = (self.settings.audience or self.settings.service_url or "").strip()
        if not audience:
            raise ValueError(
                "Phase 1 Cloud Run ASR auth_mode=id_token requires audience or service_url."
            )

        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import id_token

        token = id_token.fetch_id_token(GoogleAuthRequest(), audience)
        return {"Authorization": f"Bearer {token}"}


__all__ = ["CloudRunVibeVoiceProvider"]
