from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ._http_retry import (
    RemoteServiceHTTPError,
    TRANSIENT_HTTP_STATUS,
    retry_with_backoff,
)

logger = logging.getLogger(__name__)


SCRIBE_SPEECH_TO_TEXT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsScribeError(RemoteServiceHTTPError):
    """Raised when ElevenLabs Scribe rejects or returns an invalid response."""


@dataclass(slots=True)
class ElevenLabsScribeSettings:
    api_key: str
    model_id: str = "scribe_v2"
    endpoint_url: str = SCRIBE_SPEECH_TO_TEXT_URL
    diarize: bool = True
    tag_audio_events: bool = True
    timestamps_granularity: str = "word"
    language_code: str = "en"
    temperature: float = 0.0
    timeout_s: float = 7200.0
    max_retries: int = 2
    num_speakers: int | None = None
    diarization_threshold: float | None = None
    keyterms: tuple[str, ...] = ()
    seed: int | None = None
    url_field: str = "source_url"


@dataclass(slots=True)
class ScribeRequestOptions:
    num_speakers: int | None = None
    diarization_threshold: float | None = None
    keyterms: list[str] | None = None
    seed: int | None = None
    file_format: str | None = None


@dataclass(slots=True)
class ScribeTranscript:
    raw: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> list[dict[str, Any]]:
        words = self.raw.get("words")
        return words if isinstance(words, list) else []


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ElevenLabsScribeError):
        return exc.status_code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, OSError):
        return True
    return False


def _coerce_bool(value: bool) -> str:
    return "true" if value else "false"


def _build_request_fields(
    *,
    source_url: str,
    settings: ElevenLabsScribeSettings,
    options: ScribeRequestOptions | None,
) -> dict[str, str]:
    if not source_url.startswith("https://"):
        raise ValueError("ElevenLabs Scribe source_url must be a signed HTTPS URL.")
    opts = options or ScribeRequestOptions()
    url_field = getattr(settings, "url_field", "source_url") or "source_url"
    if url_field not in {"source_url", "cloud_storage_url"}:
        raise ValueError("ElevenLabs Scribe URL field must be source_url or cloud_storage_url.")
    fields: dict[str, str] = {
        "model_id": settings.model_id,
        url_field: source_url,
        "diarize": _coerce_bool(settings.diarize),
        "tag_audio_events": _coerce_bool(settings.tag_audio_events),
        "timestamps_granularity": settings.timestamps_granularity,
        "language_code": settings.language_code,
        "temperature": str(settings.temperature),
    }
    if opts.file_format:
        fields["file_format"] = opts.file_format
    if opts.num_speakers is not None:
        fields["num_speakers"] = str(int(opts.num_speakers))
    if opts.diarization_threshold is not None:
        fields["diarization_threshold"] = str(float(opts.diarization_threshold))
    if opts.keyterms:
        fields["keyterms"] = json.dumps(list(opts.keyterms), ensure_ascii=True)
    if opts.seed is not None:
        fields["seed"] = str(int(opts.seed))
    return fields


def _multipart_body(fields: dict[str, str], *, boundary: str) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks)


def validate_scribe_response(
    raw: Any,
    *,
    diarize: bool = True,
    require_word_tokens: bool = True,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ElevenLabsScribeError(
            f"Scribe response must be an object, got {type(raw).__name__}"
        )
    words = raw.get("words")
    if not isinstance(words, list):
        raise ElevenLabsScribeError("Scribe response must include a words list.")

    word_token_count = 0
    for index, item in enumerate(words):
        if not isinstance(item, dict):
            raise ElevenLabsScribeError(
                f"Scribe words[{index}] must be an object, got {type(item).__name__}"
            )
        if item.get("type") != "word":
            continue
        word_token_count += 1
        if item.get("start") is None or item.get("end") is None:
            raise ElevenLabsScribeError(
                f"Scribe word token at index {index} is missing start/end."
            )
        if diarize and not str(item.get("speaker_id") or "").strip():
            raise ElevenLabsScribeError(
                f"Scribe word token at index {index} is missing speaker_id."
            )

    if require_word_tokens and word_token_count == 0:
        raise ElevenLabsScribeError(
            "Scribe response contained no type=word tokens for non-empty audio."
        )
    return raw


class ElevenLabsScribeClient:
    def __init__(
        self,
        *,
        settings: ElevenLabsScribeSettings,
        boundary_factory: Any | None = None,
    ) -> None:
        self.settings = settings
        self._boundary_factory = boundary_factory or (lambda: f"scribe-{uuid4().hex}")

    def _headers(self, *, boundary: str) -> dict[str, str]:
        return {
            "xi-api-key": self.settings.api_key,
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

    def transcribe(
        self,
        *,
        source_url: str,
        options: ScribeRequestOptions | None = None,
        require_word_tokens: bool = True,
        **_metadata: Any,
    ) -> ScribeTranscript:
        if options is None:
            options = ScribeRequestOptions(
                num_speakers=getattr(self.settings, "num_speakers", None),
                diarization_threshold=getattr(
                    self.settings, "diarization_threshold", None
                ),
                keyterms=list(getattr(self.settings, "keyterms", ()) or []) or None,
                seed=getattr(self.settings, "seed", None),
            )
        fields = _build_request_fields(
            source_url=source_url,
            settings=self.settings,
            options=options,
        )
        boundary = str(self._boundary_factory())
        body = _multipart_body(fields, boundary=boundary)
        headers = self._headers(boundary=boundary)
        url = self.settings.endpoint_url

        def _do_request() -> ScribeTranscript:
            request_started = time.perf_counter()
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.settings.timeout_s) as resp:
                    raw_body = resp.read().decode("utf-8")
                    status = resp.status
                    request_id = resp.headers.get("request-id") or resp.headers.get(
                        "x-request-id"
                    )
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:  # pragma: no cover
                    err_body = ""
                raise ElevenLabsScribeError(
                    f"Scribe returned HTTP {exc.code}: {err_body[:512] or exc.reason}",
                    status_code=exc.code,
                ) from exc

            if status >= 400:
                raise ElevenLabsScribeError(
                    f"Scribe returned HTTP {status}",
                    status_code=status,
                )
            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise ElevenLabsScribeError(
                    f"Scribe returned non-JSON body: {exc}"
                ) from exc
            validated = validate_scribe_response(
                parsed,
                diarize=self.settings.diarize,
                require_word_tokens=require_word_tokens,
            )
            words = validated.get("words") or []
            word_items = [
                item
                for item in words
                if isinstance(item, dict) and item.get("type") == "word"
            ]
            tag_items = [
                item
                for item in words
                if isinstance(item, dict) and item.get("type") != "word"
            ]
            speakers = {
                str(item.get("speaker_id"))
                for item in word_items
                if str(item.get("speaker_id") or "").strip()
            }
            metrics = {
                "request_mode": "signed_gcs_url",
                "url_field": getattr(self.settings, "url_field", "source_url"),
                "wall_ms": (time.perf_counter() - request_started) * 1000.0,
                "word_count": len(word_items),
                "speaker_count": len(speakers),
                "audio_tag_count": len(tag_items),
                "language_code": validated.get("language_code"),
                "language_probability": validated.get("language_probability"),
                "elevenlabs_request_id": request_id,
            }
            return ScribeTranscript(raw=validated, metrics=metrics)

        return retry_with_backoff(
            _do_request,
            max_retries=self.settings.max_retries,
            classify_transient=_is_transient,
            log_prefix="[elevenlabs_scribe]",
            operation="speech_to_text",
            sleep=time.sleep,
            rng=random.random,
        )


__all__ = [
    "ElevenLabsScribeClient",
    "ElevenLabsScribeError",
    "ElevenLabsScribeSettings",
    "ScribeRequestOptions",
    "ScribeTranscript",
    "validate_scribe_response",
]
