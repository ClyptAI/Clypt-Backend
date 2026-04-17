from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from typing import Any

from .config import _normalize_hotwords_context

logger = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv"}

_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}

_SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]

_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
)

_FAILURE_DIR_ENV = "CLYPT_VIBEVOICE_FAILURE_DIR"
_DEFAULT_FAILURE_DIR = "backend/outputs/vibevoice_failures"


class VibeVoiceVLLMProvider:
    """
    VibeVoice ASR via a persistent local vLLM service.

    Sends audio to a Docker-managed vLLM server over localhost HTTP using the
    OpenAI-compatible ``/v1/chat/completions`` endpoint with streaming.
    Default transport is signed canonical GCS URL mode.
    Inline base64 mode is supported only when explicitly configured.
    Video files (MP4 etc.) are extracted to MP3 before sending.
    No native subprocess, no HF in-process loading, no URL fallback — fail fast.

    Outputs: ``[{"Start": float, "End": float, "Speaker": int, "Content": str}, ...]``
    """

    # Signals to extract.py that ASR calls an HTTP service (not the GPU),
    # so it can safely overlap with visual extraction.
    supports_concurrent_visual: bool = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "vibevoice",
        timeout_s: float = 7200.0,
        healthcheck_path: str = "/health",
        max_retries: int = 1,
        audio_mode: str = "url",
        audio_gcs_url_resolver: Callable[[str], str] | None = None,
        hotwords_context: str | None = None,
        max_new_tokens: int = 32768,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.03,
        num_beams: int = 1,
    ) -> None:
        if not base_url:
            raise RuntimeError(
                "VIBEVOICE_VLLM_BASE_URL is required when VIBEVOICE_BACKEND=vllm."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.healthcheck_path = healthcheck_path
        self.max_retries = max(0, int(max_retries))
        normalized_audio_mode = (audio_mode or "url").strip().lower()
        if normalized_audio_mode in {"data_url", "inline"}:
            normalized_audio_mode = "base64"
        if normalized_audio_mode not in {"url", "base64"}:
            raise ValueError(
                f"Unsupported VibeVoice audio_mode={audio_mode!r}; expected 'url' or 'base64'."
            )
        self.audio_mode = normalized_audio_mode
        self.audio_gcs_url_resolver = audio_gcs_url_resolver
        self.hotwords_context = _normalize_hotwords_context(hotwords_context)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams

    def load(self) -> None:
        """Health-check the vLLM service at startup. Fails fast if unavailable."""
        self.health_check()

    def health_check(self) -> None:
        """Assert the vLLM service is reachable. Raises RuntimeError on failure."""
        url = self.base_url + self.healthcheck_path
        logger.info("[vibevoice-vllm] health check → %s", url)
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"[vibevoice-vllm] health check failed: HTTP {exc.code} from {url}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"[vibevoice-vllm] health check failed — cannot reach {url}: {exc}"
            ) from exc

        if status not in (200, 204):
            raise RuntimeError(
                f"[vibevoice-vllm] health check: unexpected status {status} from {url}"
            )
        logger.info("[vibevoice-vllm] service healthy (HTTP %d)", status)

    def run(
        self,
        audio_path: str | Path,
        context_info: str | None = None,
        audio_gcs_uri: str | None = None,
    ) -> list[dict[str, Any]]:
        audio_path = Path(audio_path)
        context = context_info if context_info is not None else self.hotwords_context

        audio_for_req, is_temp = self._extract_audio_if_needed(audio_path)
        try:
            duration_s = self._probe_duration(audio_for_req)
            audio_url = self._resolve_audio_url(audio_gcs_uri=audio_gcs_uri)
            logger.info(
                "[vibevoice-vllm] ASR request: model=%s url=%s audio=%s (%.1f s) context=%d chars mode=%s",
                self.model,
                self.base_url,
                audio_path.name,
                duration_s,
                len(context),
                "url" if audio_url else "base64",
            )
            t0 = time.perf_counter()
            turns = self._request_with_retry(audio_for_req, context, duration_s, audio_url)
            elapsed = time.perf_counter() - t0
        finally:
            if is_temp:
                try:
                    audio_for_req.unlink()
                except Exception:
                    pass

        rtf = elapsed / duration_s if duration_s > 0 else 0.0
        logger.info(
            "[vibevoice-vllm] done in %.1f s — %d turns (RTF %.2fx)",
            elapsed,
            len(turns),
            rtf,
        )
        return turns

    def _extract_audio_if_needed(self, audio_path: Path) -> tuple[Path, bool]:
        """Extract video files to MP3. Returns (path_to_use, is_temp)."""
        if audio_path.suffix.lower() not in _VIDEO_EXTS:
            return audio_path, False
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                tmp,
            ],
            check=True,
            capture_output=True,
        )
        logger.info("[vibevoice-vllm] extracted audio to %s", tmp)
        return Path(tmp), True

    def _probe_duration(self, audio_path: Path) -> float:
        """Probe audio duration via ffprobe.

        Raises on ffprobe failure rather than returning 0.0 — a silent zero
        poisons the RTF telemetry downstream, so Phase 1 prefers a hard crash
        over a quietly wrong duration.
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)

    def _request_with_retry(
        self, audio_path: Path, context: str, duration_s: float, audio_url: str | None
    ) -> list[dict[str, Any]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.warning(
                    "[vibevoice-vllm] retry %d/%d after transient error",
                    attempt,
                    self.max_retries,
                )
            try:
                return self._do_request(audio_path, context, duration_s, audio_url)
            except RuntimeError:
                # Contract/parse errors are not transient — don't retry.
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[vibevoice-vllm] request error (attempt %d): %s", attempt + 1, exc
                )

        raise RuntimeError(
            f"[vibevoice-vllm] all {self.max_retries + 1} attempt(s) failed: {last_exc}"
        ) from last_exc

    def _do_request(
        self, audio_path: Path, context: str, duration_s: float, audio_url: str | None
    ) -> list[dict[str, Any]]:
        payload = self._build_payload(audio_path, context, duration_s, audio_url=audio_url)
        body = json.dumps(payload).encode("utf-8")
        url = self.base_url + "/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            content_parts: list[str] = []
            finish_reason: str | None = None
            completion_tokens: int | None = None
            chunk_count = 0
            ignored_chunks = 0
            saw_done = False
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:]
                    if json_str.strip() == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(json_str)
                        chunk_count += 1
                        choice = chunk["choices"][0]
                        delta_content = choice.get("delta", {}).get("content", "")
                        if delta_content:
                            content_parts.append(delta_content)
                        chunk_finish_reason = choice.get("finish_reason")
                        if chunk_finish_reason:
                            finish_reason = str(chunk_finish_reason)
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            raw_completion_tokens = usage.get("completion_tokens")
                            if isinstance(raw_completion_tokens, int):
                                completion_tokens = raw_completion_tokens
                    except (json.JSONDecodeError, KeyError, IndexError):
                        ignored_chunks += 1
        except urllib.error.HTTPError as exc:
            body_snippet = ""
            try:
                body_snippet = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                pass
            raise RuntimeError(
                f"[vibevoice-vllm] HTTP {exc.code} from {url}: {body_snippet}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"[vibevoice-vllm] request to {url} failed: {exc}"
            ) from exc

        content = "".join(content_parts)
        generated_chars = len(content)
        approx_generated_tokens = generated_chars // 4 if generated_chars else 0
        logger.info(
            "[vibevoice-vllm] stream finished: finish_reason=%s generated_chars=%d "
            "completion_tokens=%s approx_generated_tokens=%d chunks=%d ignored_chunks=%d saw_done=%s",
            finish_reason or "unknown",
            generated_chars,
            completion_tokens if completion_tokens is not None else "unknown",
            approx_generated_tokens,
            chunk_count,
            ignored_chunks,
            saw_done,
        )

        return self._parse_content(
            content,
            finish_reason=finish_reason,
            generated_chars=generated_chars,
            generated_tokens=completion_tokens,
            approx_generated_tokens=approx_generated_tokens,
            chunk_count=chunk_count,
            saw_done=saw_done,
        )

    def _build_payload(
        self, audio_path: Path, context: str, duration_s: float, *, audio_url: str | None = None
    ) -> dict[str, Any]:
        if self.audio_mode == "url":
            if not audio_url:
                raise RuntimeError(
                    "[vibevoice-vllm] audio_mode=url requires a signed canonical audio URL."
                )
            audio_part: dict[str, Any] = {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            }
        else:
            audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            mime = _MIME_MAP.get(audio_path.suffix.lower(), "audio/wav")
            audio_part = {
                "type": "audio_url",
                "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
            }

        if context.strip():
            prompt_text = (
                f"This is a {duration_s:.2f} seconds audio, "
                f"with extra info: {context.strip()}\n\n"
                f"Please transcribe it with these keys: " + ", ".join(_SHOW_KEYS)
            )
        else:
            prompt_text = (
                f"This is a {duration_s:.2f} seconds audio, "
                f"please transcribe it with these keys: " + ", ".join(_SHOW_KEYS)
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        audio_part,
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
            "stream": True,
            "top_p": self.top_p if self.do_sample else 1.0,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.num_beams > 1:
            payload["use_beam_search"] = True
            payload["best_of"] = self.num_beams
        return payload

    def _resolve_audio_url(self, *, audio_gcs_uri: str | None = None) -> str | None:
        if self.audio_mode != "url":
            return None

        if not audio_gcs_uri:
            raise RuntimeError(
                "[vibevoice-vllm] audio_mode=url requires audio_gcs_uri from canonical test-bank mapping."
            )
        if audio_gcs_uri.startswith(("https://", "http://")):
            return audio_gcs_uri
        if self.audio_gcs_url_resolver is None:
            raise RuntimeError(
                "[vibevoice-vllm] audio_mode=url received gs:// audio_gcs_uri but no signer is configured."
            )
        try:
            resolved = self.audio_gcs_url_resolver(audio_gcs_uri)
        except Exception as exc:
            raise RuntimeError(
                f"[vibevoice-vllm] failed to sign canonical audio_gcs_uri={audio_gcs_uri!r}: {exc}"
            ) from exc
        if not resolved:
            raise RuntimeError(
                f"[vibevoice-vllm] signer returned empty URL for canonical audio_gcs_uri={audio_gcs_uri!r}."
            )
        return resolved

    def _parse_content(
        self,
        content: str,
        *,
        finish_reason: str | None = None,
        generated_chars: int | None = None,
        generated_tokens: int | None = None,
        approx_generated_tokens: int | None = None,
        chunk_count: int | None = None,
        saw_done: bool | None = None,
    ) -> list[dict[str, Any]]:
        generated_chars = generated_chars if generated_chars is not None else len(content)
        approx_generated_tokens = (
            approx_generated_tokens
            if approx_generated_tokens is not None
            else (generated_chars // 4 if generated_chars else 0)
        )
        try:
            raw_turns = json.loads(content)
        except json.JSONDecodeError as exc:
            artifact_path = self._persist_failed_content(
                content=content,
                finish_reason=finish_reason,
                generated_chars=generated_chars,
                generated_tokens=generated_tokens,
                approx_generated_tokens=approx_generated_tokens,
                chunk_count=chunk_count,
                saw_done=saw_done,
            )
            raise RuntimeError(
                f"[vibevoice-vllm] content is not parseable as turns: {exc}; "
                f"finish_reason={finish_reason or 'unknown'} "
                f"generated_chars={generated_chars} "
                f"completion_tokens={generated_tokens if generated_tokens is not None else 'unknown'} "
                f"approx_generated_tokens={approx_generated_tokens} "
                f"chunks={chunk_count if chunk_count is not None else 'unknown'} "
                f"saw_done={saw_done if saw_done is not None else 'unknown'} "
                f"artifact={artifact_path}"
            ) from exc

        if not isinstance(raw_turns, list):
            raise RuntimeError(
                f"[vibevoice-vllm] expected list of turns, got {type(raw_turns).__name__}"
            )

        if finish_reason not in (None, "", "stop"):
            artifact_path = self._persist_failed_content(
                content=content,
                finish_reason=finish_reason,
                generated_chars=generated_chars,
                generated_tokens=generated_tokens,
                approx_generated_tokens=approx_generated_tokens,
                chunk_count=chunk_count,
                saw_done=saw_done,
            )
            raise RuntimeError(
                f"[vibevoice-vllm] stream ended with finish_reason={finish_reason}; "
                f"refusing possibly truncated ASR output. "
                f"generated_chars={generated_chars} "
                f"completion_tokens={generated_tokens if generated_tokens is not None else 'unknown'} "
                f"approx_generated_tokens={approx_generated_tokens} "
                f"chunks={chunk_count if chunk_count is not None else 'unknown'} "
                f"saw_done={saw_done if saw_done is not None else 'unknown'} "
                f"artifact={artifact_path}"
            )

        turns = self._normalize_turns(raw_turns)
        if not turns:
            raise RuntimeError(
                "[vibevoice-vllm] zero usable turns in response — "
                "the audio may be silent or the model output is malformed."
            )
        return turns

    def _persist_failed_content(
        self,
        *,
        content: str,
        finish_reason: str | None,
        generated_chars: int,
        generated_tokens: int | None,
        approx_generated_tokens: int,
        chunk_count: int | None,
        saw_done: bool | None,
    ) -> str:
        base_dir = Path(os.getenv(_FAILURE_DIR_ENV, _DEFAULT_FAILURE_DIR))
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp_ms = int(time.time() * 1000)
        stem = f"vibevoice_{timestamp_ms}_{os.getpid()}"
        content_path = base_dir / f"{stem}.txt"
        metadata_path = base_dir / f"{stem}.meta.json"
        content_path.write_text(content, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "base_url": self.base_url,
                    "model": self.model,
                    "finish_reason": finish_reason,
                    "generated_chars": generated_chars,
                    "completion_tokens": generated_tokens,
                    "approx_generated_tokens": approx_generated_tokens,
                    "chunk_count": chunk_count,
                    "saw_done": saw_done,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        logger.error(
            "[vibevoice-vllm] persisted failed raw content to %s (metadata=%s)",
            content_path,
            metadata_path,
        )
        return str(content_path)

    def _normalize_turns(self, raw_turns: list[Any]) -> list[dict[str, Any]]:
        normalized = []
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            sp = item.get("Speaker")
            if sp is None:
                sp = item.get("speaker") or item.get("speaker_id")
            normalized.append(
                {
                    "Start": float(
                        item.get("Start")
                        or item.get("start")
                        or item.get("start_time")
                        or 0.0
                    ),
                    "End": float(
                        item.get("End")
                        or item.get("end")
                        or item.get("end_time")
                        or 0.0
                    ),
                    "Speaker": int(sp or 0),
                    "Content": str(
                        item.get("Content")
                        or item.get("content")
                        or item.get("text")
                        or ""
                    ).strip(),
                }
            )
        return normalized

    def teardown(self) -> None:
        pass  # nothing to clean up — the service is persistent


def build_gcs_uri_url_resolver(
    *,
    storage_client: Any,
    signed_url_expiry_hours: int = 6,
) -> Callable[[str], str]:
    def _resolver(gcs_uri: str) -> str:
        return storage_client.get_https_url(gcs_uri, expiry_hours=signed_url_expiry_hours)

    return _resolver


__all__ = [
    "VibeVoiceVLLMProvider",
    "build_gcs_uri_url_resolver",
]
