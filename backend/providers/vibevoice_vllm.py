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
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_HOTWORDS = (
    "I, you, he, she, it, we, they, me, him, her, us, them, "
    "my, your, his, hers, its, our, their, mine, yours, ours, theirs, "
    "this, that, these, those, who, whom, whose, which, what, "
    "and, but, or, nor, for, so, yet, after, although, as, because, before, if, since, "
    "that, though, unless, until, when, whenever, where, whereas, while, however, therefore, "
    "moreover, furthermore, also, additionally, meanwhile, consequently, otherwise, nevertheless, "
    "for example, in addition, on the other hand, similarly, likewise, in contrast, thus, hence, "
    "indeed, finally, first, second, third"
)

_VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv"}

_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}

_TURN_SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]
_WORD_SHOW_KEYS = ["start_ms", "end_ms", "speaker_id", "word"]
_VALID_OUTPUT_MODES = {"turns", "words"}

_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output in JSON format."
)


class VibeVoiceVLLMProvider:
    """
    VibeVoice ASR via a persistent local vLLM service.

    Sends audio to a Docker-managed vLLM server over localhost HTTP using the
    OpenAI-compatible ``/v1/chat/completions`` endpoint with streaming.
    Video files (MP4 etc.) are extracted to MP3 before sending.
    No native subprocess, no HF in-process loading, no fallback — fail fast.

    Outputs:
      - ``output_mode=turns``: ``[{"Start": float, "End": float, "Speaker": int, "Content": str}, ...]``
      - ``output_mode=words``: ``[{"start_ms": int, "end_ms": int, "speaker_id": int, "word": str}, ...]``
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
        audio_mode: str = "base64",
        hotwords_context: str | None = None,
        output_mode: str = "turns",
        word_turn_gap_ms: int = 900,
        max_new_tokens: int = 32768,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
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
        self.audio_mode = audio_mode
        self.hotwords_context = (
            hotwords_context if hotwords_context is not None else _DEFAULT_HOTWORDS
        )
        self.output_mode = str(output_mode or "turns").lower()
        if self.output_mode not in _VALID_OUTPUT_MODES:
            raise RuntimeError(
                f"Unsupported VIBEVOICE_OUTPUT_MODE={output_mode!r}; expected one of {_VALID_OUTPUT_MODES}."
            )
        self.word_turn_gap_ms = max(0, int(word_turn_gap_ms))
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
    ) -> list[dict[str, Any]]:
        audio_path = Path(audio_path)
        context = context_info if context_info is not None else self.hotwords_context

        audio_for_req, is_temp = self._extract_audio_if_needed(audio_path)
        try:
            duration_s = self._probe_duration(audio_for_req)
            logger.info(
                "[vibevoice-vllm] ASR request: model=%s url=%s audio=%s (%.1f s) output_mode=%s context=%d chars",
                self.model,
                self.base_url,
                audio_path.name,
                duration_s,
                self.output_mode,
                len(context),
            )
            t0 = time.perf_counter()
            entries = self._request_with_retry(audio_for_req, context, duration_s)
            elapsed = time.perf_counter() - t0
        finally:
            if is_temp:
                try:
                    audio_for_req.unlink()
                except Exception:
                    pass

        item_kind = "words" if self.output_mode == "words" else "turns"
        rtf = elapsed / duration_s if duration_s > 0 else 0.0
        logger.info(
            "[vibevoice-vllm] done in %.1f s — %d %s (RTF %.2fx)",
            elapsed,
            len(entries),
            item_kind,
            rtf,
        )
        return entries

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
        """Probe audio duration via ffprobe. Returns 0.0 on failure."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
            return float(out)
        except Exception:
            return 0.0

    def _request_with_retry(
        self, audio_path: Path, context: str, duration_s: float
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
                return self._do_request(audio_path, context, duration_s)
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
        self, audio_path: Path, context: str, duration_s: float
    ) -> list[dict[str, Any]]:
        payload = self._build_payload(audio_path, context, duration_s)
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
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue
                    json_str = line[6:]
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                        delta_content = chunk["choices"][0]["delta"].get("content", "")
                        if delta_content:
                            content_parts.append(delta_content)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
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

        return self._parse_content("".join(content_parts))

    def _build_payload(
        self, audio_path: Path, context: str, duration_s: float
    ) -> dict[str, Any]:
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        mime = _MIME_MAP.get(audio_path.suffix.lower(), "audio/wav")

        if self.output_mode == "words":
            output_instructions = (
                "Return only a JSON array. Do not wrap in markdown.\n"
                "Each item must be exactly: "
                '{"start_ms": <int>, "end_ms": <int>, "speaker_id": <int>, "word": <string>}.\n'
                "Use one spoken word per item, in chronological order.\n"
                "Times must be milliseconds from start of audio and satisfy start_ms < end_ms."
            )
            keys_hint = ", ".join(_WORD_SHOW_KEYS)
        else:
            output_instructions = (
                "Return only a JSON array. Do not wrap in markdown.\n"
                "Each item must contain these keys: " + ", ".join(_TURN_SHOW_KEYS) + "."
            )
            keys_hint = ", ".join(_TURN_SHOW_KEYS)

        if context.strip():
            prompt_text = (
                f"This is a {duration_s:.2f} seconds audio, "
                f"with extra info: {context.strip()}\n\n"
                f"Please transcribe it with these keys: {keys_hint}\n\n"
                f"{output_instructions}"
            )
        else:
            prompt_text = (
                f"This is a {duration_s:.2f} seconds audio, "
                f"please transcribe it with these keys: {keys_hint}\n\n"
                f"{output_instructions}"
            )

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": True,
            "top_p": 1.0,
        }

    def _parse_content(self, content: str) -> list[dict[str, Any]]:
        try:
            raw_items = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[vibevoice-vllm] content is not parseable as JSON: {exc}\n"
                f"{content[:2000]}"
            ) from exc

        if not isinstance(raw_items, list):
            raise RuntimeError(
                f"[vibevoice-vllm] expected JSON list, got {type(raw_items).__name__}"
            )

        if self.output_mode == "words":
            words = self._normalize_words(raw_items)
            if not words:
                raise RuntimeError(
                    "[vibevoice-vllm] zero usable words in response for output_mode=words."
                )
            return words

        turns = self._normalize_turns(raw_items)
        if not turns:
            raise RuntimeError(
                "[vibevoice-vllm] zero usable turns in response for output_mode=turns."
            )
        return turns

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_speaker_id(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.upper().startswith("SPEAKER_"):
                stripped = stripped.split("_", 1)[1]
            return VibeVoiceVLLMProvider._to_int(stripped, default=0)
        return VibeVoiceVLLMProvider._to_int(value, default=0)

    def _normalize_turns(self, raw_turns: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in raw_turns:
            if not isinstance(item, dict):
                continue
            sp = (
                item.get("Speaker")
                or item.get("speaker")
                or item.get("speaker_id")
                or item.get("Speaker ID")
            )
            start_s = self._to_float(
                item.get("Start")
                or item.get("start")
                or item.get("start_time")
                or item.get("Start time")
                or 0.0
            )
            end_s = self._to_float(
                item.get("End")
                or item.get("end")
                or item.get("end_time")
                or item.get("End time")
                or 0.0
            )
            content = str(
                item.get("Content")
                or item.get("content")
                or item.get("text")
                or item.get("transcript")
                or ""
            ).strip()
            if not content or end_s <= start_s:
                continue
            normalized.append(
                {
                    "Start": start_s,
                    "End": end_s,
                    "Speaker": self._parse_speaker_id(sp),
                    "Content": content,
                }
            )
        normalized.sort(key=lambda t: (t["Start"], t["End"]))
        return normalized

    def _normalize_words(self, raw_words: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            start_ms_raw = (
                item.get("start_ms")
                or item.get("startMs")
                or item.get("StartMs")
                or item.get("start_ms_time")
            )
            end_ms_raw = (
                item.get("end_ms")
                or item.get("endMs")
                or item.get("EndMs")
                or item.get("end_ms_time")
            )
            if start_ms_raw is None:
                start_s_raw = item.get("start") or item.get("Start") or item.get("start_time")
                start_ms = int(round(self._to_float(start_s_raw, default=0.0) * 1000))
            else:
                start_ms = self._to_int(start_ms_raw, default=0)

            if end_ms_raw is None:
                end_s_raw = item.get("end") or item.get("End") or item.get("end_time")
                end_ms = int(round(self._to_float(end_s_raw, default=0.0) * 1000))
            else:
                end_ms = self._to_int(end_ms_raw, default=0)

            speaker = self._parse_speaker_id(
                item.get("speaker_id")
                or item.get("speaker")
                or item.get("Speaker")
                or item.get("Speaker ID")
            )
            word = str(
                item.get("word")
                or item.get("Word")
                or item.get("text")
                or item.get("Text")
                or item.get("token")
                or item.get("Content")
                or ""
            ).strip()
            if not word or end_ms <= start_ms:
                continue
            normalized.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker_id": speaker,
                    "word": word,
                }
            )
        normalized.sort(key=lambda w: (w["start_ms"], w["end_ms"]))
        return normalized

    def teardown(self) -> None:
        pass  # nothing to clean up — the service is persistent


__all__ = ["VibeVoiceVLLMProvider"]
