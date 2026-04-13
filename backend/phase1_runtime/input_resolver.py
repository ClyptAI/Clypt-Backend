from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any
from urllib.parse import parse_qs, urlparse


class Phase1InputResolutionError(ValueError):
    pass


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value:
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return ""


def _split_yaml_key_value(line: str) -> tuple[str, str]:
    for idx in range(len(line) - 1, -1, -1):
        if line[idx] != ":":
            continue
        if idx == len(line) - 1 or line[idx + 1].isspace():
            return line[:idx].rstrip(), line[idx + 1 :].strip()
    raise Phase1InputResolutionError(f"Invalid YAML mapping line: {line!r}")


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            key, value = _split_yaml_key_value(stripped)
            if value:
                root[key] = _parse_scalar(value)
                current_key = None
                current_item = None
            else:
                current_item = {}
                root[key] = current_item
                current_key = key
            continue
        if current_key is None or current_item is None:
            raise Phase1InputResolutionError(f"Unexpected YAML indentation: {raw_line!r}")
        nested_key, nested_value = _split_yaml_key_value(stripped)
        current_item[nested_key] = _parse_scalar(nested_value)

    return root


def _load_mapping_document(mapping_path: Path) -> dict[str, Any]:
    text = mapping_path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        if mapping_path.suffix.lower() not in {".yaml", ".yml"}:
            raise Phase1InputResolutionError(
                f"Unsupported mapping file format for {mapping_path.name!r}; expected JSON or YAML"
            ) from None
        document = _load_yaml_mapping(text)
    if not isinstance(document, dict):
        raise Phase1InputResolutionError("Input mapping file must contain a top-level mapping")
    return document


def _resolve_local_path(raw: str, *, mapping_path: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        return (mapping_path.parent / candidate).resolve()
    return candidate.resolve()


def _coerce_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _extract_youtube_video_id(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    if "youtu.be" in host:
        candidate = path.strip("/")
        return candidate or None
    if "youtube.com" in host:
        qs = parse_qs(parsed.query or "")
        if qs.get("v"):
            candidate = qs["v"][0].strip()
            return candidate or None
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2 and segments[0] in {"shorts", "embed", "live"}:
            return segments[1]
    return None


@dataclass(frozen=True, slots=True)
class Phase1SourceAsset:
    source_url: str
    local_video_path: Path
    local_audio_path: Path | None = None
    video_gcs_uri: str | None = None
    audio_gcs_uri: str | None = None


def _coerce_source_asset(
    *,
    source_url: str,
    entry: Any,
    mapping_path: Path,
) -> Phase1SourceAsset:
    local_video_raw: str | None = None
    local_audio_raw: str | None = None
    video_gcs_uri: str | None = None
    audio_gcs_uri: str | None = None

    if isinstance(entry, str):
        local_video_raw = entry
    elif isinstance(entry, dict):
        local_video_raw = _coerce_optional_str(entry.get("local_video_path"))
        local_audio_raw = _coerce_optional_str(entry.get("local_audio_path"))
        video_gcs_uri = _coerce_optional_str(entry.get("video_gcs_uri"))
        audio_gcs_uri = _coerce_optional_str(entry.get("audio_gcs_uri"))

    if not local_video_raw:
        raise Phase1InputResolutionError(
            f"Mapping for {source_url!r} in {mapping_path.name!r} must define local_video_path"
        )

    local_video_path = _resolve_local_path(local_video_raw, mapping_path=mapping_path)
    if not local_video_path.exists() and not video_gcs_uri:
        raise Phase1InputResolutionError(
            f"Mapped local video path does not exist for {source_url!r}: {local_video_path}. "
            "Provide video_gcs_uri to enable cache hydration."
        )

    local_audio_path: Path | None = None
    if local_audio_raw:
        local_audio_path = _resolve_local_path(local_audio_raw, mapping_path=mapping_path)
        if not local_audio_path.exists() and not audio_gcs_uri:
            raise Phase1InputResolutionError(
                f"Mapped local audio path does not exist for {source_url!r}: {local_audio_path}. "
                "Provide audio_gcs_uri to enable cache hydration."
            )

    return Phase1SourceAsset(
        source_url=source_url,
        local_video_path=local_video_path,
        local_audio_path=local_audio_path,
        video_gcs_uri=video_gcs_uri,
        audio_gcs_uri=audio_gcs_uri,
    )


@dataclass(frozen=True, slots=True)
class Phase1InputResolver:
    mapping_path: Path
    _url_to_asset: dict[str, Phase1SourceAsset]
    _video_id_to_asset: dict[str, Phase1SourceAsset]

    @classmethod
    def from_mapping_file(cls, mapping_path: str | Path) -> "Phase1InputResolver":
        mapping_path = Path(mapping_path).expanduser().resolve()
        if not mapping_path.exists():
            raise Phase1InputResolutionError(f"Input mapping file does not exist: {mapping_path}")
        document = _load_mapping_document(mapping_path)

        url_to_asset: dict[str, Phase1SourceAsset] = {}
        video_id_to_asset: dict[str, Phase1SourceAsset] = {}
        for source_url, entry in document.items():
            if not isinstance(source_url, str) or not source_url.strip():
                raise Phase1InputResolutionError("All mapping keys must be non-empty source URLs")
            source_url = source_url.strip()
            asset = _coerce_source_asset(
                source_url=source_url,
                entry=entry,
                mapping_path=mapping_path,
            )
            url_to_asset[source_url] = asset

            video_id = _extract_youtube_video_id(source_url)
            if not video_id:
                continue
            existing = video_id_to_asset.get(video_id)
            if existing and existing.source_url != source_url:
                raise Phase1InputResolutionError(
                    f"Multiple test-bank entries map to youtube_video_id={video_id!r}; "
                    f"keys: {existing.source_url!r}, {source_url!r}. Keep only one canonical URL per video ID."
                )
            video_id_to_asset[video_id] = asset

        return cls(
            mapping_path=mapping_path,
            _url_to_asset=url_to_asset,
            _video_id_to_asset=video_id_to_asset,
        )

    def resolve_source_asset(self, *, source_url: str) -> Phase1SourceAsset:
        exact = self._url_to_asset.get(source_url)
        if exact is not None:
            return exact

        video_id = _extract_youtube_video_id(source_url)
        if video_id:
            by_video_id = self._video_id_to_asset.get(video_id)
            if by_video_id is not None:
                return by_video_id

        raise Phase1InputResolutionError(
            f"No test-bank mapping found for source_url={source_url!r}"
        )

    def resolve_source_path(self, *, source_url: str) -> Path:
        return self.resolve_source_asset(source_url=source_url).local_video_path


__all__ = [
    "Phase1InputResolutionError",
    "Phase1InputResolver",
    "Phase1SourceAsset",
]
