from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


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


def _coerce_local_video_path(
    *,
    source_url: str,
    entry: Any,
    mapping_path: Path,
) -> Path:
    if isinstance(entry, str):
        local_video_path = entry
    elif isinstance(entry, dict):
        local_video_path = entry.get("local_video_path")
    else:
        local_video_path = None
    if not isinstance(local_video_path, str) or not local_video_path.strip():
        raise Phase1InputResolutionError(
            f"Mapping for {source_url!r} in {mapping_path.name!r} must define local_video_path"
        )
    candidate = Path(local_video_path)
    if not candidate.is_absolute():
        candidate = (mapping_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise Phase1InputResolutionError(
            f"Mapped local video path does not exist for {source_url!r}: {candidate}"
        )
    return candidate


@dataclass(frozen=True, slots=True)
class Phase1InputResolver:
    mapping_path: Path
    _url_to_local_path: dict[str, Path]

    @classmethod
    def from_mapping_file(cls, mapping_path: str | Path) -> "Phase1InputResolver":
        mapping_path = Path(mapping_path).expanduser().resolve()
        if not mapping_path.exists():
            raise Phase1InputResolutionError(f"Input mapping file does not exist: {mapping_path}")
        document = _load_mapping_document(mapping_path)
        url_to_local_path: dict[str, Path] = {}
        for source_url, entry in document.items():
            if not isinstance(source_url, str) or not source_url.strip():
                raise Phase1InputResolutionError("All mapping keys must be non-empty source URLs")
            url_to_local_path[source_url] = _coerce_local_video_path(
                source_url=source_url,
                entry=entry,
                mapping_path=mapping_path,
            )
        return cls(mapping_path=mapping_path, _url_to_local_path=url_to_local_path)

    def resolve_source_path(self, *, source_url: str) -> Path:
        try:
            return self._url_to_local_path[source_url]
        except KeyError as exc:
            raise Phase1InputResolutionError(
                f"No test-bank mapping found for source_url={source_url!r}"
            ) from exc


__all__ = [
    "Phase1InputResolutionError",
    "Phase1InputResolver",
]
