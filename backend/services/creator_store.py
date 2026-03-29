from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileCreatorStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_profile(self, creator_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        payload = dict(profile)
        payload.setdefault("creator_id", creator_id)
        self._write_json(self._profile_path(creator_id), payload)
        return payload

    def get_profile(self, creator_id: str) -> dict[str, Any] | None:
        return self._read_json(self._profile_path(creator_id))

    def save_preferences(self, creator_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        payload = dict(preferences)
        payload.setdefault("creator_id", creator_id)
        self._write_json(self._preferences_path(creator_id), payload)
        return payload

    def get_preferences(self, creator_id: str) -> dict[str, Any] | None:
        return self._read_json(self._preferences_path(creator_id))

    def _creator_dir(self, creator_id: str) -> Path:
        path = self.root / creator_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _profile_path(self, creator_id: str) -> Path:
        return self._creator_dir(creator_id) / "profile.json"

    def _preferences_path(self, creator_id: str) -> Path:
        return self._creator_dir(creator_id) / "preferences.json"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
