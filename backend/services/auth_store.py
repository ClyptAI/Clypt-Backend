from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class DuplicateEmailError(ValueError):
    pass


class InvalidCredentialsError(ValueError):
    pass


class InvalidTokenError(ValueError):
    pass


class FileAuthStore:
    def __init__(self, root: str | Path, *, session_ttl_hours: int = 24 * 30) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session_ttl = timedelta(hours=max(1, session_ttl_hours))
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_user(self, *, email: str, password: str, display_name: str = "") -> dict[str, Any]:
        normalized_email = self._normalize_email(email)
        normalized_name = str(display_name or "").strip()
        self._validate_signup(normalized_email, password)

        existing_user_id = self._email_index().get(normalized_email)
        if existing_user_id:
            raise DuplicateEmailError("An account with that email already exists.")

        now = self._now()
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        creator_id = f"creator_{uuid.uuid4().hex[:12]}"
        user = {
            "user_id": user_id,
            "creator_id": creator_id,
            "email": normalized_email,
            "display_name": normalized_name,
            "password_hash": self._hash_password(password),
            "created_at": now,
            "updated_at": now,
        }
        self._write_json(self._user_path(user_id), user)

        email_index = self._email_index()
        email_index[normalized_email] = user_id
        self._write_json(self._email_index_path, email_index)
        return self._public_user(user)

    def authenticate(self, *, email: str, password: str) -> dict[str, Any]:
        normalized_email = self._normalize_email(email)
        user_id = self._email_index().get(normalized_email)
        if not user_id:
            raise InvalidCredentialsError("Invalid email or password.")

        user = self._read_json(self._user_path(user_id))
        if not user:
            raise InvalidCredentialsError("Invalid email or password.")

        password_hash = str(user.get("password_hash", "") or "")
        if not password_hash or not bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8")):
            raise InvalidCredentialsError("Invalid email or password.")

        return self._public_user(user)

    def create_session(self, *, user: dict[str, Any]) -> dict[str, Any]:
        user_id = str(user.get("user_id", "") or "").strip()
        if not user_id:
            raise InvalidCredentialsError("Cannot create a session without a valid user.")

        raw_token = secrets.token_urlsafe(32)
        token_hash = self._token_hash(raw_token)
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        now = self._now()
        expires_at = (self._now_datetime() + self.session_ttl).isoformat()
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "creator_id": str(user.get("creator_id", "") or ""),
            "token_hash": token_hash,
            "created_at": now,
            "expires_at": expires_at,
        }
        self._write_json(self._session_path(session_id), session)

        session_index = self._session_index()
        session_index[token_hash] = session_id
        self._write_json(self._session_index_path, session_index)
        return {
            "token": raw_token,
            "session_id": session_id,
            "expires_at": expires_at,
        }

    def get_user_for_token(self, token: str) -> dict[str, Any]:
        normalized_token = str(token or "").strip()
        if not normalized_token:
            raise InvalidTokenError("Missing session token.")

        token_hash = self._token_hash(normalized_token)
        session_id = self._session_index().get(token_hash)
        if not session_id:
            raise InvalidTokenError("Invalid or expired session token.")

        session = self._read_json(self._session_path(session_id))
        if not session:
            self._delete_token_hash(token_hash)
            raise InvalidTokenError("Invalid or expired session token.")

        expires_at = self._parse_datetime(str(session.get("expires_at", "") or ""))
        if expires_at is None or expires_at <= self._now_datetime():
            self.revoke_session(normalized_token)
            raise InvalidTokenError("Invalid or expired session token.")

        user_id = str(session.get("user_id", "") or "")
        user = self._read_json(self._user_path(user_id))
        if not user:
            self.revoke_session(normalized_token)
            raise InvalidTokenError("Invalid or expired session token.")

        return self._public_user(user)

    def revoke_session(self, token: str) -> bool:
        normalized_token = str(token or "").strip()
        if not normalized_token:
            return False
        token_hash = self._token_hash(normalized_token)
        session_id = self._session_index().get(token_hash)
        if not session_id:
            return False

        session_path = self._session_path(session_id)
        if session_path.exists():
            session_path.unlink()
        self._delete_token_hash(token_hash)
        return True

    def _delete_token_hash(self, token_hash: str) -> None:
        session_index = self._session_index()
        if token_hash in session_index:
            session_index.pop(token_hash, None)
            self._write_json(self._session_index_path, session_index)

    def _validate_signup(self, email: str, password: str) -> None:
        if not EMAIL_RE.match(email):
            raise ValueError("Enter a valid email address.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long.")

    @staticmethod
    def _hash_password(password: str) -> str:
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        return hashed.decode("utf-8")

    @staticmethod
    def _normalize_email(email: str) -> str:
        return str(email or "").strip().lower()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_user(user: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": str(user.get("user_id", "") or ""),
            "creator_id": str(user.get("creator_id", "") or ""),
            "email": str(user.get("email", "") or ""),
            "display_name": str(user.get("display_name", "") or ""),
            "created_at": str(user.get("created_at", "") or ""),
            "updated_at": str(user.get("updated_at", "") or ""),
        }

    @property
    def _users_dir(self) -> Path:
        return self.root / "users"

    @property
    def _sessions_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def _email_index_path(self) -> Path:
        return self.root / "email_index.json"

    @property
    def _session_index_path(self) -> Path:
        return self.root / "session_index.json"

    def _user_path(self, user_id: str) -> Path:
        return self._users_dir / f"{user_id}.json"

    def _session_path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"

    def _email_index(self) -> dict[str, str]:
        payload = self._read_json(self._email_index_path)
        return payload if isinstance(payload, dict) else {}

    def _session_index(self) -> dict[str, str]:
        payload = self._read_json(self._session_index_path)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _now() -> str:
        return FileAuthStore._now_datetime().isoformat()

    @staticmethod
    def _now_datetime() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
