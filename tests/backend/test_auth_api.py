from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.onboarding_jobs import OnboardingJobStore
from backend.services.auth_store import FileAuthStore
from backend.services.creator_store import FileCreatorStore


def _temp_root() -> Path:
    root = Path(".pytest-local-temp") / f"auth-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_client(root: Path) -> TestClient:
    return TestClient(
        create_app(
            job_store=OnboardingJobStore(root / "jobs"),
            creator_store=FileCreatorStore(root / "creators"),
            auth_store=FileAuthStore(root / "auth", session_ttl_hours=24),
        )
    )


def test_signup_returns_token_and_creates_creator_shell():
    tmp_path = _temp_root()
    try:
        client = _build_client(tmp_path)

        response = client.post(
            "/api/v1/auth/signup",
            json={
                "email": "founder@example.com",
                "password": "supersecure123",
                "display_name": "Founder",
            },
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["token"]
        assert payload["user"]["email"] == "founder@example.com"
        assert payload["user"]["display_name"] == "Founder"
        creator_id = payload["user"]["creator_id"]

        me_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {payload['token']}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["user"]["creator_id"] == creator_id

        profile_response = client.get(f"/api/v1/creators/{creator_id}/profile")
        assert profile_response.status_code == 200
        profile = profile_response.json()
        assert profile["creator_id"] == creator_id
        assert profile["email"] == "founder@example.com"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_duplicate_signup_is_rejected():
    tmp_path = _temp_root()
    try:
        client = _build_client(tmp_path)

        first = client.post(
            "/api/v1/auth/signup",
            json={"email": "team@example.com", "password": "supersecure123"},
        )
        assert first.status_code == 201

        second = client.post(
            "/api/v1/auth/signup",
            json={"email": "team@example.com", "password": "anothersecure123"},
        )

        assert second.status_code == 409
        assert "already exists" in second.json()["detail"].lower()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_login_returns_new_session_token():
    tmp_path = _temp_root()
    try:
        client = _build_client(tmp_path)
        signup = client.post(
            "/api/v1/auth/signup",
            json={"email": "operator@example.com", "password": "supersecure123"},
        )
        assert signup.status_code == 201

        login = client.post(
            "/api/v1/auth/login",
            json={"email": "operator@example.com", "password": "supersecure123"},
        )

        assert login.status_code == 200
        payload = login.json()
        assert payload["token"]
        assert payload["user"]["email"] == "operator@example.com"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_logout_revokes_session():
    tmp_path = _temp_root()
    try:
        client = _build_client(tmp_path)
        signup = client.post(
            "/api/v1/auth/signup",
            json={"email": "logout@example.com", "password": "supersecure123"},
        )
        assert signup.status_code == 201
        token = signup.json()["token"]

        logout = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert logout.status_code == 200
        assert logout.json()["logged_out"] is True

        me = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me.status_code == 401
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
