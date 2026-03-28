from __future__ import annotations

import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_SENSO_BASE_URL = "https://apiv2.senso.ai/api/v1"


class SensoAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class SensoPrompt(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_id: str
    text: str
    type: str | None = None


class SensoKBNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    kb_node_id: str | None = None
    id: str | None = None
    type: str
    name: str | None = None
    parent_id: str | None = None


class SensoContentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    title: str
    processing_status: str | None = None
    latest_content_version_id: str | None = None


class SensoSearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    content_id: str
    chunk_text: str
    score: float | int
    title: str | None = None
    chunk_index: int | None = None
    version_id: str | None = None
    vector_id: str | None = None
    rank: int | None = None
    source_type: str | None = None
    content_type: str | None = None


class SensoSearchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    answer: str = ""
    results: list[SensoSearchResult] = Field(default_factory=list)
    total_results: int | None = None
    max_results: int | None = None
    processing_time_ms: int | None = None


class SensoGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    generated_text: str
    content_id: str | None = None
    processing_time_ms: int | None = None
    sources: list[SensoSearchResult] = Field(default_factory=list)


class SensoClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_SENSO_BASE_URL,
        timeout_s: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Senso API key must not be empty.")
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=self.timeout_s)

    @classmethod
    def from_env(cls) -> "SensoClient":
        api_key = str(os.getenv("SENSO_API_KEY", "") or "").strip()
        if not api_key:
            raise RuntimeError("Missing SENSO_API_KEY.")
        base_url = str(os.getenv("SENSO_API_BASE_URL", DEFAULT_SENSO_BASE_URL) or "").strip()
        timeout_s = float(os.getenv("SENSO_TIMEOUT_SECONDS", "30") or 30.0)
        return cls(api_key=api_key, base_url=base_url, timeout_s=timeout_s)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SensoClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> Any:
        response = self._client.request(
            method,
            path,
            headers=self._headers(),
            params=params,
            json=json_payload,
        )
        if response.status_code >= 400:
            payload: Any
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            message = None
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("error")
            if not message:
                message = str(payload)
            raise SensoAPIError(
                f"Senso API request failed ({response.status_code}): {message}",
                status_code=response.status_code,
                payload=payload,
            )
        if response.status_code == 204:
            return None
        return response.json()

    def get_prompt(self, prompt_id: str) -> SensoPrompt:
        payload = self._request("GET", f"/org/prompts/{prompt_id}")
        return SensoPrompt.model_validate(payload)

    def get_kb_root(self) -> SensoKBNode:
        payload = self._request("GET", "/org/kb/root")
        return SensoKBNode.model_validate(payload)

    def create_raw_content(
        self,
        *,
        title: str,
        text: str,
        summary: str | None = None,
        kb_folder_node_id: str | None = None,
        tag_ids: list[str] | None = None,
    ) -> SensoContentRecord:
        body: dict[str, Any] = {
            "title": title,
            "text": text,
        }
        if summary:
            body["summary"] = summary
        if kb_folder_node_id:
            body["kb_folder_node_id"] = kb_folder_node_id
        if tag_ids:
            body["tag_ids"] = tag_ids
        payload = self._request("POST", "/org/kb/raw", json_payload=body)
        return SensoContentRecord.model_validate(payload)

    def search(
        self,
        *,
        query: str,
        max_results: int = 5,
        content_ids: list[str] | None = None,
        require_scoped_ids: bool = False,
        include_answer: bool = True,
    ) -> SensoSearchResponse:
        body: dict[str, Any] = {"query": query, "max_results": max_results}
        if content_ids:
            body["content_ids"] = content_ids
        if require_scoped_ids:
            body["require_scoped_ids"] = True
        path = "/org/search/full" if include_answer else "/org/search/context"
        payload = self._request("POST", path, json_payload=body)
        return SensoSearchResponse.model_validate(payload)

    def wait_for_search_results(
        self,
        *,
        query: str,
        content_ids: list[str],
        max_results: int = 8,
        poll_interval_s: float = 2.0,
        max_wait_s: float = 60.0,
        include_answer: bool = True,
    ) -> SensoSearchResponse:
        deadline = time.monotonic() + max_wait_s
        last_response: SensoSearchResponse | None = None
        while True:
            response = self.search(
                query=query,
                max_results=max_results,
                content_ids=content_ids,
                require_scoped_ids=True,
                include_answer=include_answer,
            )
            last_response = response
            if response.results:
                return response
            if time.monotonic() >= deadline:
                if last_response is not None:
                    return last_response
                raise TimeoutError("Timed out waiting for Senso content to become searchable.")
            time.sleep(poll_interval_s)
