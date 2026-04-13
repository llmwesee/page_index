from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from streamlit_app.cache import LruTtlCache

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in envs missing requests
    requests = None  # type: ignore


class APIClientError(RuntimeError):
    """Raised when backend API calls fail."""


class PageIndexAPIClient:
    def __init__(
        self,
        base_url: str,
        timeout_sec: int = 45,
        max_retries: int = 3,
        backoff_sec: float = 0.8,
        metadata_cache_size: int = 512,
        metadata_cache_ttl_sec: int = 1800,
        session: Optional[Any] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max(1, max_retries)
        self.backoff_sec = max(0.1, backoff_sec)
        self._metadata_cache: LruTtlCache[str, Dict[str, Any]] = LruTtlCache(
            name="document_metadata",
            max_entries=metadata_cache_size,
            ttl_seconds=metadata_cache_ttl_sec,
        )
        if session is not None:
            self.session = session
        elif requests is not None:
            self.session = requests.Session()
        else:
            raise APIClientError(
                "The 'requests' package is required unless a custom session is supplied."
            )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        request_exception = requests.RequestException if requests is not None else Exception

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_body,
                    data=data,
                    files=files,
                    timeout=self.timeout_sec,
                )
            except request_exception as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt >= self.max_retries:
                    raise APIClientError(f"{method} {url} failed: {exc}") from exc
                time.sleep(self.backoff_sec * attempt)
                continue

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise APIClientError(f"{method} {url} returned non-JSON response") from exc

            retriable = response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
            if retriable and attempt < self.max_retries:
                time.sleep(self.backoff_sec * attempt)
                continue

            raise APIClientError(
                f"{method} {url} failed with {response.status_code}: {response.text}"
            )

        if last_exc is not None:
            raise APIClientError(f"{method} {url} failed: {last_exc}") from last_exc
        raise APIClientError(f"{method} {url} failed unexpectedly")

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/")

    def get_processing_stats(self) -> Dict[str, Any]:
        return self._request("GET", "/processing/stats/")

    def list_documents(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._request("GET", "/docs/", params={"limit": limit, "offset": offset})

    def submit_document(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        if_retrieval: bool = True,
        mode: Optional[str] = None,
        input_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_input_type = (input_type or filename.rsplit(".", 1)[-1]).strip().lower()
        if resolved_input_type not in {"pdf", "json", "xlsx", "csv"}:
            raise APIClientError("Unsupported input type. Expected PDF, JSON, XLSX, or CSV filename/input_type.")

        mime_by_type = {
            "pdf": "application/pdf",
            "json": "application/json",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "csv": "text/csv",
        }
        mime_type = mime_by_type[resolved_input_type]
        data: Dict[str, Any] = {"if_retrieval": str(if_retrieval).lower()}
        data["input_type"] = resolved_input_type
        if mode:
            data["mode"] = mode
        files = {"file": (filename, file_bytes, mime_type)}
        return self._request("POST", "/doc/", data=data, files=files)

    def submit_url(self, url: str, *, if_retrieval: bool = True) -> Dict[str, Any]:
        candidate = (url or "").strip()
        if not candidate:
            raise APIClientError("URL is required.")
        return self._request(
            "POST",
            "/url/",
            json_body={
                "url": candidate,
                "if_retrieval": bool(if_retrieval),
            },
        )

    def submit_url_discovery(
        self,
        url: str,
        *,
        max_urls: int = 25,
        if_retrieval: bool = True,
    ) -> Dict[str, Any]:
        candidate = (url or "").strip()
        if not candidate:
            raise APIClientError("URL is required.")
        return self._request(
            "POST",
            "/urls/discover/",
            json_body={
                "url": candidate,
                "max_urls": int(max_urls),
                "if_retrieval": bool(if_retrieval),
            },
        )

    def submit_urls_bulk(self, urls: List[str], *, if_retrieval: bool = True) -> Dict[str, Any]:
        cleaned = [str(item).strip() for item in (urls or []) if str(item).strip()]
        if not cleaned:
            raise APIClientError("At least one URL is required.")
        items: List[Dict[str, Any]] = []
        accepted = 0
        duplicate = 0
        failed = 0
        for item in cleaned:
            try:
                result = self.submit_url(item, if_retrieval=if_retrieval)
                accepted += 1
                if result.get("duplicate"):
                    duplicate += 1
                items.append({"url": item, **result})
            except Exception as exc:  # noqa: BLE001
                failed += 1
                items.append({"url": item, "status": "failed", "error": str(exc)})
        return {
            "accepted": accepted,
            "duplicate": duplicate,
            "failed": failed,
            "items": items,
        }

    def invalidate_document_metadata(self, doc_id: str) -> None:
        self._metadata_cache.pop(doc_id, None)

    def get_document_metadata(self, doc_id: str, refresh: bool = False) -> Dict[str, Any]:
        if not refresh:
            cached = self._metadata_cache.get(doc_id)
            if cached is not None:
                return dict(cached)
        payload = self._request("GET", f"/doc/{doc_id}/metadata/")
        if payload.get("status") == "completed":
            self._metadata_cache.set(doc_id, dict(payload))
        else:
            self._metadata_cache.pop(doc_id, None)
        return dict(payload)

    def cache_stats(self) -> Dict[str, int]:
        return self._metadata_cache.snapshot()

    def get_tree(self, doc_id: str, summary: bool = True) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/doc/{doc_id}/",
            params={"type": "tree", "summary": str(summary).lower()},
        )

    def get_page_content(self, doc_id: str, pages: str) -> Dict[str, Any]:
        return self._request("GET", f"/doc/{doc_id}/pages/", params={"pages": pages})
