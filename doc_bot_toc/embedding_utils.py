from __future__ import annotations

import random
import time
from typing import List, Optional, Sequence


class AzureEmbeddingError(RuntimeError):
    """Raised when Azure embeddings configuration or requests fail."""


class AzureEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        api_version: str,
        deployment_name: str,
        dimensions: Optional[int] = None,
        max_retries: int = 6,
        backoff_sec: float = 1.0,
    ) -> None:
        if not api_key or not endpoint or not deployment_name:
            raise ValueError("Missing Azure embedding credentials or deployment")

        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except Exception as exc:  # noqa: BLE001
            raise AzureEmbeddingError("langchain-openai is required for embeddings") from exc

        self.deployment_name = deployment_name
        self.dimensions = dimensions
        self.max_retries = max(1, int(max_retries))
        self.backoff_sec = max(0.1, float(backoff_sec))
        kwargs = {
            "api_key": api_key,
            "azure_endpoint": endpoint,
            "openai_api_version": api_version,
            "azure_deployment": deployment_name,
        }
        if dimensions:
            kwargs["dimensions"] = dimensions
        self.client = AzureOpenAIEmbeddings(**kwargs)

    def embed_query(self, text: str) -> List[float]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return list(self.client.embed_query(text or ""))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries or not self._is_rate_limit(exc):
                    break
                self._sleep_for_retry(attempt)
        raise AzureEmbeddingError(f"Failed to embed query: {last_exc}") from last_exc

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                vectors = self.client.embed_documents([str(text or "") for text in texts])
                return [list(vector) for vector in vectors]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries or not self._is_rate_limit(exc):
                    break
                self._sleep_for_retry(attempt)
        raise AzureEmbeddingError(f"Failed to embed texts: {last_exc}") from last_exc

    def _is_rate_limit(self, exc: Exception) -> bool:
        message = str(exc)
        return "429" in message or "RateLimit" in message or "RateLimitReached" in message

    def _sleep_for_retry(self, attempt: int) -> None:
        sleep_for = self.backoff_sec * (2 ** (attempt - 1))
        sleep_for += random.uniform(0, min(1.0, sleep_for))
        time.sleep(sleep_for)
