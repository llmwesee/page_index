from __future__ import annotations

import time
import random
from typing import Any, Callable, Dict, Optional

from openai import AzureOpenAI


class AzureLLMError(RuntimeError):
    """Raised when Azure OpenAI calls fail."""


class AzureLLMClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        deployment_name: str,
        timeout_sec: int = 45,
        max_retries: int = 6,
        backoff_sec: float = 1.0,
    ) -> None:
        if not api_key or not endpoint:
            raise ValueError("Missing Azure OpenAI credentials")

        self.deployment_name = deployment_name
        self.max_retries = max(1, max_retries)
        self.backoff_sec = max(0.1, backoff_sec)
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=timeout_sec,
        )

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _extract_usage(cls, usage_obj: Any) -> Dict[str, int]:
        if usage_obj is None:
            return {}

        usage: Dict[str, int] = {
            "prompt_tokens": cls._as_int(getattr(usage_obj, "prompt_tokens", 0)),
            "completion_tokens": cls._as_int(getattr(usage_obj, "completion_tokens", 0)),
            "total_tokens": cls._as_int(getattr(usage_obj, "total_tokens", 0)),
        }

        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details is not None:
            usage["cached_tokens"] = cls._as_int(getattr(prompt_details, "cached_tokens", 0))

        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        if completion_details is not None:
            usage["reasoning_tokens"] = cls._as_int(getattr(completion_details, "reasoning_tokens", 0))

        return usage

    def complete_with_usage(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1600,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                return {
                    "text": (content or "").strip(),
                    "usage": self._extract_usage(getattr(response, "usage", None)),
                    "model": str(getattr(response, "model", "") or self.deployment_name),
                }
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # detect rate limit / 429 errors from exception text or attributes
                is_rate_limit = False
                try:
                    msg = str(exc)
                    if "429" in msg or "RateLimit" in msg or "RateLimitReached" in msg:
                        is_rate_limit = True
                except Exception:
                    pass

                if attempt >= self.max_retries or not is_rate_limit:
                    break

                # exponential backoff with jitter for rate limit responses
                sleep_for = self.backoff_sec * (2 ** (attempt - 1))
                # add small jitter
                sleep_for += random.uniform(0, min(1.0, sleep_for))
                time.sleep(sleep_for)

        raise AzureLLMError(f"Azure LLM request failed after retries: {last_exc}")

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1600,
    ) -> str:
        result = self.complete_with_usage(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(result.get("text", "")).strip()

    def stream_complete_with_usage(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            parts = []
            usage: Dict[str, int] = {}
            model_name = self.deployment_name
            try:
                stream = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    chunk_usage = self._extract_usage(getattr(chunk, "usage", None))
                    if chunk_usage:
                        usage = chunk_usage
                    model_name = str(getattr(chunk, "model", "") or model_name)
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if not content:
                        continue
                    if isinstance(content, str):
                        text = content
                    else:
                        text = "".join(
                            item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                            for item in content
                        )
                    if not text:
                        continue
                    parts.append(text)
                    if on_text:
                        on_text(text)
                return {
                    "text": "".join(parts).strip(),
                    "usage": usage,
                    "model": model_name,
                }
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                is_rate_limit = False
                try:
                    msg = str(exc)
                    if "429" in msg or "RateLimit" in msg or "RateLimitReached" in msg:
                        is_rate_limit = True
                except Exception:
                    pass

                if attempt >= self.max_retries or not is_rate_limit:
                    break

                sleep_for = self.backoff_sec * (2 ** (attempt - 1))
                sleep_for += random.uniform(0, min(1.0, sleep_for))
                time.sleep(sleep_for)

        raise AzureLLMError(f"Azure LLM stream request failed after retries: {last_exc}")

    def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> str:
        result = self.stream_complete_with_usage(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            on_text=on_text,
        )
        return str(result.get("text", "")).strip()

