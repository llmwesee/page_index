#!/usr/bin/env python3
"""Connectivity checks for Azure services used by this project.

Checks:
1) Azure OpenAI chat completion deployment
2) Azure OpenAI embeddings deployment
3) Azure PostgreSQL connectivity
4) Azure Blob Storage container access
5) Azure AI Search endpoint and credentials

Usage:
    .venv\\Scripts\\python.exe test_azure_endpoints.py
    .venv\\Scripts\\python.exe test_azure_endpoints.py --only chat embedding
    .venv\\Scripts\\python.exe test_azure_endpoints.py --json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional

from dotenv import load_dotenv


load_dotenv()
_DOTENV_PATH = Path(__file__).resolve().with_name(".env")


def _parse_dotenv_key_values() -> List[tuple[str, str]]:
    if not _DOTENV_PATH.exists():
        return []

    pairs: List[tuple[str, str]] = []
    for raw in _DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            pairs.append((key, value))
    return pairs


_DOTENV_PAIRS = _parse_dotenv_key_values()


def _dotenv_first(*keys: str) -> Optional[str]:
    wanted = set(keys)
    for key, value in _DOTENV_PAIRS:
        if key in wanted and value:
            return value
    return None


def _normalize_deployment_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    cleaned = value.strip().strip("/")
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    return cleaned


def _normalize_url(value: Optional[str], default_scheme: str = "https://") -> Optional[str]:
    if not value:
        return value
    text = value.strip()
    if not text:
        return None
    if not text.startswith(("http://", "https://")):
        text = f"{default_scheme}{text}"
    return text.rstrip("/")


def _env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip() != "":
            return value.strip()
    return default


def _build_postgres_dsn() -> Optional[str]:
    existing = _env("POSTGRES_DSN", "PGVECTOR_DSN")
    if existing:
        return existing

    host = _env("POSTGRES_ENDPOINT", "POSTGRES_HOST", "postrges_endpoint", "postgres_endpoint")
    user = _env("POSTGRES_USER", "POSTGRES_ADMIN", "admin_login")
    password = _env("POSTGRES_PASSWORD", "POSTGRES_PASS", "admin_pass")
    database = _env("POSTGRES_DB", "POSTGRES_DATABASE", default="postgres")
    port = _env("POSTGRES_PORT", default="5432")
    sslmode = _env("POSTGRES_SSLMODE", default="require")

    if not (host and user and password):
        return None

    return (
        f"host={host} "
        f"port={port} "
        f"dbname={database} "
        f"user={user} "
        f"password={password} "
        f"sslmode={sslmode}"
    )


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_azure_openai_chat() -> CheckResult:
    from openai import AzureOpenAI

    endpoint = _normalize_url(
        _env("AZURE_OPENAI_CHAT_ENDPOINT") or _dotenv_first("AZURE_OPENAI_ENDPOINT") or _env("AZURE_OPENAI_ENDPOINT")
    )
    api_key = _env("AZURE_OPENAI_CHAT_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY")
    api_version = _env("AZURE_OPENAI_CHAT_API_VERSION") or _dotenv_first("AZURE_OPENAI_API_VERSION") or _env(
        "AZURE_OPENAI_API_VERSION", default="2024-12-01-preview"
    )
    deployment = _normalize_deployment_name(
        _env("AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME")
    )

    if not endpoint or not api_key or not deployment:
        return CheckResult(
            name="azure_openai_chat",
            ok=False,
            detail=(
                "Missing one or more required variables: "
                "AZURE_OPENAI_CHAT_ENDPOINT/AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_CHAT_API_KEY/AZURE_OPENAI_API_KEY/AZURE_OPENAI_KEY, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT/AZURE_OPENAI_DEPLOYMENT_NAME"
            ),
        )

    try:
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version, timeout=30.0)
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            max_tokens=8,
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return CheckResult("azure_openai_chat", False, "Chat response was empty")
        return CheckResult("azure_openai_chat", True, f"Received response: {content!r}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("azure_openai_chat", False, f"Request failed: {exc}")


def check_azure_openai_embedding() -> CheckResult:
    from openai import AzureOpenAI

    endpoint = _normalize_url(_env("AZURE_OPENAI_EMBEDDING_ENDPOINT", "AZURE_OPENAI_ENDPOINT"))
    api_key = _env("AZURE_OPENAI_EMBEDDING_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY")
    api_version = _env("AZURE_OPENAI_EMBEDDING_API_VERSION", "AZURE_OPENAI_API_VERSION", default="2024-12-01-preview")
    deployment = _normalize_deployment_name(
        _env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    )

    if not endpoint or not api_key or not deployment:
        return CheckResult(
            name="azure_openai_embedding",
            ok=False,
            detail=(
                "Missing one or more required variables: "
                "AZURE_OPENAI_EMBEDDING_ENDPOINT/AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_EMBEDDING_API_KEY/AZURE_OPENAI_KEY/AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT/AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
            ),
        )

    try:
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version, timeout=30.0)
        response = client.embeddings.create(model=deployment, input="health check")
        vector = response.data[0].embedding if response.data else []
        if not vector:
            return CheckResult("azure_openai_embedding", False, "Embedding response had no vector data")
        return CheckResult("azure_openai_embedding", True, f"Embedding vector length: {len(vector)}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("azure_openai_embedding", False, f"Request failed: {exc}")


def check_azure_postgres() -> CheckResult:
    import psycopg

    dsn = _build_postgres_dsn()
    if not dsn:
        return CheckResult(
            name="azure_postgres",
            ok=False,
            detail=(
                "Missing PostgreSQL settings. Provide POSTGRES_DSN (or PGVECTOR_DSN), "
                "or host/user/password via POSTGRES_* vars (or postrges_endpoint/admin_login/admin_pass)."
            ),
        )

    try:
        with psycopg.connect(dsn, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                value = cur.fetchone()
        if not value or value[0] != 1:
            return CheckResult("azure_postgres", False, f"Unexpected SQL result: {value}")
        return CheckResult("azure_postgres", True, "Successfully connected and executed SELECT 1")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("azure_postgres", False, f"Connection/query failed: {exc}")


def check_azure_blob() -> CheckResult:
    from azure.storage.blob import BlobServiceClient

    connection_string = _env("AZURE_STORAGE_CONNECTION_STRING")
    account_url = _normalize_url(_env("AZURE_STORAGE_ACCOUNT_URL", "AZURE_BLOB_ACCOUNT_URL"))
    account_name = _env("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = _env("AZURE_STORAGE_ACCOUNT_KEY")
    sas_token = _env("AZURE_STORAGE_SAS_TOKEN")
    container_name = _env("AZURE_STORAGE_CONTAINER", "blob_container")

    if not container_name:
        return CheckResult(
            name="azure_blob",
            ok=False,
            detail="Missing container name: AZURE_STORAGE_CONTAINER or blob_container",
        )

    try:
        if connection_string:
            service_client = BlobServiceClient.from_connection_string(connection_string)
        else:
            if not account_url and account_name:
                account_url = _normalize_url(f"{account_name}.blob.core.windows.net")

            if not account_url:
                return CheckResult(
                    name="azure_blob",
                    ok=False,
                    detail=(
                        "Missing blob endpoint. Provide AZURE_STORAGE_CONNECTION_STRING, "
                        "or AZURE_STORAGE_ACCOUNT_URL/AZURE_STORAGE_ACCOUNT_NAME."
                    ),
                )

            credential = account_key or sas_token
            if not credential:
                return CheckResult(
                    name="azure_blob",
                    ok=False,
                    detail="Missing blob credential: AZURE_STORAGE_ACCOUNT_KEY or AZURE_STORAGE_SAS_TOKEN",
                )
            service_client = BlobServiceClient(account_url=account_url, credential=credential)

        container_client = service_client.get_container_client(container_name)
        container_client.get_container_properties()
        return CheckResult("azure_blob", True, f"Container is reachable: {container_name}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("azure_blob", False, f"Blob operation failed: {exc}")


def check_azure_ai_search() -> CheckResult:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    endpoint = _normalize_url(
        _env("AZURE_AI_SEARCH_ENDPOINT", "AZURE_SEARCH_ENDPOINT", "AI_SEARCH_ENDPOINT", "ai_search_endpoint")
    )
    api_key = _env("AZURE_AI_SEARCH_API_KEY", "AZURE_SEARCH_API_KEY", "AI_SEARCH_API_KEY", "ai_search_key")
    index_name = _env("AZURE_AI_SEARCH_INDEX_NAME", "AZURE_SEARCH_INDEX_NAME", "ai_search_index", "AI_SEARCH_INDEX_NAME")

    if not endpoint or not api_key:
        missing_parts: List[str] = []
        if not endpoint:
            missing_parts.append(
                "endpoint (AZURE_AI_SEARCH_ENDPOINT/AZURE_SEARCH_ENDPOINT/AI_SEARCH_ENDPOINT/ai_search_endpoint)"
            )
        if not api_key:
            missing_parts.append("api key (AZURE_AI_SEARCH_API_KEY/AZURE_SEARCH_API_KEY/AI_SEARCH_API_KEY/ai_search_key)")
        return CheckResult(
            name="azure_ai_search",
            ok=False,
            detail=f"Missing required variables: {', '.join(missing_parts)}",
        )

    try:
        credential = AzureKeyCredential(api_key)

        if index_name:
            search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
            results = search_client.search(search_text="*", top=1, include_total_count=True)
            _ = next(iter(results), None)
            count = results.get_count()
            count_text = "unknown" if count is None else str(count)
            return CheckResult(
                "azure_ai_search",
                True,
                f"Search endpoint reachable; index '{index_name}' query succeeded (total_count={count_text})",
            )

        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        names = list(index_client.list_index_names())
        return CheckResult(
            "azure_ai_search",
            True,
            f"Search endpoint reachable; discovered {len(names)} index(es)",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("azure_ai_search", False, f"AI Search operation failed: {exc}")


CHECKS: Dict[str, Callable[[], CheckResult]] = {
    "chat": check_azure_openai_chat,
    "embedding": check_azure_openai_embedding,
    "postgres": check_azure_postgres,
    "blob": check_azure_blob,
    "ai_search": check_azure_ai_search,
}


def run_checks(selected: Iterable[str]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for name in selected:
        results.append(CHECKS[name]())
    return results


def _print_human(results: List[CheckResult]) -> None:
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Azure service endpoints used by this project.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(CHECKS.keys()),
        help="Run only selected checks (chat, embedding, postgres, blob, ai_search)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON results")
    args = parser.parse_args()

    selected = args.only or list(CHECKS.keys())
    results = run_checks(selected)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        _print_human(results)

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
