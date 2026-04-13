from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import dotenv_values, load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore
    dotenv_values = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"

# Load .env once so Streamlit runs can resolve credentials without shell exports.
if load_dotenv is not None and DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)

DOTENV_VALUES = dotenv_values(DOTENV_PATH) if dotenv_values is not None and DOTENV_PATH.exists() else {}


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is not None:
        return value
    dotenv_value = DOTENV_VALUES.get(name)
    if dotenv_value is not None:
        return str(dotenv_value)
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class AppConfig:
    pageindex_backend_url: str
    azure_openai_api_key: Optional[str]
    azure_openai_endpoint: Optional[str]
    azure_openai_api_version: str
    azure_openai_deployment_name: str
    max_tree_nodes_for_prompt: int
    max_tree_summary_chars: int
    max_page_preview_chars: int
    max_page_evidence_chars: int
    request_timeout_sec: int
    max_retries: int
    retry_backoff_sec: float
    retrieval_top_k: int
    hierarchical_child_chunk_words: int
    hierarchical_child_chunk_overlap_words: int
    hierarchical_candidate_pool_size: int
    hierarchical_top_child_chunks: int
    hierarchical_neighbor_chunks: int
    answer_max_tokens: int
    max_upload_size_mb: int
    prepared_doc_cache_size: int
    prepared_doc_cache_ttl_sec: int
    metadata_cache_size: int
    metadata_cache_ttl_sec: int
    session_memory_turn_window: int
    session_memory_char_budget: int
    documents_page_size: int
    max_workspace_documents: int
    tree_preview_doc_limit: int
    enable_document_routing: bool
    document_routing_top_k: int
    document_routing_min_docs: int
    pgvector_enabled: bool
    pgvector_dsn: Optional[str]
    pgvector_document_table: str
    pgvector_section_table: str
    pgvector_embedding_dimensions: int
    enable_section_reranking: bool
    section_rerank_top_k: int
    section_rerank_node_hints: int
    section_rerank_score_bonus: float
    azure_openai_embedding_api_key: Optional[str]
    azure_openai_embedding_endpoint: Optional[str]
    azure_openai_embedding_api_version: str
    azure_openai_embedding_deployment_name: Optional[str]
    enable_langgraph: bool
    langfuse_enabled: bool
    langfuse_public_key: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_host: Optional[str]

    def missing_azure_settings(self) -> List[str]:
        missing: List[str] = []
        if not self.azure_openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        return missing


def load_config() -> AppConfig:
    pgvector_dsn = _env_str("PGVECTOR_DSN")
    langfuse_public_key = _env_str("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = _env_str("LANGFUSE_SECRET_KEY")
    return AppConfig(
        pageindex_backend_url=_env_str("PAGEINDEX_BACKEND_URL", "http://localhost:8000").rstrip("/"),
        azure_openai_api_key=_env_str("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=_env_str("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_version=_env_str("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_openai_deployment_name=_env_str("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1"),
        max_tree_nodes_for_prompt=_env_int("MAX_TREE_NODES_FOR_PROMPT", 240),
        max_tree_summary_chars=_env_int("MAX_TREE_SUMMARY_CHARS", 1064),
        max_page_preview_chars=_env_int("MAX_PAGE_PREVIEW_CHARS", 532),
        max_page_evidence_chars=_env_int("MAX_PAGE_EVIDENCE_CHARS", 5000),
        request_timeout_sec=_env_int("REQUEST_TIMEOUT_SEC", 45),
        max_retries=_env_int("MAX_RETRIES", 3),
        retry_backoff_sec=_env_float("RETRY_BACKOFF_SEC", 0.8),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 6),
        hierarchical_child_chunk_words=_env_int("HIERARCHICAL_CHILD_CHUNK_WORDS", 180),
        hierarchical_child_chunk_overlap_words=_env_int("HIERARCHICAL_CHILD_CHUNK_OVERLAP_WORDS", 45),
        hierarchical_candidate_pool_size=_env_int("HIERARCHICAL_CANDIDATE_POOL_SIZE", 36),
        hierarchical_top_child_chunks=_env_int("HIERARCHICAL_TOP_CHILD_CHUNKS", 10),
        hierarchical_neighbor_chunks=_env_int("HIERARCHICAL_NEIGHBOR_CHUNKS", 1),
        answer_max_tokens=_env_int("ANSWER_MAX_TOKENS", 4800),
        max_upload_size_mb=_env_int("MAX_UPLOAD_SIZE_MB", 500),
        prepared_doc_cache_size=_env_int("PREPARED_DOC_CACHE_SIZE", 128),
        prepared_doc_cache_ttl_sec=_env_int("PREPARED_DOC_CACHE_TTL_SEC", 3600),
        metadata_cache_size=_env_int("METADATA_CACHE_SIZE", 512),
        metadata_cache_ttl_sec=_env_int("METADATA_CACHE_TTL_SEC", 1800),
        session_memory_turn_window=_env_int("SESSION_MEMORY_TURN_WINDOW", 6),
        session_memory_char_budget=_env_int("SESSION_MEMORY_CHAR_BUDGET", 900),
        documents_page_size=_env_int("DOCUMENTS_PAGE_SIZE", 250),
        max_workspace_documents=_env_int("MAX_WORKSPACE_DOCUMENTS", 5000),
        tree_preview_doc_limit=_env_int("TREE_PREVIEW_DOC_LIMIT", 6),
        enable_document_routing=_env_bool("ENABLE_DOCUMENT_ROUTING", True),
        document_routing_top_k=_env_int("DOCUMENT_ROUTING_TOP_K", 8),
        document_routing_min_docs=_env_int("DOCUMENT_ROUTING_MIN_DOCS", 3),
        pgvector_enabled=_env_bool("PGVECTOR_ENABLED", bool(pgvector_dsn)),
        pgvector_dsn=pgvector_dsn,
        pgvector_document_table=_env_str("PGVECTOR_DOCUMENT_TABLE", "document_index"),
        pgvector_section_table=_env_str("PGVECTOR_SECTION_TABLE", "section_index"),
        pgvector_embedding_dimensions=_env_int("PGVECTOR_EMBEDDING_DIMENSIONS", 3072),
        enable_section_reranking=_env_bool("ENABLE_SECTION_RERANKING", True),
        section_rerank_top_k=_env_int("SECTION_RERANK_TOP_K", 24),
        section_rerank_node_hints=_env_int("SECTION_RERANK_NODE_HINTS", 4),
        section_rerank_score_bonus=_env_float("SECTION_RERANK_SCORE_BONUS", 2.25),
        azure_openai_embedding_api_key=(
            _env_str("AZURE_OPENAI_EMBEDDING_API_KEY")
            or _env_str("AZURE_OPENAI_KEY")
            or _env_str("AZURE_OPENAI_API_KEY")
        ),
        azure_openai_embedding_endpoint=(
            _env_str("AZURE_OPENAI_EMBEDDING_ENDPOINT")
            or _env_str("AZURE_OPENAI_ENDPOINT")
        ),
        azure_openai_embedding_api_version=(
            _env_str("AZURE_OPENAI_EMBEDDING_API_VERSION")
            or _env_str("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        ),
        azure_openai_embedding_deployment_name=(
            _env_str("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
            or _env_str("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        ),
        enable_langgraph=_env_bool("ENABLE_LANGGRAPH", True),
        langfuse_enabled=_env_bool("LANGFUSE_ENABLED", bool(langfuse_public_key and langfuse_secret_key)),
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=_env_str("LANGFUSE_HOST") or _env_str("LANGFUSE_BASE_URL"),
    )
