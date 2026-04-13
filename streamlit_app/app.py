from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import streamlit as st

try:
    from streamlit_app.config import AppConfig, load_config
    from streamlit_app.observability import get_logger, log_query_trace
    from streamlit_app.rag.memory import build_session_memory
    from streamlit_app.rag.langgraph_orchestrator import LangGraphQueryOrchestrator
    from streamlit_app.rag.pipeline import RAGPipeline
    from streamlit_app.rag.types import PageContentResult, QueryTrace
    from streamlit_app.services.azure_llm import AzureLLMClient
    from embedding_utils import AzureEmbeddingClient
    from streamlit_app.services.observer import build_observer
    from streamlit_app.services.pageindex_api import APIClientError, PageIndexAPIClient
    from streamlit_app.services.retrieval import RetrievalService
    from streamlit_app.ui.components import (
        render_answer_section,
        render_streaming_answer,
        render_thought_summary,
        render_trace,
        render_tree_section,
    )
except ModuleNotFoundError:
    # Streamlit can execute this file as a script, where package imports may fail
    # if the project root is not on sys.path.
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from streamlit_app.config import AppConfig, load_config
    from streamlit_app.observability import get_logger, log_query_trace
    from streamlit_app.rag.memory import build_session_memory
    from streamlit_app.rag.langgraph_orchestrator import LangGraphQueryOrchestrator
    from streamlit_app.rag.pipeline import RAGPipeline
    from streamlit_app.rag.types import PageContentResult, QueryTrace
    from streamlit_app.services.azure_llm import AzureLLMClient
    from embedding_utils import AzureEmbeddingClient
    from streamlit_app.services.observer import build_observer
    from streamlit_app.services.pageindex_api import APIClientError, PageIndexAPIClient
    from streamlit_app.services.retrieval import RetrievalService
    from streamlit_app.ui.components import (
        render_answer_section,
        render_streaming_answer,
        render_thought_summary,
        render_trace,
        render_tree_section,
    )


def _apply_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
  --bg-start: #f4f7ff;
  --bg-end: #eef9f4;
  --panel: #ffffff;
  --ink: #122032;
  --muted: #5f6d82;
  --line: #d8e1ee;
  --brand: #0f6ab4;
  --brand-soft: #e6f2fd;
    --answer-shell: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    --answer-accent: #0d4f8b;
    --answer-gold: #c78a1d;
    --answer-green: #1d7c57;
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
  color: var(--ink);
}

[data-testid="stAppViewContainer"] {
  background: radial-gradient(1400px 500px at 20% -5%, var(--bg-start) 10%, var(--bg-end) 65%);
}

h1, h2, h3 {
  font-family: "IBM Plex Serif", serif;
  letter-spacing: 0.2px;
}

.stChatMessage, div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 14px;
  border: 1px solid var(--line);
  background: var(--panel);
}

.st-emotion-cache-13ln4jf {
  padding-top: 1rem;
}

.answer-hero {
    padding: 1rem 1.1rem;
    margin: 0.2rem 0 0.8rem 0;
    border: 1px solid rgba(15, 106, 180, 0.14);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(15, 106, 180, 0.08), rgba(15, 106, 180, 0.02) 55%, rgba(29, 124, 87, 0.06));
    box-shadow: 0 10px 30px rgba(18, 32, 50, 0.06);
}

.answer-hero-topline {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--brand);
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.answer-hero-title {
    font-family: "IBM Plex Serif", serif;
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1.15;
    color: var(--ink);
}

.answer-hero-subtitle {
    color: var(--muted);
    margin-top: 0.25rem;
    font-size: 0.98rem;
}

.answer-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.8rem;
}

.answer-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.answer-pill-type {
    background: rgba(15, 106, 180, 0.12);
    color: var(--answer-accent);
}

.answer-pill-confidence {
    background: rgba(29, 124, 87, 0.12);
    color: var(--answer-green);
}

.answer-stat-card {
    padding: 0.8rem 0.9rem;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--answer-shell);
    margin-bottom: 0.8rem;
}

.answer-stat-label {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.answer-stat-value {
    color: var(--ink);
    font-size: 1.02rem;
    font-weight: 700;
}

.answer-source-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 1rem 0 0.2rem 0;
}

.answer-source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.7rem;
    border-radius: 999px;
    border: 1px solid rgba(15, 106, 180, 0.16);
    background: #f8fbff;
    box-shadow: 0 4px 16px rgba(18, 32, 50, 0.04);
}

.answer-source-doc {
    color: var(--ink);
    font-weight: 600;
    font-size: 0.83rem;
}

.answer-source-page {
    color: var(--brand);
    background: rgba(15, 106, 180, 0.09);
    border-radius: 999px;
    padding: 0.18rem 0.45rem;
    font-size: 0.76rem;
    font-weight: 700;
}

.answer-live-status {
    margin: 0.25rem 0 0.8rem 0;
    color: var(--muted);
    font-size: 0.9rem;
}

.chatgpt-thought {
    margin: 0.1rem 0 0.55rem 0;
    border: none;
    padding: 0;
}

.chatgpt-thought summary {
    list-style: none;
    cursor: pointer;
    color: #7f8898;
    display: inline-flex;
    align-items: center;
    gap: 0.22rem;
    font-size: 0.95rem;
    font-weight: 500;
    padding: 0;
}

.chatgpt-thought summary::-webkit-details-marker {
    display: none;
}

.chatgpt-thought-chevron {
    color: #9aa3b2;
    font-size: 1rem;
    transform: rotate(0deg);
    transition: transform 140ms ease;
}

.chatgpt-thought[open] .chatgpt-thought-chevron {
    transform: rotate(90deg);
}

.chatgpt-thought-content {
    margin-top: 0.35rem;
    color: var(--muted);
}

.chatgpt-thought-content ul {
    margin: 0.15rem 0 0.55rem 1rem;
    padding: 0;
}

.chatgpt-thought-content li {
    margin: 0.16rem 0;
    font-size: 0.92rem;
    line-height: 1.32;
}

.chatgpt-citation-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.55rem 0 0.2rem 0;
}

.chatgpt-citation-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #dce5f1;
    background: #f1f5fb;
    color: #445267;
    font-size: 0.74rem;
    font-weight: 600;
}

.chatgpt-citation-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 0.07rem 0.34rem;
    background: rgba(15, 106, 180, 0.14);
    color: var(--answer-accent);
    font-weight: 800;
    letter-spacing: 0.01em;
}

.chatgpt-citation-meta {
    color: #4f5f77;
}

.chatgpt-citation-chip-table .chatgpt-citation-index {
    background: rgba(29, 124, 87, 0.14);
    color: var(--answer-green);
}

.chatgpt-citation-chip-image .chatgpt-citation-index,
.chatgpt-citation-chip-formula .chatgpt-citation-index {
    background: rgba(199, 138, 29, 0.16);
    color: var(--answer-gold);
}

.stChatMessage .stMarkdown a[href^="#citation-"] {
    display: inline-block;
    color: var(--answer-accent) !important;
    font-weight: 700;
    text-decoration: none;
    background: rgba(15, 106, 180, 0.12);
    border-radius: 6px;
    padding: 0 0.24rem;
    line-height: 1.3;
}

.stChatMessage .stMarkdown a[href^="#citation-"]:hover {
    background: rgba(15, 106, 180, 0.2);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _format_created_at(value: Optional[float]) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "-"


def _init_session_state() -> None:
    defaults = {
        "selected_doc_id": None,
        "selected_doc_name": None,
        "selected_doc_ids": [],
        "doc_selection_mode": "Selected documents",
        "chat_history": [],
        "last_trace": None,
        "tree_by_doc": {},
        "session_memory": {},
        "langfuse_session_id": f"pageindex:{uuid.uuid4().hex}",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


@st.cache_resource(show_spinner=False)
def _get_api_client(config: AppConfig) -> PageIndexAPIClient:
    return PageIndexAPIClient(
        base_url=config.pageindex_backend_url,
        timeout_sec=config.request_timeout_sec,
        max_retries=config.max_retries,
        backoff_sec=config.retry_backoff_sec,
        metadata_cache_size=config.metadata_cache_size,
        metadata_cache_ttl_sec=config.metadata_cache_ttl_sec,
    )


@st.cache_resource(show_spinner=False)
def _get_llm_client(config: AppConfig) -> AzureLLMClient:
    return AzureLLMClient(
        api_key=config.azure_openai_api_key or "",
        endpoint=config.azure_openai_endpoint or "",
        api_version=config.azure_openai_api_version,
        deployment_name=config.azure_openai_deployment_name,
        timeout_sec=config.request_timeout_sec,
        max_retries=config.max_retries,
        backoff_sec=config.retry_backoff_sec,
    )


def _fetch_documents(api_client: PageIndexAPIClient, config: AppConfig) -> List[Dict]:
    documents: List[Dict] = []
    limit = max(1, min(1000, config.documents_page_size))
    offset = 0
    total = None
    while total is None or offset < total:
        payload = api_client.list_documents(limit=limit, offset=offset)
        batch = payload.get("documents", []) or []
        total = int(payload.get("total", len(batch)))
        documents.extend(batch)
        offset += len(batch)
        if not batch or len(documents) >= config.max_workspace_documents:
            break
    documents = documents[: config.max_workspace_documents]
    return sorted(documents, key=lambda item: item.get("createdAt", 0), reverse=True)


@st.cache_resource(show_spinner=False)
def _get_observer(config: AppConfig):
    return build_observer(
        enabled=config.langfuse_enabled,
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
    )


@st.cache_resource(show_spinner=False)
def _get_embedding_client(config: AppConfig):
    if not config.azure_openai_embedding_api_key or not config.azure_openai_embedding_endpoint or not config.azure_openai_embedding_deployment_name:
        return None
    try:
        return AzureEmbeddingClient(
            api_key=config.azure_openai_embedding_api_key,
            endpoint=config.azure_openai_embedding_endpoint,
            api_version=config.azure_openai_embedding_api_version,
            deployment_name=config.azure_openai_embedding_deployment_name,
            dimensions=config.pgvector_embedding_dimensions,
        )
    except Exception:
        return None


def _get_selected_document(docs: List[Dict], selected_doc_id: Optional[str]) -> Optional[Dict]:
    for doc in docs:
        if doc.get("id") == selected_doc_id:
            return doc
    return docs[0] if docs else None


def _doc_label(doc: Dict) -> str:
    doc_id = str(doc.get("id") or "")
    short_id = doc_id[:8] if doc_id else "no-id"
    return (
        f"{doc.get('name', '(unnamed)')} | "
        f"{doc.get('status', 'unknown')} | "
        f"{doc.get('pageNum', '?')} pages | "
        f"{_format_created_at(doc.get('createdAt'))} | "
        f"id:{short_id}"
    )


def _is_doc_ready(doc: Dict[str, Any]) -> bool:
    status = str(doc.get("status") or "").strip().lower()
    if status == "completed":
        return True
    return bool(doc.get("retrieval_ready"))


def _ensure_tree_loaded(
    api_client: PageIndexAPIClient,
    doc_id: str,
    tree_cache: Dict[str, List[Dict]],
) -> List[Dict]:
    if doc_id in tree_cache:
        return tree_cache[doc_id]
    tree_payload = api_client.get_tree(doc_id, summary=True)
    tree_result = tree_payload.get("result", []) or []
    if _is_doc_ready(tree_payload):
        tree_cache[doc_id] = tree_result
    return tree_result


def _render_sidebar(
    api_client: PageIndexAPIClient,
    config: AppConfig,
    logger,
) -> Optional[Tuple[List[Dict], List[Dict]]]:
    with st.sidebar:
        st.header("Document Workspace")
        st.caption(
            "Upload files or ingest web URLs via trafilatura discovery/extraction, then query one, many, or all documents."
        )

        uploaded_doc = st.file_uploader("Upload Document", type=["pdf", "json", "xlsx", "csv"], accept_multiple_files=False)
        if uploaded_doc is not None and st.button("Submit Document", width="stretch"):
            if uploaded_doc.size > config.max_upload_size_mb * 1024 * 1024:
                st.error(f"File too large. Max size is {config.max_upload_size_mb} MB.")
            else:
                try:
                    with st.spinner("Uploading and queueing document..."):
                        response = api_client.submit_document(
                            file_bytes=uploaded_doc.getvalue(),
                            filename=uploaded_doc.name,
                            if_retrieval=True,
                        )
                        doc_id = response.get("doc_id")
                        if not doc_id:
                            raise APIClientError("Backend did not return doc_id")
                        api_client.invalidate_document_metadata(doc_id)
                        metadata = api_client.get_document_metadata(doc_id, refresh=True)
                        st.session_state.selected_doc_id = doc_id
                        st.session_state.selected_doc_name = metadata.get("name", uploaded_doc.name)
                        st.session_state.selected_doc_ids = [doc_id]
                        st.session_state.doc_selection_mode = "Selected documents"
                        st.session_state.tree_by_doc.pop(doc_id, None)
                    status = metadata.get("status", response.get("status", "queued"))
                    if status == "completed":
                        st.success(f"Document submitted: {st.session_state.selected_doc_name}")
                    else:
                        st.success(
                            f"Document submitted: {st.session_state.selected_doc_name} (status: {status})."
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("upload_failed")
                    st.error(f"Upload failed: {exc}")

        single_url = st.text_input(
            "Extract Single URL",
            key="single_source_url",
            placeholder="https://example.com/article",
        )
        if st.button("Submit URL", key="submit_single_url", width="stretch"):
            candidate_url = (single_url or "").strip()
            if not candidate_url:
                st.error("Please enter a URL.")
            else:
                try:
                    with st.spinner("Extracting URL and queueing document..."):
                        response = api_client.submit_url(candidate_url, if_retrieval=True)
                        doc_id = str(response.get("doc_id") or "").strip()
                        if not doc_id:
                            raise APIClientError("Backend did not return doc_id")
                        api_client.invalidate_document_metadata(doc_id)
                        metadata = api_client.get_document_metadata(doc_id, refresh=True)
                        st.session_state.selected_doc_id = doc_id
                        st.session_state.selected_doc_name = metadata.get("name", candidate_url)
                        st.session_state.selected_doc_ids = [doc_id]
                        st.session_state.doc_selection_mode = "Selected documents"
                        st.session_state.tree_by_doc.pop(doc_id, None)
                    if response.get("duplicate"):
                        st.info(
                            f"URL already indexed: {st.session_state.selected_doc_name} (status: {metadata.get('status', 'unknown')})."
                        )
                    else:
                        st.success(
                            f"URL submitted: {st.session_state.selected_doc_name} (status: {metadata.get('status', 'queued')})."
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("url_submit_failed")
                    st.error(f"URL submission failed: {exc}")

        discover_url = st.text_input(
            "Discovery Seed URL",
            key="discover_seed_url",
            placeholder="https://example.com",
        )
        max_discovery_urls = st.number_input(
            "Discovery Max URLs",
            key="discover_max_urls",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
        )
        if st.button("Run URL Discovery", key="submit_discovery_url", width="stretch"):
            seed_url = (discover_url or "").strip()
            if not seed_url:
                st.error("Please enter a discovery seed URL.")
            else:
                try:
                    with st.spinner("Running discovery and queueing document..."):
                        response = api_client.submit_url_discovery(
                            seed_url,
                            max_urls=int(max_discovery_urls),
                            if_retrieval=True,
                        )
                        doc_id = str(response.get("doc_id") or "").strip()
                        if not doc_id:
                            raise APIClientError("Backend did not return doc_id")
                        api_client.invalidate_document_metadata(doc_id)
                        metadata = api_client.get_document_metadata(doc_id, refresh=True)
                        st.session_state.selected_doc_id = doc_id
                        st.session_state.selected_doc_name = metadata.get("name", seed_url)
                        st.session_state.selected_doc_ids = [doc_id]
                        st.session_state.doc_selection_mode = "Selected documents"
                        st.session_state.tree_by_doc.pop(doc_id, None)
                    st.success(
                        f"Discovery submitted: {st.session_state.selected_doc_name} (status: {metadata.get('status', 'queued')})."
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("url_discovery_submit_failed")
                    st.error(f"URL discovery submission failed: {exc}")

        bulk_urls_text = st.text_area(
            "Bulk URL Extraction (one URL per line)",
            key="bulk_source_urls",
            height=100,
            placeholder="https://example.com/a\nhttps://example.com/b",
        )
        if st.button("Submit URL Batch", key="submit_bulk_urls", width="stretch"):
            urls = [line.strip() for line in (bulk_urls_text or "").splitlines() if line.strip()]
            if not urls:
                st.error("Please provide at least one URL.")
            else:
                try:
                    with st.spinner("Submitting URL batch..."):
                        response = api_client.submit_urls_bulk(urls, if_retrieval=True)
                        items = response.get("items", []) or []
                        accepted_doc_ids = [
                            str(item.get("doc_id"))
                            for item in items
                            if item.get("doc_id") and item.get("status") != "failed"
                        ]
                        if accepted_doc_ids:
                            st.session_state.selected_doc_ids = accepted_doc_ids
                            st.session_state.selected_doc_id = accepted_doc_ids[0]
                            st.session_state.doc_selection_mode = "Selected documents"
                            for doc_id in accepted_doc_ids:
                                st.session_state.tree_by_doc.pop(doc_id, None)

                    failed_count = int(response.get("failed", 0) or 0)
                    duplicate_count = int(response.get("duplicate", 0) or 0)
                    accepted_count = int(response.get("accepted", 0) or 0)
                    summary = (
                        f"Batch processed: accepted={accepted_count}, duplicates={duplicate_count}, failed={failed_count}."
                    )
                    if failed_count:
                        st.warning(summary)
                    else:
                        st.success(summary)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("bulk_url_submit_failed")
                    st.error(f"Bulk URL submission failed: {exc}")

        if st.button("Clear Chat", width="stretch"):
            st.session_state.chat_history = []
            st.session_state.last_trace = None
            st.session_state.session_memory = {}
            st.rerun()

        try:
            docs = _fetch_documents(api_client, config)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load documents: {exc}")
            return None

        try:
            stats = api_client.get_processing_stats()
        except Exception:
            stats = None

        if stats:
            metrics = stats.get("metrics", {}) or {}
            status_counts = stats.get("status_counts", {}) or {}
            with st.expander("Backend Processing Stats", expanded=False):
                st.caption(
                    f"Workers: {stats.get('worker_count', '?')} | Queue: {stats.get('queue_size', '?')}"
                )
                st.caption(
                    " | ".join(
                        [
                            f"queued: {status_counts.get('queued', 0)}",
                            f"processing: {status_counts.get('processing', 0)}",
                            f"completed: {status_counts.get('completed', 0)}",
                            f"failed: {status_counts.get('failed', 0)}",
                        ]
                    )
                )
                st.caption(
                    " | ".join(
                        [
                            f"avg queue wait: {metrics.get('avg_queue_wait_sec', 0.0)}s",
                            f"avg processing: {metrics.get('avg_processing_sec', 0.0)}s",
                            f"avg llm tree: {metrics.get('avg_llm_tree_sec', 0.0)}s",
                        ]
                    )
                )

        if not docs:
            st.info("No documents found. Upload a file or submit URLs to start.")
            return docs, []

        docs_by_id = {str(doc.get("id")): doc for doc in docs if doc.get("id")}
        all_doc_ids = list(docs_by_id.keys())
        labels_by_id = {doc_id: _doc_label(doc) for doc_id, doc in docs_by_id.items()}

        st.radio(
            "Document scope",
            options=["Selected documents", "All documents"],
            key="doc_selection_mode",
            help="Use all docs or select a specific subset for each answer.",
        )

        default_selected_ids = [
            doc_id for doc_id in (st.session_state.selected_doc_ids or []) if doc_id in docs_by_id
        ]
        if not default_selected_ids:
            selected_doc = _get_selected_document(docs, st.session_state.selected_doc_id)
            if selected_doc and selected_doc.get("id"):
                default_selected_ids = [str(selected_doc.get("id"))]
            elif all_doc_ids:
                default_selected_ids = [all_doc_ids[0]]

        if st.session_state.doc_selection_mode == "All documents":
            selected_doc_ids = all_doc_ids
            st.caption(f"Selected {len(selected_doc_ids)} of {len(all_doc_ids)} documents.")
        else:
            selected_doc_ids = st.multiselect(
                "Select document(s)",
                options=all_doc_ids,
                default=default_selected_ids,
                format_func=lambda doc_id: labels_by_id.get(doc_id, doc_id),
                help="Choose one or more documents for grounded retrieval.",
            )

        st.session_state.selected_doc_ids = selected_doc_ids
        if selected_doc_ids:
            primary_doc = docs_by_id[selected_doc_ids[0]]
            st.session_state.selected_doc_id = selected_doc_ids[0]
            st.session_state.selected_doc_name = str(primary_doc.get("name", "unknown.pdf"))
        else:
            st.session_state.selected_doc_id = None
            st.session_state.selected_doc_name = None

        selected_docs = [docs_by_id[doc_id] for doc_id in selected_doc_ids if doc_id in docs_by_id]
        return docs, selected_docs


def _render_trees(api_client: PageIndexAPIClient, selected_docs: List[Dict], logger, config: AppConfig) -> Dict[str, List[Dict]]:
    trees: Dict[str, List[Dict]] = {}
    with st.expander("Simplified Tree Structure of Selected Documents", expanded=False):
        if not selected_docs:
            st.info("Select at least one document to view tree context.")
            return trees

        if len(selected_docs) > config.tree_preview_doc_limit:
            st.caption(
                f"Tree preview limited to {config.tree_preview_doc_limit} documents. Full tree loading happens only for routed query candidates."
            )
            selected_docs = selected_docs[: config.tree_preview_doc_limit]

        if len(selected_docs) == 1:
            doc = selected_docs[0]
            doc_id = str(doc.get("id"))
            doc_name = str(doc.get("name", "unknown.pdf"))
            if not _is_doc_ready(doc):
                st.caption(f"Document: {doc_name} (id:{doc_id[:8]})")
                st.info(f"Tree is not ready yet. Current status: {doc.get('status', 'unknown')}")
                return trees
            try:
                tree = _ensure_tree_loaded(api_client, doc_id, st.session_state.tree_by_doc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tree_load_failed")
                st.error(f"Failed to load tree: {exc}")
                return trees
            trees[doc_id] = tree
            st.caption(f"Document: {doc_name} (id:{doc_id[:8]})")
            render_tree_section(tree, max_summary_chars=240)
            if st.toggle("Show raw tree JSON", value=False, key=f"show_raw_tree_{doc_id}"):
                st.json(tree)
            return trees

        tabs = st.tabs(
            [f"{str(doc.get('name', 'unknown.pdf'))[:28]} ({str(doc.get('id', ''))[:8]})" for doc in selected_docs]
        )
        for idx, doc in enumerate(selected_docs):
            doc_id = str(doc.get("id"))
            with tabs[idx]:
                if not _is_doc_ready(doc):
                    st.info(f"Tree is not ready yet. Current status: {doc.get('status', 'unknown')}")
                    continue
                try:
                    tree = _ensure_tree_loaded(api_client, doc_id, st.session_state.tree_by_doc)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("tree_load_failed")
                    st.error(f"Failed to load tree for {doc.get('name', 'unknown.pdf')}: {exc}")
                    continue
                trees[doc_id] = tree
                render_tree_section(tree, max_summary_chars=240)
                if st.toggle("Show raw tree JSON", value=False, key=f"show_raw_tree_{doc_id}"):
                    st.json(tree)
    return trees


def main() -> None:
    st.set_page_config(
        page_title="PageIndex Streamlit RAG",
        page_icon="PI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_css()
    _init_session_state()

    config = load_config()
    logger = get_logger()
    api_client = _get_api_client(config)
    observer = _get_observer(config)
    embedding_client = _get_embedding_client(config)
    retrieval_service = RetrievalService(
        api_client=api_client,
        config=config,
        logger=logger,
        observer=observer,
        embedding_client=embedding_client,
    )

    st.title("PageIndex RAG Assistant")
    st.caption(
        "Minimal grounded chat with live thinking and cited answers."
    )

    missing = config.missing_azure_settings()
    llm_client = None
    if missing:
        st.error("Missing Azure OpenAI env vars: " + ", ".join(missing))
        st.info(
            "Set these in `.env` at project root and restart Streamlit. "
            "Upload/browse still works, but answer generation requires Azure credentials."
        )
    else:
        llm_client = _get_llm_client(config)

    sidebar_result = _render_sidebar(api_client, config, logger)
    if sidebar_result is None:
        return
    docs, selected_docs = sidebar_result
    if not docs:
        return
    if not selected_docs:
        st.warning("Select at least one document from the sidebar.")
        return

    selected_doc_id = str(selected_docs[0].get("id"))
    selected_doc_name = str(selected_docs[0].get("name", "unknown.pdf"))
    st.session_state.selected_doc_id = selected_doc_id
    st.session_state.selected_doc_name = selected_doc_name
    st.session_state.selected_doc_ids = [str(doc.get("id")) for doc in selected_docs if doc.get("id")]

    trees_by_doc = _render_trees(api_client, selected_docs, logger, config)
    ready_docs = [doc for doc in selected_docs if _is_doc_ready(doc)]
    pending_docs = [doc for doc in selected_docs if not _is_doc_ready(doc)]
    total_selected = max(1, len(selected_docs))
    progress_value = len(ready_docs) / total_selected
    st.progress(
        progress_value,
        text=f"Document processing progress: {len(ready_docs)}/{len(selected_docs)} completed",
    )
    if pending_docs:
        pending_names = ", ".join(str(doc.get("name", "unknown.pdf")) for doc in pending_docs)
        st.info(f"Still processing: {pending_names}")
    if not ready_docs:
        st.warning("Selected documents are still processing. Querying is available after tree generation completes.")
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "trace" in message:
                render_trace(message["trace"], max_preview_chars=config.max_page_preview_chars)
            else:
                st.markdown(message.get("content", ""))

    scope_label = "selected documents" if len(ready_docs) > 1 else "selected document"
    query = st.chat_input(f"Ask a question about the {scope_label}")
    if not query:
        return

    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        pipeline = RAGPipeline(
            api_client=api_client,
            llm_client=llm_client,
            config=config,
            logger=logger,
            observer=observer,
        )
        graph_orchestrator = LangGraphQueryOrchestrator(retrieval_service=retrieval_service, pipeline=pipeline)

        latest_page_result: Optional[PageContentResult] = None
        thinking_steps: List[str] = []
        live_status = "Thinking through relevant evidence..."
        thinking_started_at = time.perf_counter()

        def _trimmed_thought(value: str, max_chars: int = 220) -> str:
            compact = " ".join(str(value or "").split())
            if len(compact) <= max_chars:
                return compact
            return compact[:max_chars].rstrip() + "..."

        def _append_thinking_step(value: str) -> None:
            step = _trimmed_thought(value)
            if not step:
                return
            if not thinking_steps or thinking_steps[-1] != step:
                thinking_steps.append(step)

        def _render_live_answer(partial_text: str = "") -> None:
            with answer_placeholder.container():
                render_streaming_answer(
                    partial_text=partial_text,
                    thinking_steps=thinking_steps,
                    status=live_status,
                    thought_seconds=int(max(1, time.perf_counter() - thinking_started_at)),
                    collapse_thinking=False,
                )

        def on_stage(stage: str, payload: Dict) -> None:
            nonlocal latest_page_result, live_status
            if stage == "tree_search":
                tree_search = payload.get("tree_search")
                doc_ref = payload.get("doc_ref", "document")
                if tree_search and getattr(tree_search, "thinking", None):
                    _append_thinking_step(f"Scanned {doc_ref}: {tree_search.thinking}")
                    _render_live_answer()
            elif stage == "retrieved_nodes":
                retrieved_nodes = payload.get("retrieved_nodes", []) or []
                if retrieved_nodes:
                    _append_thinking_step(f"Selected {len(retrieved_nodes)} relevant evidence nodes.")
                    _render_live_answer()
            elif stage == "page_content":
                latest_page_result = payload.get("page_result")
                snippet_count = len((latest_page_result.content if latest_page_result else []) or [])
                if snippet_count:
                    _append_thinking_step(f"Grounded on {snippet_count} evidence snippets.")
                    _render_live_answer()
            elif stage == "answer_start":
                live_status = "Drafting final answer..."
                _append_thinking_step("Composing concise response with citations.")
                _render_live_answer()
            elif stage == "answer_stream":
                partial_text = str(payload.get("text", "")).strip()
                _render_live_answer(partial_text)
            elif stage == "error":
                with answer_placeholder.container():
                    st.error(payload.get("error", "Unknown pipeline error"))

        if llm_client is None:
            trace = QueryTrace(
                trace_id="missing-llm-config",
                query=query,
                doc_id=selected_doc_id,
                doc_name=selected_doc_name,
                selected_doc_ids=[str(item.get("id")) for item in ready_docs if item.get("id")],
                selected_doc_names=[str(item.get("name", "unknown.pdf")) for item in ready_docs],
                errors=["Azure OpenAI credentials are missing. Set env vars and retry."],
            )
        else:
            session_memory = build_session_memory(
                st.session_state.chat_history,
                selected_docs=selected_docs,
                max_turns=config.session_memory_turn_window,
                max_chars=config.session_memory_char_budget,
            )
            st.session_state.session_memory = session_memory
            observer_trace = observer.start_trace(
                "pageindex_query_routing",
                session_id=st.session_state.langfuse_session_id,
                input_payload={
                    "query": query,
                    "candidate_doc_count": len(ready_docs),
                },
                metadata={
                    "selection_mode": st.session_state.doc_selection_mode,
                },
            )
            if config.enable_langgraph:
                graph_result = graph_orchestrator.invoke(
                    query=query,
                    ready_documents=ready_docs,
                    tree_cache=st.session_state.tree_by_doc,
                    session_memory=st.session_state.session_memory,
                    stage_callback=on_stage,
                    trace_handle=observer_trace,
                    langfuse_session_id=st.session_state.langfuse_session_id,
                )
                trace = graph_result.get("trace")
                route_info = graph_result.get("route_info", {}) or {}
                if trace is None:
                    trace = QueryTrace(
                        trace_id="routing-empty",
                        query=query,
                        doc_id=selected_doc_id,
                        doc_name=selected_doc_name,
                        selected_doc_ids=[str(item.get("id")) for item in ready_docs if item.get("id")],
                        selected_doc_names=[str(item.get("name", "unknown.pdf")) for item in ready_docs],
                        routing=route_info,
                        errors=["No routed documents were available for query execution."],
                    )
            else:
                query_documents, route_info = retrieval_service.prepare_query_documents(
                    query=query,
                    documents=ready_docs,
                    tree_cache=st.session_state.tree_by_doc,
                    session_memory=st.session_state.session_memory,
                    trace_handle=observer_trace,
                )
                if not query_documents:
                    trace = QueryTrace(
                        trace_id="routing-empty",
                        query=query,
                        doc_id=selected_doc_id,
                        doc_name=selected_doc_name,
                        selected_doc_ids=[str(item.get("id")) for item in ready_docs if item.get("id")],
                        selected_doc_names=[str(item.get("name", "unknown.pdf")) for item in ready_docs],
                        routing=route_info,
                        errors=["No routed documents were available for query execution."],
                    )
                else:
                    trace = pipeline.run_query_multi(
                        query=query,
                        documents=query_documents,
                        session_memory=st.session_state.session_memory,
                        langfuse_session_id=st.session_state.langfuse_session_id,
                        stage_callback=on_stage,
                        parent_trace_handle=observer_trace,
                        route_info=route_info,
                    )
                    trace.routing = route_info

            if trace.errors and not route_info:
                trace = QueryTrace(
                    trace_id="routing-empty",
                    query=query,
                    doc_id=selected_doc_id,
                    doc_name=selected_doc_name,
                    selected_doc_ids=[str(item.get("id")) for item in ready_docs if item.get("id")],
                    selected_doc_names=[str(item.get("name", "unknown.pdf")) for item in ready_docs],
                    routing={},
                    errors=["No routed documents were available for query execution."],
                )
            else:
                route_info = trace.routing or route_info
                observer.end_trace(
                    observer_trace,
                    output_payload={
                        "routed_doc_ids": route_info.get("route_doc_ids", []),
                        "routed_doc_names": route_info.get("route_doc_names", []),
                    },
                    metadata=route_info,
                )

        answer_placeholder.empty()
        with answer_placeholder.container():
            if thinking_steps:
                thought_seconds = int(max(1, time.perf_counter() - thinking_started_at))
                render_thought_summary(
                    thinking_steps=thinking_steps,
                    thought_seconds=thought_seconds,
                )
            render_answer_section(
                answer_result=trace.answer_result,
                page_result=trace.page_result or latest_page_result,
                max_preview_chars=config.max_page_preview_chars,
                trace=trace,
            )
            if trace.errors:
                st.error(" | ".join(trace.errors))

    st.session_state.last_trace = trace
    st.session_state.chat_history.append({"role": "assistant", "trace": trace})
    st.session_state.session_memory = build_session_memory(
        st.session_state.chat_history,
        selected_docs=selected_docs,
        max_turns=config.session_memory_turn_window,
        max_chars=config.session_memory_char_budget,
    )
    log_query_trace(logger, trace)


if __name__ == "__main__":
    main()
