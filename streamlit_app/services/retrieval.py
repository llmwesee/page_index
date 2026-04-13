from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from pgvector.psycopg import register_vector

from streamlit_app.config import AppConfig


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _tokenize(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", (value or "").lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12f}" for value in values) + "]"


def _is_doc_ready(doc: Dict[str, Any]) -> bool:
    status = str(doc.get("status") or "").strip().lower()
    if status == "completed":
        return True
    return bool(doc.get("retrieval_ready"))


class RetrievalService:
    def __init__(
        self,
        *,
        api_client: Any,
        config: AppConfig,
        logger: Optional[Any] = None,
        observer: Optional[Any] = None,
        embedding_client: Optional[Any] = None,
    ) -> None:
        self.api_client = api_client
        self.config = config
        self.logger = logger
        self.observer = observer
        self.embedding_client = embedding_client

    def prepare_query_documents(
        self,
        *,
        query: str,
        documents: Sequence[Dict[str, Any]],
        tree_cache: Dict[str, List[Dict[str, Any]]],
        session_memory: Optional[Dict[str, Any]] = None,
        trace_handle: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        started = time.perf_counter()
        ready_docs = [doc for doc in documents if _is_doc_ready(doc) and doc.get("id")]
        route_info: Dict[str, Any] = {
            "enabled": bool(self.config.enable_document_routing),
            "strategy": "none",
            "candidate_count": len(ready_docs),
            "routed_count": 0,
            "route_doc_ids": [],
            "route_doc_names": [],
            "section_hint_count": 0,
            "section_hints_by_doc": {},
            "latency_sec": 0.0,
        }

        if not ready_docs:
            route_info["latency_sec"] = round(time.perf_counter() - started, 4)
            return [], route_info

        if not self.config.enable_document_routing or len(ready_docs) < self.config.document_routing_min_docs:
            routed_docs = ready_docs
            route_info["strategy"] = "disabled_or_small_candidate_set"
        else:
            routed_docs, strategy = self._route_documents(
                query,
                ready_docs,
                session_memory=session_memory,
                trace_handle=trace_handle,
            )
            route_info["strategy"] = strategy

        prepared_docs: List[Dict[str, Any]] = []
        section_hints_by_doc: Dict[str, List[str]] = {}
        if self.config.enable_section_reranking:
            section_hints_by_doc = self._fetch_section_hints(
                query,
                routed_docs,
                trace_handle=trace_handle,
            )
        for doc in routed_docs[: max(1, self.config.document_routing_top_k)]:
            doc_id = str(doc.get("id"))
            tree = tree_cache.get(doc_id)
            if tree is None:
                try:
                    tree_payload = self.api_client.get_tree(doc_id, summary=True)
                    tree = tree_payload.get("result", []) or []
                    if _is_doc_ready(tree_payload):
                        tree_cache[doc_id] = tree
                except Exception as exc:  # noqa: BLE001
                    if self.logger:
                        self.logger.warning("route_tree_load_failed doc_id=%s error=%s", doc_id, exc)
                    continue
            prepared_docs.append(
                {
                    "doc_id": doc_id,
                    "doc_name": str(doc.get("name", "unknown.pdf")),
                    "tree": tree or [],
                    "preferred_node_ids": list(section_hints_by_doc.get(doc_id, [])),
                }
            )

        route_info["route_doc_ids"] = [item["doc_id"] for item in prepared_docs]
        route_info["route_doc_names"] = [item["doc_name"] for item in prepared_docs]
        route_info["section_hints_by_doc"] = {
            doc_id: node_ids
            for doc_id, node_ids in section_hints_by_doc.items()
            if node_ids
        }
        route_info["section_hint_count"] = sum(len(node_ids) for node_ids in route_info["section_hints_by_doc"].values())
        route_info["routed_count"] = len(prepared_docs)
        route_info["latency_sec"] = round(time.perf_counter() - started, 4)

        if self.observer:
            self.observer.event(trace_handle, "document_routing", route_info)
        return prepared_docs, route_info

    def _route_documents(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        *,
        session_memory: Optional[Dict[str, Any]] = None,
        trace_handle: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        pgvector_docs = self._route_with_pgvector(query, documents, trace_handle=trace_handle)
        if pgvector_docs is not None:
            return pgvector_docs, "pgvector"
        return (
            self._route_with_local_scoring(
                query,
                documents,
                session_memory=session_memory,
                trace_handle=trace_handle,
            ),
            "local_heuristic",
        )

    def _fetch_section_hints(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        *,
        trace_handle: Optional[Any] = None,
    ) -> Dict[str, List[str]]:
        pgvector_hints = self._fetch_section_hints_with_pgvector(
            query,
            documents,
            trace_handle=trace_handle,
        )
        if pgvector_hints:
            return pgvector_hints
        return {}

    def _route_with_pgvector(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        *,
        trace_handle: Optional[Any] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.config.pgvector_enabled or not self.config.pgvector_dsn or self.embedding_client is None:
            return None

        doc_ids = [str(doc.get("id")) for doc in documents if doc.get("id")]
        if not doc_ids:
            return None

        try:
            query_vector = self.embedding_client.embed_query(query)
        except Exception:
            return None

        vector_literal = _vector_literal(query_vector)

        try:
            with psycopg.connect(self.config.pgvector_dsn) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT doc_id, doc_name
                        FROM {self.config.pgvector_document_table}
                        WHERE doc_id = ANY(%s)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            doc_ids,
                            vector_literal,
                            max(1, self.config.document_routing_top_k),
                        ),
                    )
                    rows = cur.fetchall()
        except Exception:
            return None

        if not rows:
            return None

        if self.observer:
            self.observer.event(
                trace_handle,
                "pgvector_document_routing_candidates",
                {
                    "candidate_doc_count": len(doc_ids),
                    "top_k": max(1, self.config.document_routing_top_k),
                    "query_chars": len(query or ""),
                    "rows": [
                        {
                            "rank": idx + 1,
                            "doc_id": str(row[0] or ""),
                            "doc_name": str(row[1] or ""),
                        }
                        for idx, row in enumerate(rows)
                    ],
                },
            )

        by_id = {str(doc.get("id")): dict(doc) for doc in documents if doc.get("id")}
        routed_docs: List[Dict[str, Any]] = []
        for row in rows:
            doc_id = str(row[0])
            if doc_id in by_id:
                routed_docs.append(by_id[doc_id])

        if self.observer:
            self.observer.event(
                trace_handle,
                "pgvector_document_routing_selected",
                {
                    "selected_doc_ids": [str(doc.get("id")) for doc in routed_docs if doc.get("id")],
                    "selected_doc_names": [str(doc.get("name", "")) for doc in routed_docs],
                    "selected_count": len(routed_docs),
                },
            )
        return routed_docs or None

    def _fetch_section_hints_with_pgvector(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        *,
        trace_handle: Optional[Any] = None,
    ) -> Dict[str, List[str]]:
        if (
            not self.config.pgvector_enabled
            or not self.config.pgvector_dsn
            or self.embedding_client is None
            or not self.config.enable_section_reranking
        ):
            return {}

        doc_ids = [str(doc.get("id")) for doc in documents if doc.get("id")]
        if not doc_ids:
            return {}

        try:
            query_vector = self.embedding_client.embed_query(query)
        except Exception:
            return {}

        vector_literal = _vector_literal(query_vector)

        rows: List[Tuple[Any, ...]] = []
        try:
            with psycopg.connect(self.config.pgvector_dsn) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT doc_id, node_id
                        FROM {self.config.pgvector_section_table}
                        WHERE doc_id = ANY(%s)
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            doc_ids,
                            vector_literal,
                            max(1, self.config.section_rerank_top_k),
                        ),
                    )
                    rows = cur.fetchall()
        except Exception:
            return {}

        if self.observer:
            self.observer.event(
                trace_handle,
                "pgvector_section_hint_candidates",
                {
                    "candidate_doc_ids": doc_ids,
                    "top_k": max(1, self.config.section_rerank_top_k),
                    "query_chars": len(query or ""),
                    "rows": [
                        {
                            "rank": idx + 1,
                            "doc_id": str(row[0] or ""),
                            "node_id": str(row[1] or ""),
                        }
                        for idx, row in enumerate(rows)
                    ],
                },
            )

        hints: Dict[str, List[str]] = defaultdict(list)
        for row in rows:
            doc_id = str(row[0] or "")
            node_id = str(row[1] or "")
            if not doc_id or not node_id:
                continue
            existing = hints[doc_id]
            if node_id in existing:
                continue
            if len(existing) >= max(1, self.config.section_rerank_node_hints):
                continue
            existing.append(node_id)

        if self.observer:
            self.observer.event(
                trace_handle,
                "pgvector_section_hints_selected",
                {
                    "hint_count": sum(len(node_ids) for node_ids in hints.values()),
                    "hints_by_doc": dict(hints),
                    "node_hints_limit": max(1, self.config.section_rerank_node_hints),
                },
            )
        return dict(hints)

    def _route_with_local_scoring(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        *,
        session_memory: Optional[Dict[str, Any]] = None,
        trace_handle: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        query_text = (query or "").lower()
        query_terms = set(_tokenize(query))
        selected_names = {
            str(name).lower()
            for name in (session_memory or {}).get("selected_doc_names", [])
            if str(name).strip()
        }

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in documents:
            doc_name = str(doc.get("name", "")).lower()
            doc_terms = set(_tokenize(doc_name))
            overlap = len(query_terms & doc_terms)
            score = float(overlap)
            if query_text and doc_name and query_text in doc_name:
                score += 4.0
            if any(term and term in doc_name for term in query_terms):
                score += 1.5
            if doc_name in selected_names:
                score += 0.75
            score += min(float(doc.get("createdAt", 0) or 0), time.time()) / 1_000_000_000_000
            scored.append((score, dict(doc)))

        scored.sort(key=lambda item: item[0], reverse=True)
        routed_docs = [item[1] for item in scored[: max(1, self.config.document_routing_top_k)]]
        if self.observer:
            self.observer.event(
                trace_handle,
                "local_document_routing_candidates",
                {
                    "query_chars": len(query or ""),
                    "top_k": max(1, self.config.document_routing_top_k),
                    "candidates": [
                        {
                            "rank": idx + 1,
                            "doc_id": str(item[1].get("id") or ""),
                            "doc_name": str(item[1].get("name") or ""),
                            "score": round(float(item[0]), 6),
                        }
                        for idx, item in enumerate(scored[: max(1, self.config.document_routing_top_k)])
                    ],
                },
            )
        return routed_docs if routed_docs else list(documents[: max(1, self.config.document_routing_top_k)])
