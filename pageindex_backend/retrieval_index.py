from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from pgvector.psycopg import register_vector

from embedding_utils import AzureEmbeddingClient


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _flatten_with_path(tree_nodes: Sequence[Dict[str, Any]], parents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    lineage = list(parents or [])
    flattened: List[Dict[str, Any]] = []
    for node in tree_nodes or []:
        title = _clean_text(node.get("title") or "Untitled")
        current_path = lineage + [title]
        flattened.append(
            {
                "node_id": str(node.get("node_id") or ""),
                "title": title,
                "path": " > ".join(current_path),
                "summary": _clean_text(node.get("summary") or node.get("prefix_summary") or node.get("text") or title),
                "page_index": int(node.get("page_index") or node.get("start_index") or 0),
            }
        )
        children = node.get("nodes") or []
        flattened.extend(_flatten_with_path(children, current_path))
    return flattened


def build_document_summary(doc_name: str, tree_nodes: Sequence[Dict[str, Any]], ocr_pages: Sequence[Dict[str, Any]]) -> Tuple[str, List[str]]:
    flattened = _flatten_with_path(tree_nodes)
    headings = [item["title"] for item in flattened[:12] if item.get("title")]
    section_summaries = [
        f"{item['title']}: {_truncate(item.get('summary', ''), 220)}"
        for item in flattened[:8]
        if item.get("title")
    ]
    if not section_summaries:
        section_summaries = [
            _truncate(item.get("text", ""), 220)
            for item in (ocr_pages or [])[:4]
            if _clean_text(item.get("text", ""))
        ]
    body = "\n".join(section_summaries[:8])
    summary = _truncate(f"Document: {doc_name}\n{body}", 2400)
    return summary, headings[:20]


def build_section_records(doc_id: str, tree_nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened = _flatten_with_path(tree_nodes)
    records: List[Dict[str, Any]] = []
    for item in flattened:
        node_id = str(item.get("node_id") or "")
        if not node_id:
            continue
        summary = _truncate(item.get("summary"), 1600)
        records.append(
            {
                "section_uid": f"{doc_id}:{node_id}",
                "doc_id": doc_id,
                "node_id": node_id,
                "title": item.get("title"),
                "path": item.get("path"),
                "start_page": int(item.get("page_index") or 0),
                "end_page": int(item.get("page_index") or 0),
                "summary": summary,
                "metadata": {
                    "path": item.get("path"),
                    "title": item.get("title"),
                },
            }
        )
    return records


class PgVectorIndexer:
    def __init__(
        self,
        *,
        dsn: str,
        embedding_client: AzureEmbeddingClient,
        document_table: str,
        section_table: str,
        embedding_dimensions: int,
    ) -> None:
        self.dsn = dsn
        self.embedding_client = embedding_client
        self.document_table = document_table
        self.section_table = section_table
        self.embedding_dimensions = embedding_dimensions
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.document_table} (
                        doc_id TEXT PRIMARY KEY,
                        doc_name TEXT NOT NULL,
                        file_hash TEXT,
                        created_at DOUBLE PRECISION,
                        updated_at DOUBLE PRECISION,
                        doc_summary TEXT,
                        headings JSONB DEFAULT '[]'::jsonb,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({self.embedding_dimensions})
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.section_table} (
                        section_uid TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        node_id TEXT,
                        title TEXT,
                        path TEXT,
                        start_page INTEGER,
                        end_page INTEGER,
                        summary TEXT,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({self.embedding_dimensions})
                    )
                    """
                )
            conn.commit()
        self._schema_ready = True

    def upsert_document(
        self,
        *,
        doc_id: str,
        doc_name: str,
        file_hash: str,
        created_at: float,
        updated_at: float,
        doc_summary: str,
        headings: Sequence[str],
        metadata: Dict[str, Any],
        section_records: Sequence[Dict[str, Any]],
    ) -> None:
        self.ensure_schema()
        doc_embedding = self.embedding_client.embed_query(doc_summary)
        section_texts = [
            _truncate(f"{item.get('path', '')}\n{item.get('summary', '')}", 1800)
            for item in section_records
        ]
        section_vectors = self.embedding_client.embed_texts(section_texts) if section_texts else []

        with psycopg.connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.document_table} (
                        doc_id, doc_name, file_hash, created_at, updated_at, doc_summary, headings, metadata, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        doc_name = EXCLUDED.doc_name,
                        file_hash = EXCLUDED.file_hash,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        doc_summary = EXCLUDED.doc_summary,
                        headings = EXCLUDED.headings,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        doc_id,
                        doc_name,
                        file_hash,
                        created_at,
                        updated_at,
                        doc_summary,
                        json.dumps(list(headings)),
                        json.dumps(metadata),
                        doc_embedding,
                    ),
                )
                cur.execute(f"DELETE FROM {self.section_table} WHERE doc_id = %s", (doc_id,))
                for section, vector in zip(section_records, section_vectors):
                    cur.execute(
                        f"""
                        INSERT INTO {self.section_table} (
                            section_uid, doc_id, node_id, title, path, start_page, end_page, summary, metadata, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            section.get("section_uid"),
                            section.get("doc_id"),
                            section.get("node_id"),
                            section.get("title"),
                            section.get("path"),
                            section.get("start_page"),
                            section.get("end_page"),
                            section.get("summary"),
                            json.dumps(section.get("metadata", {})),
                            vector,
                        ),
                    )
            conn.commit()


def build_pgvector_indexer(
    *,
    dsn: Optional[str],
    embedding_client: Optional[AzureEmbeddingClient],
    document_table: str,
    section_table: str,
    embedding_dimensions: int,
) -> Optional[PgVectorIndexer]:
    if not dsn or embedding_client is None:
        return None
    try:
        return PgVectorIndexer(
            dsn=dsn,
            embedding_client=embedding_client,
            document_table=document_table,
            section_table=section_table,
            embedding_dimensions=embedding_dimensions,
        )
    except Exception:
        return None
