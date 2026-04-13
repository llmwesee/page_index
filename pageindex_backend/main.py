from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Any, Dict, List, Optional
import uuid
import time
import json
from pathlib import Path
import fitz
import re
from collections import Counter
import copy
import asyncio
import sys
import hashlib
import os
import contextlib
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import dotenv_values, load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore
    dotenv_values = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if load_dotenv is not None:
    dotenv_path = ROOT_DIR / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)

DOTENV_VALUES = dotenv_values(ROOT_DIR / ".env") if dotenv_values is not None and (ROOT_DIR / ".env").exists() else {}


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is not None:
        return value
    dotenv_value = DOTENV_VALUES.get(name)
    if dotenv_value is not None:
        return str(dotenv_value)
    return default


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default

from embedding_utils import AzureEmbeddingClient
from pageindex.utils import ConfigLoader, set_llm_defaults_from_opt, load_json_multimodal_payload
from pageindex.excel_parser import load_excel_multimodal_payload
from pageindex.excel_toc import create_excel_toc_input_file
from pageindex.page_index_md import generate_summaries_for_structure_md
from pageindex.page_index import page_index as llm_page_index
from pageindex_backend.retrieval_index import (
    build_document_summary,
    build_pgvector_indexer,
    build_section_records,
)
from pageindex_backend.url_ingestion import (
    URLIngestionError,
    ingest_url_to_local_artifact,
    normalize_source_url,
)

HEADING_PATTERNS = [
    re.compile(r"^PART\s+[IVXLC]+", re.I),
    re.compile(r"^CHAPTER\s+[IVXLC]+", re.I),
    re.compile(r"^SECTION\s+\d+", re.I),
]


def is_heading(text: str) -> bool:
    if any(p.match(text) for p in HEADING_PATTERNS):
        return True
    if text.isupper() and 2 <= len(text) <= 80:
        return True
    if text.startswith("(") and text.endswith(")") and 2 <= len(text) <= 80:
        return True
    return False


def _chunked_page_ranges(total_pages: int, worker_count: int) -> List[tuple[int, int]]:
    if total_pages <= 0:
        return []
    resolved_workers = max(1, min(worker_count, total_pages))
    chunk_size = max(1, (total_pages + resolved_workers - 1) // resolved_workers)
    return [
        (start, min(total_pages, start + chunk_size))
        for start in range(0, total_pages, chunk_size)
    ]


def _extract_blocks_chunk(pdf_path: str, page_range: tuple[int, int]) -> List[dict]:
    start, end = page_range
    doc = fitz.open(pdf_path)
    blocks = []

    for page_index in range(start, end):
        page_text = doc.load_page(page_index).get_text("dict")
        for block in page_text["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                text = " ".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                blocks.append(
                    {
                        "text": text,
                        "font_size": round(max(s["size"] for s in spans), 1),
                        "page_index": page_index + 1,
                    }
                )

    doc.close()
    return blocks


def classify_levels(blocks):
    candidates = [b for b in blocks if is_heading(b["text"])]
    if candidates:
        freq = Counter(b["font_size"] for b in candidates)
    else:
        freq = Counter(b["font_size"] for b in blocks)

    ranked_sizes = sorted(freq.keys(), key=lambda s: (-s, freq[s]))

    levels = {}
    if len(ranked_sizes) > 0:
        levels[ranked_sizes[0]] = "H1"
    if len(ranked_sizes) > 1:
        levels[ranked_sizes[1]] = "H2"
    if len(ranked_sizes) > 2:
        levels[ranked_sizes[2]] = "H3"

    return levels


def extract_blocks(pdf_path: str):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    page_ranges = _chunked_page_ranges(total_pages, PDF_PAGE_WORKER_COUNT)
    if len(page_ranges) <= 1:
        return _extract_blocks_chunk(pdf_path, (0, total_pages))

    blocks = []
    with ThreadPoolExecutor(max_workers=min(len(page_ranges), PDF_PAGE_WORKER_COUNT)) as executor:
        for chunk_blocks in executor.map(lambda page_range: _extract_blocks_chunk(pdf_path, page_range), page_ranges):
            blocks.extend(chunk_blocks)
    return blocks


def build_tree_from_pdf(pdf_path: str):
    blocks = extract_blocks(pdf_path)
    level_map = classify_levels(blocks)

    tree = []
    stack = []  # [(level, node)]
    node_counter = 0

    def new_node(title, page_index):
        nonlocal node_counter
        node = {
            "title": title,
            "node_id": f"{node_counter:04d}",
            "page_index": page_index,
            "text": "",
        }
        node_counter += 1
        return node

    def level_rank(level):
        return {"H1": 1, "H2": 2, "H3": 3}.get(level, 99)

    for b in blocks:
        text = b["text"]
        level = level_map.get(b["font_size"]) if (is_heading(text) or b["font_size"] in level_map) else None

        if level:
            node = new_node(text, b["page_index"])

            while stack and level_rank(stack[-1][0]) >= level_rank(level):
                stack.pop()

            if not stack:
                tree.append(node)
            else:
                stack[-1][1].setdefault("nodes", []).append(node)

            stack.append((level, node))
        else:
            if not stack:
                preface = new_node("Preface", b["page_index"])
                tree.append(preface)
                stack.append(("H1", preface))
            current = stack[-1][1]
            current["text"] = (current["text"] + "\n" + text).strip() if current["text"] else text

    def trim(node):
        if "text" in node:
            node["text"] = node["text"].strip()
        for c in node.get("nodes", []):
            trim(c)

    for n in tree:
        trim(n)

    return tree


def summarize_text(text: str, max_len: int = 160) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


def add_summaries(tree_nodes: List[dict]) -> List[dict]:
    for node in tree_nodes:
        text = node.get("text", "")
        if node.get("nodes"):
            prefix = summarize_text(text or node.get("title", ""))
            if prefix:
                node["prefix_summary"] = prefix
            node.pop("summary", None)
            add_summaries(node["nodes"])
        else:
            summary = summarize_text(text or node.get("title", ""))
            if summary:
                node["summary"] = summary
            node.pop("prefix_summary", None)
    return tree_nodes


def strip_summaries(tree_nodes: List[dict]) -> List[dict]:
    for node in tree_nodes:
        node.pop("summary", None)
        node.pop("prefix_summary", None)
        if node.get("nodes"):
            strip_summaries(node["nodes"])
    return tree_nodes


def remove_text_fields(tree_nodes: List[dict]) -> List[dict]:
    for node in tree_nodes:
        node.pop("text", None)
        if node.get("nodes"):
            remove_text_fields(node["nodes"])
    return tree_nodes


def flatten_tree(tree_nodes: List[dict]) -> List[dict]:
    flat = []
    for node in tree_nodes:
        flat.append(node)
        if node.get("nodes"):
            flat.extend(flatten_tree(node["nodes"]))
    return flat


def _extract_ocr_pages_chunk(pdf_path: str, page_range: tuple[int, int]) -> List[dict]:
    start, end = page_range
    doc = fitz.open(pdf_path)
    pages = []
    for page_index in range(start, end):
        text = doc.load_page(page_index).get_text().strip()
        pages.append({"page": page_index + 1, "text": text})
    doc.close()
    return pages


def extract_ocr_pages(pdf_path: str) -> List[dict]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    page_ranges = _chunked_page_ranges(total_pages, PDF_PAGE_WORKER_COUNT)
    if len(page_ranges) <= 1:
        return _extract_ocr_pages_chunk(pdf_path, (0, total_pages))

    pages = []
    with ThreadPoolExecutor(max_workers=min(len(page_ranges), PDF_PAGE_WORKER_COUNT)) as executor:
        for chunk_pages in executor.map(lambda page_range: _extract_ocr_pages_chunk(pdf_path, page_range), page_ranges):
            pages.extend(chunk_pages)
    pages.sort(key=lambda item: item["page"])
    return pages


def extract_ocr_pages_from_json(json_path: str) -> List[dict]:
    payload = load_json_multimodal_payload(json_path)
    return [
        {
            "page": int(item.get("page") or 0),
            "text": str(item.get("text", "")).strip(),
        }
        for item in payload.get("pages", [])
        if int(item.get("page") or 0) > 0
    ]


def extract_multimodal_payload_from_json(json_path: str) -> Dict[str, Any]:
    payload = load_json_multimodal_payload(json_path)
    pages = [
        {
            "page": int(item.get("page") or 0),
            "text": str(item.get("text", "")).strip(),
        }
        for item in payload.get("pages", [])
        if int(item.get("page") or 0) > 0
    ]
    return {
        "document": payload.get("document", {}),
        "ocr_pages": pages,
        "page_metadata": payload.get("page_metadata", {}),
        "evidence_units": payload.get("evidence_units", []),
    }


def extract_multimodal_payload_from_excel(file_path: str) -> Dict[str, Any]:
    payload = load_excel_multimodal_payload(file_path)
    pages = [
        {
            "page": int(item.get("page") or 0),
            "text": str(item.get("text", "")).strip(),
        }
        for item in payload.get("pages", [])
        if int(item.get("page") or 0) > 0
    ]
    return {
        "document": payload.get("document", {}),
        "ocr_pages": pages,
        "page_metadata": payload.get("page_metadata", {}),
        "evidence_units": payload.get("evidence_units", []),
    }


def simple_retrieval(tree_nodes: List[dict], query: str, top_k: int = 5) -> dict:
    tokens = set(re.findall(r"[a-zA-Z0-9']+", query.lower()))
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by", "is", "are", "was", "were"}
    tokens = {t for t in tokens if t not in stop}

    scored = []
    for node in flatten_tree(tree_nodes):
        content = " ".join(
            filter(
                None,
                [
                    node.get("title", ""),
                    node.get("summary", ""),
                    node.get("prefix_summary", ""),
                    node.get("text", ""),
                ],
            )
        )
        node_tokens = set(re.findall(r"[a-zA-Z0-9']+", content.lower()))
        score = len(tokens & node_tokens)
        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        scored = [(0, n) for n in flatten_tree(tree_nodes)[:top_k]]

    top_nodes = [n for _, n in scored[:top_k]]
    reasoning = "Selected nodes with the highest keyword overlap with the query."

    return {
        "thinking": reasoning,
        "node_list": [n.get("node_id") for n in top_nodes if n.get("node_id")],
        "nodes": [
            {
                "node_id": n.get("node_id"),
                "page_index": n.get("page_index"),
                "title": n.get("title"),
            }
            for n in top_nodes
        ],
    }


async def generate_llm_summaries(tree_nodes: List[dict]) -> List[dict]:
    tree_nodes = await generate_summaries_for_structure_md(
        tree_nodes,
        summary_token_threshold=SUMMARY_TOKEN_THRESHOLD,
        model=DEFAULT_OPT.model,
    )
    return tree_nodes


async def build_tree_llm(pdf_path: str, with_summary: bool) -> List[dict]:
    def _run():
        return llm_page_index(
            pdf_path,
            model=DEFAULT_OPT.model,
            toc_check_page_num=DEFAULT_OPT.toc_check_page_num,
            max_page_num_each_node=DEFAULT_OPT.max_page_num_each_node,
            max_token_num_each_node=DEFAULT_OPT.max_token_num_each_node,
            if_add_node_id="yes",
            if_add_node_summary="yes" if with_summary else "no",
            if_add_doc_description="no",
            if_add_node_text="no",
        )

    result = await asyncio.to_thread(_run)
    return result.get("structure", [])


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_paths(file_hash: str):
    summary_path = CACHE_DIR / f"{file_hash}_tree_summary.json"
    no_summary_path = CACHE_DIR / f"{file_hash}_tree_no_summary.json"
    return summary_path, no_summary_path


def _tree_cache_exists(file_hash: Optional[str]) -> bool:
    if not file_hash:
        return False
    summary_path, no_summary_path = _cache_paths(file_hash)
    return summary_path.exists() and no_summary_path.exists()


def _load_cached_tree(file_hash: str):
    summary_path, no_summary_path = _cache_paths(file_hash)
    if summary_path.exists() and no_summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            tree_summary = json.load(f)
        with open(no_summary_path, "r", encoding="utf-8") as f:
            tree_no_summary = json.load(f)
        return tree_summary, tree_no_summary
    return None, None


def _save_cached_tree(file_hash: str, tree_summary: List[dict], tree_no_summary: List[dict]):
    summary_path, no_summary_path = _cache_paths(file_hash)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(tree_summary, f, indent=2, ensure_ascii=False)
    with open(no_summary_path, "w", encoding="utf-8") as f:
        json.dump(tree_no_summary, f, indent=2, ensure_ascii=False)


DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = BACKEND_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DOC_REGISTRY_PATH = DATA_DIR / "documents.json"

DEFAULT_OPT = ConfigLoader().load()
set_llm_defaults_from_opt(DEFAULT_OPT)
SUMMARY_TOKEN_THRESHOLD = 200
TREE_WORKER_COUNT = max(
    1,
    int(getattr(DEFAULT_OPT, "tree_worker_count", os.getenv("PAGEINDEX_TREE_WORKER_COUNT", "2"))),
)
PDF_PAGE_WORKER_COUNT = max(
    1,
    int(getattr(DEFAULT_OPT, "pdf_page_worker_count", os.getenv("PAGEINDEX_PDF_PAGE_WORKER_COUNT", "8"))),
)
PGVECTOR_DSN = _env_str("PGVECTOR_DSN")
PGVECTOR_ENABLED = (_env_str("PGVECTOR_ENABLED", "true" if PGVECTOR_DSN else "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
PGVECTOR_DOCUMENT_TABLE = _env_str("PGVECTOR_DOCUMENT_TABLE", "document_index")
PGVECTOR_SECTION_TABLE = _env_str("PGVECTOR_SECTION_TABLE", "section_index")
PGVECTOR_EMBEDDING_DIMENSIONS = max(1, int(_env_str("PGVECTOR_EMBEDDING_DIMENSIONS", "3072") or "3072"))
URL_INGESTION_ENABLED = _env_bool("URL_INGESTION_ENABLED", True)
URL_DISCOVERY_MAX_URLS = max(1, _env_int("URL_DISCOVERY_MAX_URLS", 25))
URL_DISCOVERY_INCLUDE_CRAWLER = _env_bool("URL_DISCOVERY_INCLUDE_CRAWLER", True)
URL_FETCH_TIMEOUT_SEC = max(5, _env_int("URL_FETCH_TIMEOUT_SEC", 20))
URL_FETCH_MAX_DOWNLOAD_MB = max(1, _env_int("URL_FETCH_MAX_DOWNLOAD_MB", 80))
URL_INCLUDE_COMMENTS = _env_bool("URL_INCLUDE_COMMENTS", True)
URL_INCLUDE_TABLES = _env_bool("URL_INCLUDE_TABLES", True)
URL_USER_AGENT = _env_str("URL_USER_AGENT", "PageIndex/1.0 (mailto:ops@example.com)") or "PageIndex/1.0 (mailto:ops@example.com)"

_embedding_api_key = _env_str("AZURE_OPENAI_EMBEDDING_API_KEY") or _env_str("AZURE_OPENAI_KEY") or _env_str("AZURE_OPENAI_API_KEY")
_embedding_endpoint = _env_str("AZURE_OPENAI_EMBEDDING_ENDPOINT") or _env_str("AZURE_OPENAI_ENDPOINT")
_embedding_api_version = _env_str("AZURE_OPENAI_EMBEDDING_API_VERSION") or _env_str("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
_embedding_deployment = _env_str("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT") or _env_str("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

try:
    EMBEDDING_CLIENT = AzureEmbeddingClient(
        api_key=_embedding_api_key or "",
        endpoint=_embedding_endpoint or "",
        api_version=_embedding_api_version,
        deployment_name=_embedding_deployment or "",
        dimensions=PGVECTOR_EMBEDDING_DIMENSIONS,
    )
except Exception:
    EMBEDDING_CLIENT = None

PGVECTOR_INDEXER = build_pgvector_indexer(
    dsn=PGVECTOR_DSN if PGVECTOR_ENABLED else None,
    embedding_client=EMBEDDING_CLIENT,
    document_table=PGVECTOR_DOCUMENT_TABLE,
    section_table=PGVECTOR_SECTION_TABLE,
    embedding_dimensions=PGVECTOR_EMBEDDING_DIMENSIONS,
)
DOCUMENT_QUEUE = None
WORKER_TASKS = []
PROCESSING_METRICS = {
    "documents_started": 0,
    "documents_completed": 0,
    "documents_failed": 0,
    "queue_wait_total_sec": 0.0,
    "processing_total_sec": 0.0,
    "ocr_extract_total_sec": 0.0,
    "llm_tree_total_sec": 0.0,
    "fallback_tree_total_sec": 0.0,
    "cache_write_total_sec": 0.0,
    "last_updated_at": None,
}


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _resolve_doc_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    normalized = Path(str(path_value).replace("\\", "/"))
    return (BACKEND_DIR / normalized).resolve()


def _storage_path_value(path_value: Optional[str]) -> Optional[str]:
    resolved = _resolve_doc_path(path_value)
    if resolved is None:
        return None
    try:
        return str(resolved.relative_to(BACKEND_DIR))
    except ValueError:
        return str(resolved)


def _index_document_record(doc_id: str, doc: Dict[str, Any], *, force: bool = False) -> bool:
    if PGVECTOR_INDEXER is None:
        return False
    if doc.get("status") != "completed" or not doc.get("retrieval_ready"):
        return False
    if doc.get("retrieval_indexed_at") and not force:
        return False

    loaded = _ensure_doc_loaded(doc_id)
    if not loaded:
        return False

    tree_summary = loaded.get("tree_llm_summary") or loaded.get("tree") or []
    ocr_pages = loaded.get("ocr_pages") or []
    file_hash = str(loaded.get("file_hash") or "")
    if not tree_summary or not file_hash:
        return False

    retrieval_summary = loaded.get("retrieval_summary")
    retrieval_headings = loaded.get("retrieval_headings")
    retrieval_sections = loaded.get("retrieval_sections")

    if not retrieval_summary or not retrieval_headings:
        retrieval_summary, retrieval_headings = build_document_summary(
            str(loaded.get("name", "unknown.pdf")),
            tree_summary,
            ocr_pages,
        )
        loaded["retrieval_summary"] = retrieval_summary
        loaded["retrieval_headings"] = retrieval_headings

    if not retrieval_sections:
        retrieval_sections = build_section_records(doc_id, tree_summary)
        loaded["retrieval_sections"] = retrieval_sections

    source_path = _resolve_doc_path(loaded.get("path"))
    index_metadata: Dict[str, Any] = {
        "page_count": loaded.get("pageNum"),
        "source_path": str(source_path) if source_path else str(loaded.get("path", "")),
    }
    if loaded.get("source_url"):
        index_metadata["source_url"] = loaded.get("source_url")
    if loaded.get("resolved_url"):
        index_metadata["resolved_url"] = loaded.get("resolved_url")
    if loaded.get("source_type"):
        index_metadata["source_type"] = loaded.get("source_type")

    index_started = time.perf_counter()
    PGVECTOR_INDEXER.upsert_document(
        doc_id=doc_id,
        doc_name=str(loaded.get("name", "unknown.pdf")),
        file_hash=file_hash,
        created_at=float(loaded.get("createdAt") or time.time()),
        updated_at=time.time(),
        doc_summary=str(retrieval_summary),
        headings=retrieval_headings or [],
        metadata=index_metadata,
        section_records=[{**item, "doc_id": doc_id} for item in (retrieval_sections or [])],
    )
    loaded["retrieval_indexed_at"] = time.time()
    loaded.setdefault("timings", {})["retrieval_index_sec"] = _round_metric(time.perf_counter() - index_started)
    return True


def _backfill_pgvector_index(*, force: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
    if PGVECTOR_INDEXER is None:
        return {"enabled": False, "indexed": 0, "skipped": 0, "errors": []}

    indexed = 0
    skipped = 0
    errors: List[Dict[str, str]] = []
    for doc_id, doc in DOCUMENTS.items():
        if limit is not None and indexed >= limit:
            break
        try:
            changed = _index_document_record(doc_id, doc, force=force)
            if changed:
                indexed += 1
            else:
                skipped += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"doc_id": doc_id, "error": str(exc)})
    if indexed:
        _save_registry()
    return {"enabled": True, "indexed": indexed, "skipped": skipped, "errors": errors}


def _status_counts() -> Dict[str, int]:
    counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
    for doc in DOCUMENTS.values():
        status = doc.get("status", "queued")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _processing_stats_payload() -> Dict[str, Any]:
    completed = PROCESSING_METRICS["documents_completed"]
    failed = PROCESSING_METRICS["documents_failed"]
    finished = completed + failed
    return {
        "worker_count": TREE_WORKER_COUNT,
        "queue_size": DOCUMENT_QUEUE.qsize() if DOCUMENT_QUEUE is not None else 0,
        "status_counts": _status_counts(),
        "metrics": {
            "documents_started": PROCESSING_METRICS["documents_started"],
            "documents_completed": completed,
            "documents_failed": failed,
            "avg_queue_wait_sec": _round_metric(
                PROCESSING_METRICS["queue_wait_total_sec"] / PROCESSING_METRICS["documents_started"]
            )
            if PROCESSING_METRICS["documents_started"]
            else 0.0,
            "avg_processing_sec": _round_metric(
                PROCESSING_METRICS["processing_total_sec"] / finished
            )
            if finished
            else 0.0,
            "avg_ocr_extract_sec": _round_metric(
                PROCESSING_METRICS["ocr_extract_total_sec"] / finished
            )
            if finished
            else 0.0,
            "avg_llm_tree_sec": _round_metric(
                PROCESSING_METRICS["llm_tree_total_sec"] / finished
            )
            if finished
            else 0.0,
            "avg_fallback_tree_sec": _round_metric(
                PROCESSING_METRICS["fallback_tree_total_sec"] / finished
            )
            if finished
            else 0.0,
            "avg_cache_write_sec": _round_metric(
                PROCESSING_METRICS["cache_write_total_sec"] / finished
            )
            if finished
            else 0.0,
            "last_updated_at": PROCESSING_METRICS["last_updated_at"],
        },
    }


def _load_registry():
    if not DOC_REGISTRY_PATH.exists():
        return {}
    try:
        with open(DOC_REGISTRY_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Build initial list keyed by id
        docs_by_id = {}
        for item in raw:
            doc_id = item.get("id")
            if doc_id:
                resolved_path = _resolve_doc_path(item.get("path"))
                if resolved_path is not None:
                    item["path"] = str(resolved_path)
                if not item.get("input_type"):
                    suffix = (resolved_path.suffix.lower() if resolved_path else Path(str(item.get("path", ""))).suffix.lower())
                    if suffix == ".json":
                        item["input_type"] = "json"
                    elif suffix in {".xlsx", ".csv"}:
                        item["input_type"] = suffix.lstrip(".")
                    else:
                        item["input_type"] = "pdf"
                docs_by_id[doc_id] = item

        # Deduplicate by file_hash: prefer entries that are retrieval_ready, have existing path, or most recent
        filehash_map = {}
        for doc in docs_by_id.values():
            fh = doc.get("file_hash")
            if not fh:
                # keep docs without a file_hash using their id
                filehash_map.setdefault(None, {})[doc.get("id")] = doc
                continue

            existing = filehash_map.get(fh)
            if not existing:
                filehash_map[fh] = doc
                continue

            # choose between existing and doc
            def score_candidate(d):
                score = 0
                if d.get("retrieval_ready"):
                    score += 10
                try:
                    resolved = _resolve_doc_path(d.get("path"))
                    if resolved and resolved.exists():
                        score += 5
                except Exception:
                    pass
                # prefer more recent createdAt
                score += (d.get("createdAt") or 0) / (60 * 60 * 24 * 365 + 1)
                return score

            if score_candidate(doc) > score_candidate(existing):
                filehash_map[fh] = doc

        # Reconstruct docs dict keyed by chosen id
        deduped = {}
        for fh, v in filehash_map.items():
            if fh is None:
                # multiple no-hash docs kept by id
                for _id, doc in v.items():
                    deduped[_id] = doc
            else:
                if v and v.get("id"):
                    deduped[v.get("id")] = v

        return deduped
    except Exception:
        return {}


def _save_registry():
    payload = []
    for doc in DOCUMENTS.values():
        payload.append(
            {
                "id": doc.get("id"),
                "name": doc.get("name"),
                "path": _storage_path_value(doc.get("path")),
                "status": doc.get("status"),
                "input_type": doc.get("input_type", "pdf"),
                "retrieval_ready": doc.get("retrieval_ready"),
                "createdAt": doc.get("createdAt"),
                "queuedAt": doc.get("queuedAt"),
                "startedAt": doc.get("startedAt"),
                "completedAt": doc.get("completedAt"),
                "pageNum": doc.get("pageNum"),
                "file_hash": doc.get("file_hash"),
                "retrieval_summary": doc.get("retrieval_summary"),
                "retrieval_headings": doc.get("retrieval_headings"),
                "retrieval_indexed_at": doc.get("retrieval_indexed_at"),
                "source_url": doc.get("source_url"),
                "source_url_normalized": doc.get("source_url_normalized"),
                "resolved_url": doc.get("resolved_url"),
                "source_type": doc.get("source_type"),
                "source_metadata": doc.get("source_metadata"),
                "error": doc.get("error"),
                "timings": doc.get("timings"),
            }
        )
    with open(DOC_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


async def _build_document_payload_from_pdf(pdf_path: str, file_hash: str) -> dict:
    timing_started = time.perf_counter()

    ocr_started = time.perf_counter()
    ocr_pages = await asyncio.to_thread(extract_ocr_pages, pdf_path)
    ocr_extract_sec = time.perf_counter() - ocr_started
    page_num = len(ocr_pages)

    llm_tree_sec = 0.0
    cache_write_sec = 0.0
    cached_summary, cached_no_summary = _load_cached_tree(file_hash)
    if cached_summary is not None and cached_no_summary is not None:
        tree_llm_summary = cached_summary
        tree_llm_no_summary = cached_no_summary
    else:
        llm_started = time.perf_counter()
        tree_llm_summary = await build_tree_llm(pdf_path, with_summary=True)
        llm_tree_sec = time.perf_counter() - llm_started
        tree_llm_no_summary = strip_summaries(copy.deepcopy(tree_llm_summary))
        cache_started = time.perf_counter()
        await asyncio.to_thread(_save_cached_tree, file_hash, tree_llm_summary, tree_llm_no_summary)
        cache_write_sec = time.perf_counter() - cache_started

    fallback_started = time.perf_counter()
    tree_fallback = await asyncio.to_thread(build_tree_from_pdf, pdf_path)
    fallback_tree_sec = time.perf_counter() - fallback_started
    retrieval_summary, retrieval_headings = build_document_summary(
        Path(pdf_path).name,
        tree_llm_summary or tree_fallback,
        ocr_pages,
    )
    retrieval_sections = build_section_records(file_hash, tree_llm_summary or tree_fallback)
    return {
        "pageNum": page_num,
        "tree": tree_fallback,
        "ocr_pages": ocr_pages,
        "tree_llm_summary": tree_llm_summary,
        "tree_llm_no_summary": tree_llm_no_summary,
        "retrieval_summary": retrieval_summary,
        "retrieval_headings": retrieval_headings,
        "retrieval_sections": retrieval_sections,
        "timings": {
            "ocr_extract_sec": _round_metric(ocr_extract_sec),
            "llm_tree_sec": _round_metric(llm_tree_sec),
            "fallback_tree_sec": _round_metric(fallback_tree_sec),
            "cache_write_sec": _round_metric(cache_write_sec),
            "payload_total_sec": _round_metric(time.perf_counter() - timing_started),
        },
    }


async def _build_document_payload_from_json(json_path: str, file_hash: str) -> dict:
    timing_started = time.perf_counter()

    ocr_started = time.perf_counter()
    multimodal_payload = await asyncio.to_thread(extract_multimodal_payload_from_json, json_path)
    ocr_pages = multimodal_payload.get("ocr_pages", [])
    ocr_extract_sec = time.perf_counter() - ocr_started
    page_num = len(ocr_pages)

    llm_tree_sec = 0.0
    cache_write_sec = 0.0
    cached_summary, cached_no_summary = _load_cached_tree(file_hash)
    if cached_summary is not None and cached_no_summary is not None:
        tree_llm_summary = cached_summary
        tree_llm_no_summary = cached_no_summary
    else:
        llm_started = time.perf_counter()
        tree_llm_summary = await build_tree_llm(json_path, with_summary=True)
        llm_tree_sec = time.perf_counter() - llm_started
        tree_llm_no_summary = strip_summaries(copy.deepcopy(tree_llm_summary))
        cache_started = time.perf_counter()
        await asyncio.to_thread(_save_cached_tree, file_hash, tree_llm_summary, tree_llm_no_summary)
        cache_write_sec = time.perf_counter() - cache_started

    # JSON sources do not have PDF font-size structure; use LLM tree output as fallback tree.
    tree_fallback = copy.deepcopy(tree_llm_no_summary)
    fallback_tree_sec = 0.0
    retrieval_summary, retrieval_headings = build_document_summary(
        Path(json_path).name,
        tree_llm_summary or tree_fallback,
        ocr_pages,
    )
    retrieval_sections = build_section_records(file_hash, tree_llm_summary or tree_fallback)
    return {
        "pageNum": page_num,
        "tree": tree_fallback,
        "ocr_pages": ocr_pages,
        "document": multimodal_payload.get("document", {}),
        "page_metadata": multimodal_payload.get("page_metadata", {}),
        "evidence_units": multimodal_payload.get("evidence_units", []),
        "tree_llm_summary": tree_llm_summary,
        "tree_llm_no_summary": tree_llm_no_summary,
        "retrieval_summary": retrieval_summary,
        "retrieval_headings": retrieval_headings,
        "retrieval_sections": retrieval_sections,
        "timings": {
            "ocr_extract_sec": _round_metric(ocr_extract_sec),
            "llm_tree_sec": _round_metric(llm_tree_sec),
            "fallback_tree_sec": _round_metric(fallback_tree_sec),
            "cache_write_sec": _round_metric(cache_write_sec),
            "payload_total_sec": _round_metric(time.perf_counter() - timing_started),
        },
    }


async def _build_document_payload_from_excel(file_path: str, file_hash: str) -> dict:
    timing_started = time.perf_counter()

    ocr_started = time.perf_counter()
    multimodal_payload = await asyncio.to_thread(extract_multimodal_payload_from_excel, file_path)
    ocr_pages = multimodal_payload.get("ocr_pages", [])
    ocr_extract_sec = time.perf_counter() - ocr_started
    page_num = len(ocr_pages)

    llm_tree_sec = 0.0
    cache_write_sec = 0.0
    cached_summary, cached_no_summary = _load_cached_tree(file_hash)
    if cached_summary is not None and cached_no_summary is not None:
        tree_llm_summary = cached_summary
        tree_llm_no_summary = cached_no_summary
    else:
        llm_started = time.perf_counter()
        llm_input_path = await asyncio.to_thread(create_excel_toc_input_file, ocr_pages)
        try:
            tree_llm_summary = await build_tree_llm(llm_input_path, with_summary=True)
        finally:
            with contextlib.suppress(OSError):
                Path(llm_input_path).unlink()
        llm_tree_sec = time.perf_counter() - llm_started
        tree_llm_no_summary = strip_summaries(copy.deepcopy(tree_llm_summary))
        cache_started = time.perf_counter()
        await asyncio.to_thread(_save_cached_tree, file_hash, tree_llm_summary, tree_llm_no_summary)
        cache_write_sec = time.perf_counter() - cache_started

    # Spreadsheet sources use LLM TOC output as fallback tree.
    tree_fallback = copy.deepcopy(tree_llm_no_summary)
    fallback_tree_sec = 0.0
    retrieval_summary, retrieval_headings = build_document_summary(
        Path(file_path).name,
        tree_llm_summary or tree_fallback,
        ocr_pages,
    )
    retrieval_sections = build_section_records(file_hash, tree_llm_summary or tree_fallback)
    return {
        "pageNum": page_num,
        "tree": tree_fallback,
        "ocr_pages": ocr_pages,
        "document": multimodal_payload.get("document", {}),
        "page_metadata": multimodal_payload.get("page_metadata", {}),
        "evidence_units": multimodal_payload.get("evidence_units", []),
        "tree_llm_summary": tree_llm_summary,
        "tree_llm_no_summary": tree_llm_no_summary,
        "retrieval_summary": retrieval_summary,
        "retrieval_headings": retrieval_headings,
        "retrieval_sections": retrieval_sections,
        "timings": {
            "ocr_extract_sec": _round_metric(ocr_extract_sec),
            "llm_tree_sec": _round_metric(llm_tree_sec),
            "fallback_tree_sec": _round_metric(fallback_tree_sec),
            "cache_write_sec": _round_metric(cache_write_sec),
            "payload_total_sec": _round_metric(time.perf_counter() - timing_started),
        },
    }


async def _process_document(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        return

    doc_path = _resolve_doc_path(doc.get("path"))
    if not doc_path or not doc_path.exists():
        doc["status"] = "failed"
        doc["retrieval_ready"] = False
        doc["error"] = "Document file not found"
        _save_registry()
        return

    doc["path"] = str(doc_path)
    input_type = str(doc.get("input_type") or "").strip().lower()
    if input_type not in {"pdf", "json", "xlsx", "csv"}:
        suffix = doc_path.suffix.lower()
        if suffix == ".json":
            input_type = "json"
        elif suffix in {".xlsx", ".csv"}:
            input_type = suffix.lstrip(".")
        else:
            input_type = "pdf"
    doc["input_type"] = input_type

    doc["status"] = "processing"
    doc["retrieval_ready"] = False
    doc.pop("error", None)
    doc["startedAt"] = time.time()
    _save_registry()

    queued_at = doc.get("queuedAt") or doc.get("createdAt") or time.time()
    queue_wait_sec = max(0.0, time.time() - queued_at)
    started = time.perf_counter()
    PROCESSING_METRICS["documents_started"] += 1
    PROCESSING_METRICS["queue_wait_total_sec"] += queue_wait_sec
    PROCESSING_METRICS["last_updated_at"] = time.time()

    try:
        file_hash = doc.get("file_hash") or await asyncio.to_thread(_hash_file, doc_path)
        if input_type == "json":
            payload = await _build_document_payload_from_json(str(doc_path), file_hash)
        elif input_type in {"xlsx", "csv"}:
            payload = await _build_document_payload_from_excel(str(doc_path), file_hash)
        else:
            payload = await _build_document_payload_from_pdf(str(doc_path), file_hash)
        doc.update(payload)
        doc["file_hash"] = file_hash
        if PGVECTOR_INDEXER is not None:
            index_started = time.perf_counter()
            index_metadata: Dict[str, Any] = {
                "page_count": payload.get("pageNum"),
                "source_path": str(doc_path),
            }
            if doc.get("source_url"):
                index_metadata["source_url"] = doc.get("source_url")
            if doc.get("resolved_url"):
                index_metadata["resolved_url"] = doc.get("resolved_url")
            if doc.get("source_type"):
                index_metadata["source_type"] = doc.get("source_type")

            await asyncio.to_thread(
                PGVECTOR_INDEXER.upsert_document,
                doc_id=doc_id,
                doc_name=str(doc.get("name", "unknown.pdf")),
                file_hash=file_hash,
                created_at=float(doc.get("createdAt") or time.time()),
                updated_at=time.time(),
                doc_summary=str(payload.get("retrieval_summary", "")),
                headings=payload.get("retrieval_headings", []) or [],
                metadata=index_metadata,
                section_records=[
                    {**item, "doc_id": doc_id}
                    for item in (payload.get("retrieval_sections", []) or [])
                ],
            )
            doc["retrieval_indexed_at"] = time.time()
            doc.setdefault("timings", {})["retrieval_index_sec"] = _round_metric(time.perf_counter() - index_started)
        doc["status"] = "completed"
        doc["retrieval_ready"] = True
        doc.pop("error", None)
        doc["completedAt"] = time.time()
        total_processing_sec = time.perf_counter() - started
        doc.setdefault("timings", {})["queue_wait_sec"] = _round_metric(queue_wait_sec)
        doc["timings"]["total_processing_sec"] = _round_metric(total_processing_sec)
        PROCESSING_METRICS["documents_completed"] += 1
        PROCESSING_METRICS["processing_total_sec"] += total_processing_sec
        PROCESSING_METRICS["ocr_extract_total_sec"] += payload.get("timings", {}).get("ocr_extract_sec", 0.0)
        PROCESSING_METRICS["llm_tree_total_sec"] += payload.get("timings", {}).get("llm_tree_sec", 0.0)
        PROCESSING_METRICS["fallback_tree_total_sec"] += payload.get("timings", {}).get("fallback_tree_sec", 0.0)
        PROCESSING_METRICS["cache_write_total_sec"] += payload.get("timings", {}).get("cache_write_sec", 0.0)
    except Exception as exc:
        total_processing_sec = time.perf_counter() - started
        doc["status"] = "failed"
        doc["retrieval_ready"] = False
        doc["error"] = str(exc)
        doc["completedAt"] = time.time()
        doc.setdefault("timings", {})["queue_wait_sec"] = _round_metric(queue_wait_sec)
        doc["timings"]["total_processing_sec"] = _round_metric(total_processing_sec)
        PROCESSING_METRICS["documents_failed"] += 1
        PROCESSING_METRICS["processing_total_sec"] += total_processing_sec

    PROCESSING_METRICS["last_updated_at"] = time.time()
    _save_registry()


async def _tree_generation_worker(worker_index: int):
    while True:
        doc_id = await DOCUMENT_QUEUE.get()
        try:
            if doc_id is None:
                return
            await _process_document(doc_id)
        except Exception:
            doc = DOCUMENTS.get(doc_id)
            if doc:
                doc["status"] = "failed"
                doc["retrieval_ready"] = False
                doc["error"] = f"Unhandled worker failure in worker {worker_index}"
                _save_registry()
        finally:
            DOCUMENT_QUEUE.task_done()


def _enqueue_pending_documents():
    for doc_id, doc in DOCUMENTS.items():
        resolved_path = _resolve_doc_path(doc.get("path"))
        if doc.get("status") in {"queued", "processing"} and resolved_path and resolved_path.exists():
            DOCUMENT_QUEUE.put_nowait(doc_id)


def _maybe_requeue_missing_cache(doc: Dict[str, Any]) -> bool:
    if not doc:
        return False
    if doc.get("status") != "completed":
        return False
    if _tree_cache_exists(doc.get("file_hash")):
        return False
    doc_path = _resolve_doc_path(doc.get("path"))
    if not doc_path or not doc_path.exists():
        doc["status"] = "failed"
        doc["retrieval_ready"] = False
        doc["error"] = "Document cache missing and source document not found"
        _save_registry()
        return True

    doc["status"] = "queued"
    doc["retrieval_ready"] = False
    doc["queuedAt"] = time.time()
    doc["error"] = "Tree cache missing; document queued for regeneration"
    doc.pop("tree", None)
    doc.pop("ocr_pages", None)
    doc.pop("tree_llm_summary", None)
    doc.pop("tree_llm_no_summary", None)
    if DOCUMENT_QUEUE is not None:
        DOCUMENT_QUEUE.put_nowait(doc["id"])
    _save_registry()
    return True


def _ensure_doc_loaded(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        return None
    _maybe_requeue_missing_cache(doc)
    if doc.get("status") == "completed" and doc.get("file_hash") and (
        "tree_llm_summary" not in doc or "tree_llm_no_summary" not in doc
    ):
        cached_summary, cached_no_summary = _load_cached_tree(doc["file_hash"])
        if cached_summary is not None and cached_no_summary is not None:
            doc["tree_llm_summary"] = cached_summary
            doc["tree_llm_no_summary"] = cached_no_summary
    resolved_path = _resolve_doc_path(doc.get("path"))
    if doc.get("status") == "completed" and "ocr_pages" not in doc and resolved_path and resolved_path.exists():
        input_type = str(doc.get("input_type") or "").strip().lower()
        suffix = resolved_path.suffix.lower()
        if input_type == "json" or resolved_path.suffix.lower() == ".json":
            multimodal_payload = extract_multimodal_payload_from_json(str(resolved_path))
            doc["ocr_pages"] = multimodal_payload.get("ocr_pages", [])
            doc["document"] = multimodal_payload.get("document", {})
            doc["page_metadata"] = multimodal_payload.get("page_metadata", {})
            doc["evidence_units"] = multimodal_payload.get("evidence_units", [])
            doc["input_type"] = "json"
        elif input_type in {"xlsx", "csv"} or suffix in {".xlsx", ".csv"}:
            multimodal_payload = extract_multimodal_payload_from_excel(str(resolved_path))
            doc["ocr_pages"] = multimodal_payload.get("ocr_pages", [])
            doc["document"] = multimodal_payload.get("document", {})
            doc["page_metadata"] = multimodal_payload.get("page_metadata", {})
            doc["evidence_units"] = multimodal_payload.get("evidence_units", [])
            doc["input_type"] = suffix.lstrip(".") if suffix in {".xlsx", ".csv"} else input_type
        else:
            doc["ocr_pages"] = extract_ocr_pages(str(resolved_path))
            doc["input_type"] = "pdf"
        doc["pageNum"] = len(doc["ocr_pages"])
    return doc


def _find_existing_by_source_url(source_url_normalized: str) -> Optional[Dict[str, Any]]:
    target = source_url_normalized.strip().lower()
    for doc in DOCUMENTS.values():
        current = str(doc.get("source_url_normalized") or "").strip().lower()
        if current != target:
            continue
        if doc.get("status") in {"queued", "processing", "completed"}:
            return doc
    return None


async def _submit_url_for_processing(
    source_url: str,
    *,
    discover: bool,
    max_urls: Optional[int] = None,
) -> Dict[str, Any]:
    if not URL_INGESTION_ENABLED:
        raise HTTPException(status_code=503, detail="URL ingestion is disabled on this server.")

    normalized_url = normalize_source_url(source_url)
    existing = _find_existing_by_source_url(normalized_url)
    if existing:
        return {
            "doc_id": existing.get("id"),
            "status": existing.get("status", "duplicate"),
            "duplicate": True,
            "source_type": existing.get("source_type"),
        }

    doc_id = str(uuid.uuid4())
    resolved_max_urls = max(1, int(max_urls or URL_DISCOVERY_MAX_URLS))

    try:
        ingest_result = await asyncio.to_thread(
            ingest_url_to_local_artifact,
            source_url,
            DATA_DIR,
            doc_id,
            timeout_sec=URL_FETCH_TIMEOUT_SEC,
            max_download_mb=URL_FETCH_MAX_DOWNLOAD_MB,
            user_agent=URL_USER_AGENT,
            discover=discover,
            max_urls=resolved_max_urls,
            include_comments=URL_INCLUDE_COMMENTS,
            include_tables=URL_INCLUDE_TABLES,
            include_crawler=URL_DISCOVERY_INCLUDE_CRAWLER,
        )
    except URLIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to ingest URL content: {exc}") from exc

    try:
        file_hash = await asyncio.to_thread(_hash_file, ingest_result.path)
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(OSError):
            if ingest_result.path.exists():
                ingest_result.path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to hash URL artifact: {exc}") from exc

    existing_by_hash = next((d for d in DOCUMENTS.values() if d.get("file_hash") == file_hash), None)
    if existing_by_hash:
        with contextlib.suppress(OSError):
            if ingest_result.path.exists():
                ingest_result.path.unlink()
        return {
            "doc_id": existing_by_hash.get("id"),
            "status": existing_by_hash.get("status", "duplicate"),
            "duplicate": True,
            "source_type": existing_by_hash.get("source_type") or ingest_result.source_type,
        }

    DOCUMENTS[doc_id] = {
        "id": doc_id,
        "name": ingest_result.name,
        "path": str(ingest_result.path),
        "input_type": ingest_result.input_type,
        "file_hash": file_hash,
        "status": "queued",
        "retrieval_ready": False,
        "createdAt": time.time(),
        "queuedAt": time.time(),
        "source_url": ingest_result.source_url,
        "source_url_normalized": ingest_result.source_url_normalized,
        "resolved_url": ingest_result.resolved_url,
        "source_type": ingest_result.source_type,
        "source_metadata": ingest_result.source_metadata,
    }
    _save_registry()
    if DOCUMENT_QUEUE is not None:
        await DOCUMENT_QUEUE.put(doc_id)

    return {
        "doc_id": doc_id,
        "status": "queued",
        "duplicate": False,
        "source_type": ingest_result.source_type,
    }


app = FastAPI(title="PageIndex Backend", version="local-dev")

DOCUMENTS = _load_registry()
RETRIEVALS = {}


@app.on_event("startup")
async def startup_event():
    global DOCUMENT_QUEUE
    global WORKER_TASKS

    if DOCUMENT_QUEUE is None:
        DOCUMENT_QUEUE = asyncio.Queue()
    if not WORKER_TASKS:
        WORKER_TASKS = [
            asyncio.create_task(_tree_generation_worker(worker_index))
            for worker_index in range(TREE_WORKER_COUNT)
        ]
        _enqueue_pending_documents()


@app.on_event("shutdown")
async def shutdown_event():
    global WORKER_TASKS

    if DOCUMENT_QUEUE is not None:
        for _ in WORKER_TASKS:
            DOCUMENT_QUEUE.put_nowait(None)

    for task in WORKER_TASKS:
        with contextlib.suppress(Exception):
            await task
    WORKER_TASKS = []

UI_DIST_DIR = Path(__file__).parent / "ui-react" / "dist"
if UI_DIST_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIST_DIR), html=True), name="ui")
else:
    @app.get("/ui")
    def ui_placeholder():
        return {
            "status": "ui_not_built",
            "message": "UI not built yet. Run npm install && npm run build in pageindex_backend/ui-react.",
        }


@app.get("/")
def root():
    return {"status": "ok", "service": "pageindex"}


@app.get("/processing/stats/")
def get_processing_stats():
    return _processing_stats_payload()


@app.post("/admin/reindex/pgvector/")
def reindex_pgvector(force: bool = False, limit: Optional[int] = None):
    return _backfill_pgvector_index(force=force, limit=limit)


@app.post("/doc/")
async def submit_document(
    file: UploadFile = File(...),
    if_retrieval: bool = Form(True),
    mode: Optional[str] = Form(None),
    input_type: Optional[str] = Form(None),
):
    doc_id = str(uuid.uuid4())
    raw_filename = str(file.filename or "document")
    inferred_input_type = Path(raw_filename).suffix.lower().lstrip(".")
    resolved_input_type = str(input_type or inferred_input_type).strip().lower()
    if resolved_input_type not in {"pdf", "json", "xlsx", "csv"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, JSON, XLSX, or CSV.")

    extension_by_type = {
        "pdf": ".pdf",
        "json": ".json",
        "xlsx": ".xlsx",
        "csv": ".csv",
    }
    file_extension = extension_by_type[resolved_input_type]
    doc_path = DATA_DIR / f"{doc_id}{file_extension}"
    file_bytes = await file.read()

    if resolved_input_type == "json":
        try:
            json.loads(file_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    with open(doc_path, "wb") as f:
        f.write(file_bytes)

    # compute file hash early and avoid creating duplicate registry entries
    try:
        file_hash = _hash_file(doc_path)
        # If we've already processed this exact file (same hash), return existing doc_id
        existing = next((d for d in DOCUMENTS.values() if d.get("file_hash") == file_hash), None)
        if existing:
            # Remove the newly uploaded duplicate file to avoid clutter
            try:
                if doc_path.exists():
                    doc_path.unlink()
            except Exception:
                pass

            return {
                "doc_id": existing.get("id"),
                "status": existing.get("status", "duplicate"),
                "duplicate": True,
            }
    except Exception as exc:
        DOCUMENTS[doc_id] = {
            "id": doc_id,
            "name": file.filename,
            "path": str(doc_path),
            "input_type": resolved_input_type,
            "status": "failed",
            "retrieval_ready": False,
            "error": str(exc),
            "createdAt": time.time(),
        }
        return {"doc_id": doc_id}

    DOCUMENTS[doc_id] = {
        "id": doc_id,
        "name": file.filename,
        "path": str(doc_path),
        "input_type": resolved_input_type,
        "file_hash": file_hash,
        "status": "queued",
        "retrieval_ready": False,
        "createdAt": time.time(),
        "queuedAt": time.time(),
    }
    _save_registry()
    if DOCUMENT_QUEUE is not None:
        await DOCUMENT_QUEUE.put(doc_id)

    return {"doc_id": doc_id, "status": "queued"}


@app.post("/url/")
async def submit_url(payload: Dict[str, Any]):
    source_url = str(payload.get("url") or "").strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="`url` is required.")
    return await _submit_url_for_processing(source_url, discover=False)


@app.post("/urls/discover/")
async def submit_url_discovery(payload: Dict[str, Any]):
    source_url = str(payload.get("url") or "").strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="`url` is required.")
    max_urls = payload.get("max_urls")
    try:
        max_urls_value = int(max_urls) if max_urls is not None else URL_DISCOVERY_MAX_URLS
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="`max_urls` must be an integer.") from exc
    return await _submit_url_for_processing(source_url, discover=True, max_urls=max_urls_value)


@app.get("/doc/{doc_id}/")
async def get_doc(
    doc_id: str,
    type: str = Query(...),
    format: Optional[str] = None,
    summary: Optional[bool] = None,
):
    doc = _ensure_doc_loaded(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if type == "ocr":
        if format not in {None, "page", "node"}:
            raise HTTPException(status_code=400, detail="format must be 'page' or 'node'")
        pages = doc.get("ocr_pages", [])
        if format in (None, "page"):
            return {"status": doc.get("status", "completed"), "pages": pages}

        tree = doc.get("tree", [])
        return {
            "status": doc.get("status", "completed"),
            "nodes": [
                {
                    "node_id": n.get("node_id"),
                    "title": n.get("title"),
                    "page_index": n.get("page_index"),
                    "text": n.get("text", ""),
                }
                for n in flatten_tree(tree)
            ],
        }

    if type == "tree":
        if summary:
            result = copy.deepcopy(doc.get("tree_llm_summary", []))
        else:
            result = copy.deepcopy(doc.get("tree_llm_no_summary", []))

        return {
            "status": doc.get("status", "queued"),
            "retrieval_ready": doc.get("retrieval_ready", False),
            "result": result,
        }

    raise HTTPException(status_code=400, detail="Invalid type")


@app.get("/doc/{doc_id}/metadata/")
def get_metadata(doc_id: str):
    doc = _ensure_doc_loaded(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/doc/{doc_id}/")
def delete_doc(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")
    DOCUMENTS.pop(doc_id, None)
    _save_registry()
    return {"status": "deleted"}


@app.get("/docs/")
def list_docs(limit: int = 50, offset: int = 0):
    for doc in DOCUMENTS.values():
        _maybe_requeue_missing_cache(doc)
    docs = list(DOCUMENTS.values())[offset : offset + limit]
    return {"documents": docs, "total": len(DOCUMENTS), "limit": limit, "offset": offset}


@app.get("/doc/{doc_id}/pages/")
def get_page_content(doc_id: str, pages: str = Query(...)):
    doc = _ensure_doc_loaded(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    ocr_pages = doc.get("ocr_pages", [])
    total_pages = len(ocr_pages)

    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", pages)
    if not m:
        raise HTTPException(status_code=400, detail="pages must be in 'start-end' format")
    start_page = int(m.group(1))
    end_page = int(m.group(2))
    if start_page < 1 or end_page < 1 or start_page > end_page or end_page > total_pages:
        raise HTTPException(status_code=400, detail="pages out of range")

    content = [
        {"page": p["page"], "text": p["text"]}
        for p in ocr_pages[start_page - 1 : end_page]
    ]

    return {
        "success": True,
        "doc_name": doc.get("name"),
        "content": content,
        "total_pages": total_pages,
        "requested_pages": pages,
        "returned_pages": f"{start_page}-{end_page}",
        "next_steps": {
            "summary": f"Successfully retrieved content for {len(content)} pages.",
            "options": [
                "Use get_document_structure() to understand document organization",
                "Extract specific content sections for analysis",
                "Request additional pages as needed",
                "When citing, use SINGLE page numbers: <doc=x.pdf;page=1> (NOT page=1-3)",
            ],
        },
    }


@app.post("/retrieval/")
def submit_retrieval(payload: dict):
    doc_id = payload.get("doc_id")
    doc = _ensure_doc_loaded(doc_id) if doc_id else None
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.get("retrieval_ready"):
        raise HTTPException(status_code=409, detail="Document is still processing")

    retrieval_id = str(uuid.uuid4())
    tree = doc.get("tree_llm_summary", []) or doc.get("tree", [])
    retrieval_result = simple_retrieval(tree, payload.get("query", ""))
    RETRIEVALS[retrieval_id] = {
        "id": retrieval_id,
        "status": "completed",
        "doc_id": doc_id,
        "query": payload.get("query"),
        "thinking": retrieval_result["thinking"],
        "node_list": retrieval_result["node_list"],
        "nodes": retrieval_result["nodes"],
    }
    return {"retrieval_id": retrieval_id}


@app.get("/retrieval/{retrieval_id}/")
def get_retrieval(retrieval_id: str):
    if retrieval_id not in RETRIEVALS:
        raise HTTPException(status_code=404, detail="Not found")
    return RETRIEVALS[retrieval_id]


@app.post("/chat/completions/")
def chat_completions(payload: dict):
    stream = payload.get("stream", False)

    if not stream:
        return {"choices": [{"message": {"role": "assistant", "content": "Hello from PageIndex backend"}}]}

    def event_stream():
        for chunk in ["Hello ", "from ", "PageIndex ", "stream"]:
            yield f"data: {json.dumps({'choices':[{'delta':{'content':chunk}}]})}\n\n"
            time.sleep(0.3)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
