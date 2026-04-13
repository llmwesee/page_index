from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from hashlib import sha1
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from streamlit_app.cache import LruTtlCache
from streamlit_app.config import AppConfig
from streamlit_app.rag.memory import resolve_query_with_session_memory
from streamlit_app.rag.prompts import build_answer_prompt
from streamlit_app.rag.types import (
    AnswerResult,
    PageContentResult,
    QueryTrace,
    RetrievedNode,
    TreeSearchResult,
)

try:
    import tiktoken  # type: ignore

    def _count_tokens(text: Optional[str]) -> int:
        if not text:
            return 0
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.encoding_for_model("gpt-4o-mini") if hasattr(tiktoken, "encoding_for_model") else tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

except Exception:
    import re

    def _count_tokens(text: Optional[str]) -> int:
        if not text:
            return 0
        # very rough fallback: split on whitespace
        return len(re.findall(r"\S+", text))

StageCallback = Callable[[str, Dict[str, Any]], None]

_PREPARED_DOC_CACHE: LruTtlCache[str, Dict[str, Any]] = LruTtlCache(
    name="prepared_doc_hierarchy",
    max_entries=128,
    ttl_seconds=3600,
)

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

EVIDENCE_MARKER_PATTERN = re.compile(
    r"\[EVID\s+type=([A-Za-z_]+)\s+element=([^\]]+)\]",
    flags=re.IGNORECASE,
)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _extract_evidence_marker(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = EVIDENCE_MARKER_PATTERN.search(text or "")
    if not match:
        return None, None
    content_type = str(match.group(1) or "").strip().upper() or None
    element_id = str(match.group(2) or "").strip() or None
    return content_type, element_id


def _strip_evidence_marker(text: str) -> str:
    return EVIDENCE_MARKER_PATTERN.sub("", text or "").strip()


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _tokenize(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", (value or "").lower())
        if token not in STOP_WORDS and len(token) > 1
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _node_start_page(node: Dict[str, Any]) -> Optional[int]:
    return _as_int(node.get("page_index")) or _as_int(node.get("start_index"))


def _node_summary(node: Dict[str, Any], max_chars: Optional[int] = None) -> str:
    summary = _clean_text(node.get("summary") or node.get("prefix_summary") or node.get("text") or node.get("title"))
    if max_chars and len(summary) > max_chars:
        return summary[:max_chars].rstrip() + "..."
    return summary


def flatten_tree(tree: Any) -> List[Dict[str, Any]]:
    if isinstance(tree, dict):
        nodes = [tree]
        for child in tree.get("nodes", []) or []:
            nodes.extend(flatten_tree(child))
        return nodes
    if isinstance(tree, list):
        all_nodes: List[Dict[str, Any]] = []
        for item in tree:
            all_nodes.extend(flatten_tree(item))
        return all_nodes
    return []


def build_node_map(tree: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    nodes = flatten_tree(tree)
    node_map: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("node_id")
        if node_id:
            node_map[str(node_id)] = node
    return node_map


def _split_words(words: Sequence[str], chunk_size: int, overlap: int) -> List[str]:
    if not words:
        return []
    size = max(60, chunk_size)
    step = max(1, size - max(0, overlap))
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            continue
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def _build_parent_sections(
    tree: Sequence[Dict[str, Any]],
    ocr_pages: Sequence[Dict[str, Any]],
    max_summary_chars: int,
) -> List[Dict[str, Any]]:
    page_lookup: Dict[int, str] = {
        int(item.get("page")): str(item.get("text", "")).strip()
        for item in ocr_pages
        if _as_int(item.get("page"))
    }
    total_pages = len(page_lookup) or max((_as_int(item.get("page")) or 0 for item in ocr_pages), default=0) or 1
    sections: List[Dict[str, Any]] = []

    def walk(
        nodes: Sequence[Dict[str, Any]],
        path_titles: List[str],
        path_ids: List[str],
        parent_end_page: int,
        depth: int,
    ) -> None:
        for index, node in enumerate(nodes):
            node_id = str(node.get("node_id") or f"node_{depth}_{index}")
            title = str(node.get("title") or "(untitled)").strip()
            start_page = _node_start_page(node) or 1

            next_page = None
            for sibling in nodes[index + 1 :]:
                candidate = _node_start_page(sibling)
                if candidate:
                    next_page = candidate
                    break

            end_page = parent_end_page
            if next_page:
                end_page = min(parent_end_page, max(start_page, next_page - 1))
            end_page = max(start_page, min(end_page, total_pages))

            section_pages = []
            for page_num in range(start_page, end_page + 1):
                text = page_lookup.get(page_num, "").strip()
                if text:
                    section_pages.append({"page": page_num, "text": text})

            path = path_titles + [title]
            section = {
                "node_id": node_id,
                "title": title,
                "path": " > ".join(path),
                "path_titles": path,
                "path_ids": path_ids + [node_id],
                "depth": depth,
                "start_page": start_page,
                "end_page": end_page,
                "summary": _node_summary(node, max_chars=max_summary_chars),
                "pages": section_pages,
                "is_leaf": not bool(node.get("nodes")),
            }
            if section_pages:
                sections.append(section)

            children = node.get("nodes") or []
            if children:
                walk(children, path, path_ids + [node_id], end_page, depth + 1)

    walk(tree, [], [], total_pages, 1)
    return sections


def _build_child_chunks(
    sections: Sequence[Dict[str, Any]],
    chunk_words: int,
    overlap_words: int,
) -> List[Dict[str, Any]]:
    child_chunks: List[Dict[str, Any]] = []
    for section in sections:
        local_index = 0
        for page_item in section.get("pages", []):
            page = _as_int(page_item.get("page"))
            words = re.findall(r"\S+", str(page_item.get("text", "")))
            for snippet in _split_words(words, chunk_size=chunk_words, overlap=overlap_words):
                local_index += 1
                child_chunks.append(
                    {
                        "chunk_id": f"{section['node_id']}::{page or 0}::{local_index}",
                        "parent_id": section["node_id"],
                        "parent_title": section["title"],
                        "parent_path": section["path"],
                        "depth": section["depth"],
                        "page": page,
                        "chunk_index": local_index,
                        "text": snippet,
                    }
                )
    return child_chunks


def _compute_bm25_stats(chunks: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, float], float]:
    if not chunks:
        return {}, 1.0

    doc_freq: Counter[str] = Counter()
    total_tokens = 0
    for chunk in chunks:
        tokens = _tokenize(str(chunk.get("text", "")))
        chunk["_tokens"] = tokens
        chunk["_token_counts"] = Counter(tokens)
        total_tokens += len(tokens)
        for token in set(tokens):
            doc_freq[token] += 1

    corpus_size = max(1, len(chunks))
    avg_doc_len = max(1.0, total_tokens / corpus_size)
    idf = {
        token: max(0.0, (corpus_size - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
    return idf, avg_doc_len


def _prepared_doc_cache_key(
    doc_id: str,
    tree: Sequence[Dict[str, Any]],
    ocr_pages: Sequence[Dict[str, Any]],
    config: AppConfig,
) -> str:
    tree_ids = ",".join(str(node.get("node_id", "")) for node in flatten_tree(tree)[:400])
    page_sample = ",".join(str(_as_int(item.get("page")) or 0) for item in list(ocr_pages)[:40])
    fingerprint = sha1(f"{doc_id}|{tree_ids}|{page_sample}".encode("utf-8")).hexdigest()[:16]
    return "|".join(
        [
            doc_id,
            str(config.max_tree_summary_chars),
            str(config.hierarchical_child_chunk_words),
            str(config.hierarchical_child_chunk_overlap_words),
            fingerprint,
        ]
    )


def _prepare_doc_hierarchy(
    doc_id: str,
    doc_tree: Sequence[Dict[str, Any]],
    ocr_pages: Sequence[Dict[str, Any]],
    config: AppConfig,
) -> Dict[str, Any]:
    _PREPARED_DOC_CACHE.configure(
        max_entries=config.prepared_doc_cache_size,
        ttl_seconds=config.prepared_doc_cache_ttl_sec,
    )
    cache_key = _prepared_doc_cache_key(doc_id, doc_tree, ocr_pages, config)
    cached = _PREPARED_DOC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    node_map = build_node_map(doc_tree)
    sections = _build_parent_sections(
        doc_tree,
        ocr_pages=ocr_pages,
        max_summary_chars=config.max_tree_summary_chars,
    )
    child_chunks = _build_child_chunks(
        sections,
        chunk_words=config.hierarchical_child_chunk_words,
        overlap_words=config.hierarchical_child_chunk_overlap_words,
    )
    idf, avg_doc_len = _compute_bm25_stats(child_chunks)
    prepared = {
        "node_map": node_map,
        "sections": sections,
        "child_chunks": child_chunks,
        "idf": idf,
        "avg_doc_len": avg_doc_len,
        "ocr_pages": list(ocr_pages),
    }
    _PREPARED_DOC_CACHE.set(cache_key, prepared)
    return prepared


def _score_child_chunk(
    chunk: Dict[str, Any],
    query: str,
    query_terms: Sequence[str],
    idf: Dict[str, float],
    avg_doc_len: float,
) -> float:
    token_counts: Counter[str] = chunk.get("_token_counts", Counter())
    doc_len = max(1, len(chunk.get("_tokens", [])))
    score = 0.0
    k1 = 1.5
    b = 0.75

    for term in query_terms:
        tf = token_counts.get(term, 0)
        if not tf:
            continue
        term_idf = idf.get(term, 0.0)
        denom = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
        score += term_idf * ((tf * (k1 + 1)) / max(denom, 1e-9))

    chunk_text = str(chunk.get("text", "")).lower()
    parent_title = str(chunk.get("parent_title", "")).lower()
    parent_path = str(chunk.get("parent_path", "")).lower()
    query_lower = query.lower().strip()

    if query_lower and query_lower in chunk_text:
        score += 3.0
    if query_lower and query_lower in parent_title:
        score += 1.5

    title_terms = set(_tokenize(parent_title))
    path_terms = set(_tokenize(parent_path))
    overlap_count = len(set(query_terms) & title_terms)
    score += overlap_count * 0.8
    score += len(set(query_terms) & path_terms) * 0.25
    score += max(0, int(chunk.get("depth", 1)) - 1) * 0.1
    return score


def _build_hierarchical_tree_search_result(
    query: str,
    sections: Sequence[Dict[str, Any]],
    child_chunks: Sequence[Dict[str, Any]],
    config: AppConfig,
    idf: Optional[Dict[str, float]] = None,
    avg_doc_len: Optional[float] = None,
    preferred_node_ids: Optional[Sequence[str]] = None,
) -> Tuple[TreeSearchResult, List[Dict[str, Any]], List[Dict[str, Any]]]:
    def _shares_lineage(left: Sequence[str], right: Sequence[str]) -> bool:
        shorter = min(len(left), len(right))
        return list(left[:shorter]) == list(right[:shorter])

    if not sections or not child_chunks:
        return (
            TreeSearchResult(
                thinking="Hierarchical retrieval had no indexed sections to search.",
                node_list=[],
                raw="",
            ),
            [],
            [],
        )

    query_terms = _tokenize(query)
    preferred_node_id_set = {str(node_id) for node_id in (preferred_node_ids or []) if str(node_id).strip()}
    if idf is None or avg_doc_len is None:
        idf, avg_doc_len = _compute_bm25_stats(child_chunks)
    for chunk in child_chunks:
        chunk["score"] = _score_child_chunk(chunk, query, query_terms, idf, avg_doc_len)
        if preferred_node_id_set and str(chunk.get("parent_id")) in preferred_node_id_set:
            chunk["score"] = _safe_float(chunk.get("score")) + float(config.section_rerank_score_bonus)

    ranked_chunks = sorted(child_chunks, key=lambda item: (_safe_float(item.get("score")), -int(item.get("chunk_index", 0))), reverse=True)
    positive_chunks = [chunk for chunk in ranked_chunks if _safe_float(chunk.get("score")) > 0]
    if positive_chunks:
        candidate_chunks = positive_chunks[: max(config.hierarchical_candidate_pool_size, config.hierarchical_top_child_chunks)]
    else:
        candidate_chunks = ranked_chunks[: max(config.hierarchical_top_child_chunks, config.retrieval_top_k)]

    section_by_id = {section["node_id"]: section for section in sections}
    parent_scores: Dict[str, Dict[str, Any]] = {}
    for chunk in candidate_chunks:
        parent_id = str(chunk.get("parent_id"))
        bucket = parent_scores.setdefault(
            parent_id,
            {
                "parent_id": parent_id,
                "score_sum": 0.0,
                "score_max": 0.0,
                "hit_count": 0,
                "pages": set(),
                "best_chunk": chunk,
            },
        )
        score = _safe_float(chunk.get("score"))
        bucket["score_sum"] += score
        bucket["score_max"] = max(bucket["score_max"], score)
        bucket["hit_count"] += 1
        if chunk.get("page"):
            bucket["pages"].add(int(chunk["page"]))
        if score > _safe_float(bucket["best_chunk"].get("score")):
            bucket["best_chunk"] = chunk

    ranked_parents: List[Dict[str, Any]] = []
    for parent_id, bucket in parent_scores.items():
        section = section_by_id.get(parent_id)
        if not section:
            continue
        aggregate_score = bucket["score_max"] + (0.2 * bucket["hit_count"]) + (0.1 * min(bucket["score_sum"], 8.0))
        if preferred_node_id_set and parent_id in preferred_node_id_set:
            aggregate_score += float(config.section_rerank_score_bonus)
        ranked_parents.append(
            {
                "section": section,
                "score": aggregate_score,
                "pages": sorted(bucket["pages"]),
                "best_chunk": bucket["best_chunk"],
            }
        )

    ranked_parents.sort(key=lambda item: (item["score"], item["section"].get("depth", 0)), reverse=True)
    selected_parents: List[Dict[str, Any]] = []
    for candidate in ranked_parents:
        candidate_path = list(candidate["section"].get("path_ids", []))
        replaced = False
        should_skip = False
        for index, existing in enumerate(list(selected_parents)):
            existing_path = list(existing["section"].get("path_ids", []))
            if not _shares_lineage(candidate_path, existing_path):
                continue
            candidate_depth = int(candidate["section"].get("depth", 0))
            existing_depth = int(existing["section"].get("depth", 0))
            if candidate_depth > existing_depth and candidate["score"] >= (existing["score"] * 0.7):
                selected_parents[index] = candidate
                replaced = True
            else:
                should_skip = True
            break
        if should_skip:
            continue
        if not replaced:
            selected_parents.append(candidate)
        if len(selected_parents) >= max(1, config.retrieval_top_k):
            break

    if not selected_parents:
        selected_parents = ranked_parents[: max(1, config.retrieval_top_k)]
    selected_parent_ids = {item["section"]["node_id"] for item in selected_parents}

    parent_chunk_pool = [chunk for chunk in candidate_chunks if chunk.get("parent_id") in selected_parent_ids]
    chunk_lookup = {
        (str(chunk.get("parent_id")), int(chunk.get("chunk_index", 0))): chunk
        for chunk in child_chunks
    }
    expanded_chunks: List[Dict[str, Any]] = []
    seen_chunk_ids = set()
    for chunk in sorted(parent_chunk_pool, key=lambda item: _safe_float(item.get("score")), reverse=True):
        parent_id = str(chunk.get("parent_id"))
        for offset in range(-max(0, config.hierarchical_neighbor_chunks), max(0, config.hierarchical_neighbor_chunks) + 1):
            neighbor = chunk_lookup.get((parent_id, int(chunk.get("chunk_index", 0)) + offset))
            if not neighbor:
                continue
            chunk_id = str(neighbor.get("chunk_id"))
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            expanded_chunks.append(neighbor)

    expanded_chunks.sort(key=lambda item: (_safe_float(item.get("score")), -(_as_int(item.get("page")) or 0)), reverse=True)
    selected_chunks = expanded_chunks[: max(1, config.hierarchical_top_child_chunks)]
    if not selected_chunks:
        selected_chunks = candidate_chunks[: max(1, config.hierarchical_top_child_chunks)]

    reasoning_lines: List[str] = []
    for idx, parent in enumerate(selected_parents[:3], start=1):
        section = parent["section"]
        best_chunk = parent["best_chunk"]
        pages = ", ".join(str(p) for p in parent.get("pages") or [] if p)
        reasoning_lines.append(
            f"{idx}. `{section['title']}` matched the query through child chunks in `{section['path']}`"
            f" on page(s) {pages or section['start_page']} with score {parent['score']:.2f}."
        )
        if preferred_node_id_set and str(section.get("node_id")) in preferred_node_id_set:
            reasoning_lines.append("   Vector reranking also prioritized this section from the pgvector section index.")
        reasoning_lines.append(
            f"   Best evidence snippet: \"{_clean_text(best_chunk.get('text'))[:180]}{'...' if len(_clean_text(best_chunk.get('text'))) > 180 else ''}\""
        )

    tree_search = TreeSearchResult(
        thinking=(
            "Hierarchical retrieval searched fixed-size child chunks inside PageIndex sections, then promoted the"
            " highest-scoring parent sections for answer grounding.\n"
            + "\n".join(reasoning_lines)
        ).strip(),
        node_list=[item["section"]["node_id"] for item in selected_parents],
        raw=json.dumps(
            {
                "query": query,
                "selected_parents": [
                    {
                        "node_id": item["section"]["node_id"],
                        "title": item["section"]["title"],
                        "path": item["section"]["path"],
                        "score": round(item["score"], 4),
                        "pages": item.get("pages", []),
                    }
                    for item in selected_parents
                ],
                "selected_chunks": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "parent_id": chunk.get("parent_id"),
                        "page": chunk.get("page"),
                        "score": round(_safe_float(chunk.get("score")), 4),
                    }
                    for chunk in selected_chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return tree_search, selected_parents, selected_chunks


def parse_llm_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}

    candidates: List[str] = [raw.strip()]

    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    candidates.extend(fenced)

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(raw[first : last + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.replace("\r", "")
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", normalized)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return {}


def build_retrieved_nodes(
    node_ids: Iterable[str],
    node_map: Dict[str, Dict[str, Any]],
    section_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    parent_scores: Optional[Dict[str, float]] = None,
    doc_id: Optional[str] = None,
    doc_name: Optional[str] = None,
    doc_ref: Optional[str] = None,
) -> List[RetrievedNode]:
    result: List[RetrievedNode] = []
    for node_id in node_ids:
        node = node_map.get(node_id) or (section_lookup or {}).get(node_id)
        if not node:
            continue
        page = _as_int(node.get("page_index")) or _as_int(node.get("start_page")) or _as_int(node.get("start_index"))
        result.append(
            RetrievedNode(
                node_id=node_id,
                title=str(node.get("title", "")).strip() or "(untitled)",
                page=page,
                doc_id=doc_id,
                doc_name=doc_name,
                doc_ref=doc_ref,
                section_path=str(node.get("path", "")).strip() or None,
                score=(parent_scores or {}).get(node_id),
            )
        )
    return result


def build_evidence_context(page_content: Sequence[Dict[str, Any]], max_chars_per_page: int) -> str:
    chunks: List[str] = []
    for item in page_content:
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or "unknown_doc").strip()
        page = _as_int(item.get("page"))
        content_type = str(item.get("content_type") or "TEXT").strip().upper() or "TEXT"
        element_id = str(item.get("element_id") or "unknown").strip() or "unknown"
        text = _truncate_text(item.get("text", ""), max_chars_per_page)
        if not doc_ref or not page or not text:
            continue
        chunks.append(
            f"<doc={doc_ref};page={page};type={content_type};element={element_id}>\n"
            f"{text}\n"
            f"</doc={doc_ref};page={page};type={content_type};element={element_id}>"
        )
    return "\n\n".join(chunks)


def extract_inline_citation_pages(
    answer_markdown: str,
    doc_name: Optional[str] = None,
    allowed_doc_refs: Optional[Sequence[str]] = None,
) -> List[int]:
    citations = extract_inline_citations(
        answer_markdown=answer_markdown,
        doc_name=doc_name,
        allowed_doc_refs=allowed_doc_refs,
    )
    pages: List[int] = []
    for citation in citations:
        page = _as_int(citation.get("page"))
        if page is not None and page not in pages:
            pages.append(page)
    return pages


def extract_inline_citations(
    answer_markdown: str,
    doc_name: Optional[str] = None,
    allowed_doc_refs: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"<doc=([^;>]+);page=(\d+)(?:;type=([^;>]+))?(?:;element=([^;>]+))?>",
        flags=re.IGNORECASE,
    )
    allowed = {value for value in (allowed_doc_refs or []) if value}
    citations: List[Dict[str, Any]] = []
    for match in pattern.findall(answer_markdown or ""):
        found_doc, page_raw, content_type_raw, element_raw = match
        if doc_name and found_doc != doc_name:
            continue
        if allowed and found_doc not in allowed:
            continue
        page = _as_int(page_raw)
        if not page:
            continue
        content_type = (content_type_raw or "").strip().upper() or None
        element_id = (element_raw or "").strip() or None
        citation = {
            "doc_ref": found_doc,
            "page": page,
            "content_type": content_type,
            "element_id": element_id,
            "citation_id": f"{found_doc}#p{page}:{(content_type or 'TEXT').lower()}:{element_id or 'unknown'}",
        }
        if citation not in citations:
            citations.append(citation)
    return citations


def _normalize_cited_pages(
    cited_pages: Sequence[Any],
    available_pages: Sequence[int],
) -> List[int]:
    available_set = {p for p in available_pages if p is not None}
    normalized: List[int] = []
    for value in cited_pages:
        page = _as_int(value)
        if page is None:
            continue
        if available_set and page not in available_set:
            continue
        if page not in normalized:
            normalized.append(page)
    return normalized


def _normalize_cited_sources(
    citations: Sequence[Dict[str, Any]],
    page_content: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    allowed_content_types = {"TEXT", "TABLE", "IMAGE", "FORMULA"}

    def _infer_content_type_from_element_id(raw_element_id: Any) -> str:
        element = str(raw_element_id or "").strip().lower()
        if not element:
            return ""
        table_tokens = ("table", "tbl", "tabular", "grid")
        image_tokens = ("image", "img", "figure", "fig", "map", "photo", "diagram", "chart")
        formula_tokens = ("formula", "equation", "eq", "math", "latex")
        if any(token in element for token in table_tokens):
            return "TABLE"
        if any(token in element for token in image_tokens):
            return "IMAGE"
        if any(token in element for token in formula_tokens):
            return "FORMULA"
        return ""

    by_page: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    by_element: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for item in page_content:
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or "").strip()
        page = _as_int(item.get("page"))
        element_id = str(item.get("element_id") or "").strip()
        if not doc_ref or not page:
            continue
        by_page.setdefault((doc_ref, page), []).append(item)
        if element_id:
            by_element[(doc_ref, page, element_id)] = item

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for citation in citations:
        doc_ref = str(citation.get("doc_ref") or "").strip()
        page = _as_int(citation.get("page"))
        element_id = str(citation.get("element_id") or "").strip()
        citation_content_type = str(citation.get("content_type") or "").strip().upper()
        if citation_content_type not in allowed_content_types:
            citation_content_type = ""
        inferred_citation_type = _infer_content_type_from_element_id(element_id)
        if not citation_content_type and inferred_citation_type:
            citation_content_type = inferred_citation_type
        elif citation_content_type == "TEXT" and inferred_citation_type in {"TABLE", "IMAGE", "FORMULA"}:
            # Prefer structural modality over a generic TEXT fallback.
            citation_content_type = inferred_citation_type
        if not doc_ref or not page:
            continue

        key = (doc_ref, page, element_id or "")
        if key in seen:
            continue

        matched_item = None
        exact_element_match = False
        if element_id:
            matched_item = by_element.get((doc_ref, page, element_id))
            exact_element_match = matched_item is not None
        if matched_item is None:
            candidates = by_page.get((doc_ref, page), [])
            matched_item = next(
                (
                    item
                    for item in candidates
                    if _clean_text(item.get("text", ""))
                ),
                None,
            )
        if matched_item is None:
            continue

        seen.add(key)
        matched_content_type = str(matched_item.get("content_type") or "").strip().upper()
        if matched_content_type not in allowed_content_types:
            matched_content_type = ""
        matched_element_id = str(matched_item.get("element_id") or "").strip()
        inferred_matched_type = _infer_content_type_from_element_id(matched_element_id)
        if not matched_content_type and inferred_matched_type:
            matched_content_type = inferred_matched_type

        # Preserve explicit modality from inline citations when we only have a page-level fallback.
        if exact_element_match:
            source_content_type = matched_content_type or citation_content_type or "TEXT"
        else:
            source_content_type = citation_content_type or matched_content_type or "TEXT"

        source_element_id = str(element_id or matched_element_id or "unknown").strip() or "unknown"
        normalized.append(
            {
                "doc_ref": doc_ref,
                "page": page,
                "content_type": source_content_type,
                "element_id": source_element_id,
                "citation_id": f"{doc_ref}#p{page}:{source_content_type.lower()}:{source_element_id}",
            }
        )
    return normalized


def _fallback_cited_sources_from_evidence(
    page_content: Sequence[Dict[str, Any]],
    *,
    fallback_doc_ref: Optional[str] = None,
    max_sources: int = 2,
) -> List[Dict[str, Any]]:
    ranked_items = sorted(
        [item for item in page_content if _clean_text(item.get("text", ""))],
        key=lambda item: _safe_float(item.get("score"), 0.0),
        reverse=True,
    )

    fallback_sources: List[Dict[str, Any]] = []
    seen = set()
    fallback_doc_ref = str(fallback_doc_ref or "").strip()
    for item in ranked_items:
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or fallback_doc_ref).strip()
        page = _as_int(item.get("page"))
        content_type = str(item.get("content_type") or "TEXT").strip().upper() or "TEXT"
        element_id = str(item.get("element_id") or "unknown").strip() or "unknown"
        if not doc_ref or not page:
            continue
        if content_type == "IMAGE":
            # Images are supporting-only and cannot serve as sole factual grounding.
            continue

        key = (doc_ref, page, content_type, element_id)
        if key in seen:
            continue
        seen.add(key)
        fallback_sources.append(
            {
                "doc_ref": doc_ref,
                "page": page,
                "content_type": content_type,
                "element_id": element_id,
                "citation_id": f"{doc_ref}#p{page}:{content_type.lower()}:{element_id}",
            }
        )
        if len(fallback_sources) >= max(1, max_sources):
            break

    return fallback_sources


def _build_sources_markdown(cited_sources: Sequence[Dict[str, Any]]) -> str:
    if not cited_sources:
        return ""

    lines = ["### Sources"]
    seen = set()
    for item in cited_sources:
        doc_ref = str(item.get("doc_ref") or "").strip()
        page = _as_int(item.get("page"))
        content_type = str(item.get("content_type") or "TEXT").strip().upper() or "TEXT"
        element_id = str(item.get("element_id") or "unknown").strip() or "unknown"
        citation_id = str(item.get("citation_id") or "").strip()
        if not doc_ref or not page:
            continue
        dedupe_key = (doc_ref, page, content_type, element_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        citation_suffix = f" [{citation_id}]" if citation_id else ""
        lines.append(
            f"- `{doc_ref}`, page {page}, {content_type} `{element_id}`{citation_suffix}"
        )
    return "\n".join(lines)


def _normalize_answer_markdown(answer_markdown: str, cited_sources: Sequence[Dict[str, Any]]) -> str:
    answer = (answer_markdown or "").strip()
    if not answer:
        return answer

    answer = re.sub(r"\n#{1,6}\s*Sources\b[\s\S]*$", "", answer, flags=re.IGNORECASE).strip()
    if not re.match(r"^\s{0,3}##\s+Final Answer\b", answer, flags=re.IGNORECASE):
        answer = f"## Final Answer\n\n{answer}"

    sources_block = _build_sources_markdown(cited_sources)
    if sources_block:
        answer = f"{answer}\n\n{sources_block}"
    return answer.strip()


def _infer_answer_meta(answer_markdown: str, cited_sources: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    answer_lower = (answer_markdown or "").lower()
    if not cited_sources or "insufficient evidence" in answer_lower:
        return "LOW", "INSUFFICIENT"

    non_image_sources = [
        item for item in cited_sources
        if str(item.get("content_type") or "").strip().upper() != "IMAGE"
    ]
    if not non_image_sources:
        # Images are supporting-only evidence, not primary factual grounding.
        return "LOW", "INSUFFICIENT"

    partial_markers = [
        "not explicit",
        "not expressly",
        "does not specify",
        "does not explicitly",
        "partially",
        "however",
    ]
    if any(marker in answer_lower for marker in partial_markers):
        return "MEDIUM", "PARTIAL"
    if len(cited_sources) >= 2:
        return "HIGH", "DIRECT"
    return "MEDIUM", "DIRECT"


def _build_local_answer_thinking(
    query: str,
    answer_markdown: str,
    page_content: Sequence[Dict[str, Any]],
    cited_sources: Sequence[Dict[str, Any]],
) -> str:
    cited_set = {
        (
            str(item.get("doc_ref") or "").strip(),
            _as_int(item.get("page")),
            str(item.get("element_id") or "").strip(),
        )
        for item in cited_sources
        if str(item.get("doc_ref") or "").strip() and _as_int(item.get("page"))
    }
    matched = [
        item for item in page_content
        if (
            str(item.get("doc_ref") or item.get("doc_name") or "").strip(),
            _as_int(item.get("page")),
            str(item.get("element_id") or "").strip(),
        ) in cited_set
    ]
    if not matched:
        matched = list(page_content)[: min(3, len(page_content))]

    bullets = [f"- Query focus: {_clean_text(query)[:180]}"]
    seen = set()
    for item in matched[:3]:
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or "unknown_doc").strip()
        page = _as_int(item.get("page"))
        title = _clean_text(item.get("title")) or "Relevant section"
        snippet = _truncate_text(item.get("text", ""), 180)
        content_type = str(item.get("content_type") or "TEXT").strip().upper() or "TEXT"
        element_id = str(item.get("element_id") or "unknown").strip() or "unknown"
        key = (doc_ref, page, element_id)
        if key in seen or not page:
            continue
        seen.add(key)
        bullets.append(
            f"- `{title}` on <doc={doc_ref};page={page};type={content_type};element={element_id}> supports the answer with: \"{snippet}\""
        )

    confidence, answer_type = _infer_answer_meta(answer_markdown, cited_sources)
    bullets.append(f"- Overall assessment: {answer_type} answer with {confidence} confidence based on cited page evidence.")
    return "\n".join(bullets)


def parse_answer_result(
    raw: str,
    doc_name: str,
    available_pages: Sequence[int],
    page_content: Optional[Sequence[Dict[str, Any]]] = None,
    query: Optional[str] = None,
    allowed_doc_refs: Optional[Sequence[str]] = None,
    fallback_doc_ref: Optional[str] = None,
) -> AnswerResult:
    parsed = parse_llm_json(raw)
    thinking = str(parsed.get("thinking", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    confidence = str(parsed.get("confidence", "")).strip().upper()
    answer_type = str(parsed.get("answer_type", "")).strip().upper()

    if not answer:
        answer = raw.strip()
    if not answer:
        answer = "Insufficient evidence to answer reliably from retrieved pages."

    explicit_citation_sources = extract_inline_citations(
        answer,
        doc_name=doc_name if not allowed_doc_refs else None,
        allowed_doc_refs=allowed_doc_refs,
    )
    explicit_citations = extract_inline_citation_pages(
        answer,
        doc_name=doc_name if not allowed_doc_refs else None,
        allowed_doc_refs=allowed_doc_refs,
    )
    cited_pages_raw = parsed.get("cited_pages", [])
    if not isinstance(cited_pages_raw, list):
        cited_pages_raw = []

    cited_pages = _normalize_cited_pages(
        list(cited_pages_raw) + explicit_citations,
        [p for p in available_pages if p is not None],
    )

    cited_sources = _normalize_cited_sources(
        explicit_citation_sources,
        page_content or [],
    )

    if not cited_sources:
        fallback_sources = _fallback_cited_sources_from_evidence(
            page_content or [],
            fallback_doc_ref=fallback_doc_ref,
        )
        if fallback_sources:
            cited_sources = fallback_sources
            cited_pages = sorted({_as_int(item.get("page")) for item in cited_sources if _as_int(item.get("page"))})
            citation_tokens = " ".join(
                [
                    (
                        f"<doc={item['doc_ref']};page={item['page']};"
                        f"type={item['content_type']};element={item['element_id']}>"
                    )
                    for item in cited_sources
                ]
            )
            if citation_tokens:
                answer = answer.rstrip() + (" " if answer else "") + citation_tokens
        else:
            answer = (
                "Insufficient evidence: the response did not include verifiable citations tied to retrieved evidence "
                "units (page + element)."
            )
            cited_pages = []
            confidence = "LOW"
            answer_type = "INSUFFICIENT"
    elif all(str(item.get("content_type") or "").strip().upper() == "IMAGE" for item in cited_sources):
        answer = (
            "Insufficient evidence: only image-description evidence was cited. "
            "Primary factual claims require text/table evidence citations."
        )
        cited_pages = sorted({_as_int(item.get("page")) for item in cited_sources if _as_int(item.get("page"))})
        confidence = "LOW"
        answer_type = "INSUFFICIENT"

    answer = _normalize_answer_markdown(answer, cited_sources)

    if not thinking:
        thinking = _build_local_answer_thinking(
            query=query or "",
            answer_markdown=answer,
            page_content=page_content or [],
            cited_sources=cited_sources,
        )

    if confidence not in {"HIGH", "MEDIUM", "LOW"} or answer_type not in {"DIRECT", "PARTIAL", "INSUFFICIENT"}:
        confidence, answer_type = _infer_answer_meta(answer, cited_sources)

    return AnswerResult(
        thinking=thinking,
        answer_markdown=answer,
        cited_pages=cited_pages,
        cited_sources=cited_sources,
        confidence=confidence,
        answer_type=answer_type,
        raw=raw,
    )


class RAGPipeline:
    def __init__(
        self,
        api_client: Any,
        llm_client: Any,
        config: AppConfig,
        logger: Optional[Any] = None,
        observer: Optional[Any] = None,
    ) -> None:
        self.api_client = api_client
        self.llm_client = llm_client
        self.config = config
        self.logger = logger
        self.observer = observer

    def run_query(
        self,
        query: str,
        doc_id: str,
        doc_name: str,
        tree: Sequence[Dict[str, Any]],
        langfuse_session_id: Optional[str] = None,
        stage_callback: Optional[StageCallback] = None,
        parent_trace_handle: Optional[Any] = None,
        route_info: Optional[Dict[str, Any]] = None,
    ) -> QueryTrace:
        return self.run_query_multi(
            query=query,
            documents=[{"doc_id": doc_id, "doc_name": doc_name, "tree": list(tree)}],
            langfuse_session_id=langfuse_session_id,
            stage_callback=stage_callback,
            parent_trace_handle=parent_trace_handle,
            route_info=route_info,
        )

    def _doc_ref(self, doc_id: str, doc_name: str) -> str:
        safe_name = re.sub(r"[<>;]+", "_", str(doc_name or "unknown.pdf")).strip() or "unknown.pdf"
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "", str(doc_id or ""))[:8]
        return f"{safe_name}#{safe_id}" if safe_id else safe_name

    def run_query_multi(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        session_memory: Optional[Dict[str, Any]] = None,
        langfuse_session_id: Optional[str] = None,
        stage_callback: Optional[StageCallback] = None,
        parent_trace_handle: Optional[Any] = None,
        route_info: Optional[Dict[str, Any]] = None,
    ) -> QueryTrace:
        if not documents:
            raise ValueError("At least one document is required")

        primary_doc_id = str(documents[0].get("doc_id") or documents[0].get("id") or "")
        primary_doc_name = str(documents[0].get("doc_name") or documents[0].get("name") or "unknown.pdf")
        trace = QueryTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            doc_id=primary_doc_id,
            doc_name=primary_doc_name,
            selected_doc_ids=[
                str(doc.get("doc_id") or doc.get("id") or "") for doc in documents if doc.get("doc_id") or doc.get("id")
            ],
            selected_doc_names=[str(doc.get("doc_name") or doc.get("name") or "unknown.pdf") for doc in documents],
        )
        trace.routing = dict(route_info or {})
        total_start = time.perf_counter()

        # Token usage summary (counts are approximate when tiktoken not available)
        token_summary = {
            # informational
            "user_query_tokens": _count_tokens(query),
            "metadata_tokens": 0,  # tokens in compact tree JSON (informational)
            "page_fetch_tokens": 0,  # tokens in evidence context (informational)
            "session_memory_tokens": 0,
            # per-LLM-call breakdown
            "tree_search_input_tokens": 0,
            "tree_search_output_tokens": 0,
            "answer_input_tokens": 0,
            "answer_output_tokens": 0,
            "answer_instruction_tokens": 0,
            "answer_retrieval_context_tokens": 0,
            "answer_query_tokens": 0,
            "answer_allowed_refs_tokens": 0,
            "answer_prompt_tokens_provider": 0,
            "answer_completion_tokens_provider": 0,
            "answer_total_tokens_provider": 0,
            "answer_cached_tokens_provider": 0,
            "answer_reasoning_tokens_provider": 0,
            # totals for cost calculation (tokens actually sent/received to LLMs)
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

        def emit(stage: str, payload: Dict[str, Any]) -> None:
            if stage_callback:
                stage_callback(stage, payload)

        observer_trace = None

        def emit_observer_event(name: str, payload: Dict[str, Any]) -> None:
            if self.observer is not None:
                self.observer.event(observer_trace, name, payload)

        try:
            if self.observer is not None:
                inherited_trace_id = getattr(parent_trace_handle, "trace_id", None)
                inherited_parent_id = getattr(parent_trace_handle, "id", None)
                observer_trace = self.observer.start_trace(
                    "pageindex_rag_query",
                    trace_id=inherited_trace_id or trace.trace_id,
                    parent_observation_id=inherited_parent_id,
                    session_id=langfuse_session_id,
                    input_payload={
                        "query": query,
                        "selected_doc_ids": trace.selected_doc_ids,
                        "selected_doc_names": trace.selected_doc_names,
                    },
                    metadata={
                        "doc_count": len(documents),
                        "routing_strategy": trace.routing.get("strategy"),
                        "routed_count": trace.routing.get("routed_count"),
                    },
                )
            emit_observer_event(
                "query_execution_started",
                {
                    "query_chars": len(query or ""),
                    "doc_count": len(documents),
                    "selected_doc_ids": trace.selected_doc_ids,
                    "selected_doc_names": trace.selected_doc_names,
                    "routing": trace.routing,
                },
            )
            trace.timings["metadata_sec"] = 0.0
            trace.timings["tree_search_sec"] = 0.0
            trace.timings["page_fetch_sec"] = 0.0

            memory_resolution = resolve_query_with_session_memory(query, session_memory)
            retrieval_query = memory_resolution["resolved_query"]
            answer_context = memory_resolution["answer_context"]
            token_summary["session_memory_tokens"] = _count_tokens(answer_context)
            trace.memory = {
                "used": bool(memory_resolution.get("used")),
                "reason": memory_resolution.get("reason", "none"),
                "context_chars": int(memory_resolution.get("context_chars", 0) or 0),
                "turn_count": int((session_memory or {}).get("turn_count", 0) or 0),
                "selected_doc_names": list((session_memory or {}).get("selected_doc_names", []) or []),
            }
            emit_observer_event(
                "query_memory_resolution",
                {
                    "memory_used": bool(memory_resolution.get("used")),
                    "reason": memory_resolution.get("reason", "none"),
                    "resolved_query_chars": len(retrieval_query or ""),
                    "context_chars": int(memory_resolution.get("context_chars", 0) or 0),
                },
            )

            collected_tree_thinking: List[str] = []
            collected_tree_node_ids: List[str] = []
            tree_raw_parts: List[str] = []

            all_page_content: List[Dict[str, Any]] = []
            requested_by_doc: Dict[str, str] = {}
            returned_by_doc: Dict[str, str] = {}
            selected_parents_by_doc: Dict[str, List[str]] = {}
            selected_chunks_by_doc: Dict[str, List[str]] = {}
            doc_refs: List[str] = []

            for doc in documents:
                doc_id = str(doc.get("doc_id") or doc.get("id") or "")
                doc_name = str(doc.get("doc_name") or doc.get("name") or "unknown.pdf")
                doc_tree = doc.get("tree", []) or []
                preferred_node_ids = list(doc.get("preferred_node_ids") or [])
                doc_ref = self._doc_ref(doc_id=doc_id, doc_name=doc_name)
                if doc_ref not in doc_refs:
                    doc_refs.append(doc_ref)
                emit_observer_event(
                    "retrieval_document_started",
                    {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "doc_ref": doc_ref,
                        "preferred_node_ids": preferred_node_ids,
                    },
                )

                metadata_start = time.perf_counter()
                metadata = self.api_client.get_document_metadata(doc_id)
                trace.timings["metadata_sec"] += time.perf_counter() - metadata_start
                ocr_pages = metadata.get("ocr_pages", []) or []
                evidence_units = metadata.get("evidence_units", []) or []
                if not ocr_pages:
                    total_pages = _as_int(metadata.get("pageNum")) or 1
                    ocr_pages = [{"page": page_num, "text": ""} for page_num in range(1, total_pages + 1)]

                prepared = _prepare_doc_hierarchy(
                    doc_id=doc_id,
                    doc_tree=doc_tree,
                    ocr_pages=ocr_pages,
                    config=self.config,
                )
                node_map = prepared["node_map"]
                sections = prepared["sections"]
                child_chunks = prepared["child_chunks"]
                retrieval_outline = json.dumps(
                    [
                        {
                            "node_id": section.get("node_id"),
                            "title": section.get("title"),
                            "path": section.get("path"),
                            "start_page": section.get("start_page"),
                            "end_page": section.get("end_page"),
                            "summary": section.get("summary"),
                        }
                        for section in sections[: self.config.max_tree_nodes_for_prompt]
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                token_summary["metadata_tokens"] += _count_tokens(retrieval_outline)
                tree_search_start = time.perf_counter()
                tree_search, selected_parents, selected_chunks = _build_hierarchical_tree_search_result(
                    query=retrieval_query,
                    sections=sections,
                    child_chunks=child_chunks,
                    config=self.config,
                    idf=prepared.get("idf"),
                    avg_doc_len=prepared.get("avg_doc_len"),
                    preferred_node_ids=preferred_node_ids,
                )
                trace.timings["tree_search_sec"] += time.perf_counter() - tree_search_start
                collected_tree_thinking.append(f"[{doc_ref}] {tree_search.thinking}")
                collected_tree_node_ids.extend([f"{doc_ref}:{node_id}" for node_id in tree_search.node_list])
                tree_raw_parts.append(tree_search.raw)
                emit(
                    "tree_search",
                    {"tree_search": tree_search, "doc_id": doc_id, "doc_name": doc_name, "doc_ref": doc_ref},
                )
                emit_observer_event(
                    "retrieval_toc_parent_nodes_selected",
                    {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "doc_ref": doc_ref,
                        "selected_parent_count": len(selected_parents),
                        "selected_parent_nodes": [
                            {
                                "node_id": str(item.get("section", {}).get("node_id") or ""),
                                "title": str(item.get("section", {}).get("title") or ""),
                                "path": str(item.get("section", {}).get("path") or ""),
                                "start_page": _as_int(item.get("section", {}).get("start_page")),
                                "end_page": _as_int(item.get("section", {}).get("end_page")),
                                "score": round(_safe_float(item.get("score")), 4),
                                "pages": [int(p) for p in (item.get("pages") or []) if _as_int(p)],
                            }
                            for item in selected_parents
                        ],
                    },
                )

                section_lookup = {section["node_id"]: section for section in sections}
                parent_scores = {
                    item["section"]["node_id"]: _safe_float(item.get("score"))
                    for item in selected_parents
                }
                retrieved_for_doc = build_retrieved_nodes(
                    tree_search.node_list,
                    node_map,
                    section_lookup=section_lookup,
                    parent_scores=parent_scores,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_ref=doc_ref,
                )
                trace.retrieved_nodes.extend(retrieved_for_doc)
                emit(
                    "retrieved_nodes",
                    {"retrieved_nodes": trace.retrieved_nodes, "doc_id": doc_id, "doc_name": doc_name, "doc_ref": doc_ref},
                )
                emit_observer_event(
                    "retrieval_child_chunks_selected",
                    {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "doc_ref": doc_ref,
                        "selected_chunk_count": len(selected_chunks),
                        "selected_child_chunks": [
                            {
                                "chunk_id": str(chunk.get("chunk_id") or ""),
                                "parent_id": str(chunk.get("parent_id") or ""),
                                "parent_title": str(chunk.get("parent_title") or ""),
                                "page": _as_int(chunk.get("page")),
                                "chunk_index": int(chunk.get("chunk_index") or 0),
                                "score": round(_safe_float(chunk.get("score")), 4),
                                "text_preview": _truncate_text(chunk.get("text", ""), 180),
                            }
                            for chunk in selected_chunks
                        ],
                    },
                )

                pages_start = time.perf_counter()
                snippet_items: List[Dict[str, Any]] = []
                seen_snippets = set()
                for chunk in selected_chunks:
                    page = _as_int(chunk.get("page"))
                    raw_text = str(chunk.get("text", ""))
                    marker_content_type, marker_element_id = _extract_evidence_marker(raw_text)
                    text = _clean_text(_strip_evidence_marker(raw_text))
                    if not page or not text:
                        continue
                    content_type = marker_content_type or "TEXT"
                    element_id = marker_element_id or f"chunk:{chunk.get('chunk_id') or page}"
                    key = (page, text, content_type, element_id)
                    if key in seen_snippets:
                        continue
                    seen_snippets.add(key)
                    citation_id = f"{doc_ref}#p{page}:{content_type.lower()}:{element_id}"
                    snippet_items.append(
                        {
                            "page": page,
                            "text": text,
                            "content_type": content_type,
                            "element_id": element_id,
                            "citation_id": citation_id,
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "doc_ref": doc_ref,
                            "node_id": chunk.get("parent_id"),
                            "title": chunk.get("parent_title"),
                            "path": chunk.get("parent_path"),
                            "chunk_id": chunk.get("chunk_id"),
                            "score": round(_safe_float(chunk.get("score")), 4),
                        }
                    )
                trace.timings["page_fetch_sec"] += time.perf_counter() - pages_start

                if not snippet_items:
                    fallback_units = [
                        unit for unit in evidence_units
                        if _as_int(unit.get("page")) is not None and _clean_text(unit.get("text", ""))
                    ]
                    for unit in fallback_units[: max(1, self.config.hierarchical_top_child_chunks)]:
                        unit_page = _as_int(unit.get("page"))
                        if not unit_page:
                            continue
                        unit_text = _clean_text(unit.get("text", ""))
                        if not unit_text:
                            continue
                        content_type = str(unit.get("content_type") or "TEXT").strip().upper() or "TEXT"
                        element_id = str(unit.get("element_id") or f"page:{unit_page}:unit").strip()
                        key = (unit_page, unit_text, content_type, element_id)
                        if key in seen_snippets:
                            continue
                        seen_snippets.add(key)
                        snippet_items.append(
                            {
                                "page": unit_page,
                                "text": unit_text,
                                "content_type": content_type,
                                "element_id": element_id,
                                "citation_id": f"{doc_ref}#p{unit_page}:{content_type.lower()}:{element_id}",
                                "doc_id": doc_id,
                                "doc_name": doc_name,
                                "doc_ref": doc_ref,
                            }
                        )

                if not snippet_items:
                    fallback_pages = sorted(
                        {
                            _as_int(section.get("start_page"))
                            for section in sections[: max(1, self.config.retrieval_top_k)]
                            if _as_int(section.get("start_page"))
                        }
                    )
                    for page in fallback_pages:
                        page_text = next(
                            (
                                _clean_text(item.get("text"))
                                for item in prepared.get("ocr_pages", ocr_pages)
                                if _as_int(item.get("page")) == page and _clean_text(item.get("text"))
                            ),
                            "",
                        )
                        if page_text:
                            element_id = f"page:{page}:fallback"
                            snippet_items.append(
                                {
                                    "page": page,
                                    "text": page_text,
                                    "content_type": "TEXT",
                                    "element_id": element_id,
                                    "citation_id": f"{doc_ref}#p{page}:text:{element_id}",
                                    "doc_id": doc_id,
                                    "doc_name": doc_name,
                                    "doc_ref": doc_ref,
                                }
                            )

                selected_pages = sorted({_as_int(item.get("page")) for item in snippet_items if _as_int(item.get("page"))})
                pages = ",".join(str(page) for page in selected_pages) or "n/a"
                requested_pages = pages
                returned_pages = pages
                requested_by_doc[doc_ref] = requested_pages
                returned_by_doc[doc_ref] = returned_pages
                selected_parents_by_doc[doc_ref] = [
                    {
                        "node_id": str(item.get("section", {}).get("node_id") or ""),
                        "title": str(item.get("section", {}).get("title") or ""),
                        "path": str(item.get("section", {}).get("path") or ""),
                        "start_page": _as_int(item.get("section", {}).get("start_page")),
                        "end_page": _as_int(item.get("section", {}).get("end_page")),
                        "score": round(_safe_float(item.get("score")), 4),
                        "pages": [int(p) for p in (item.get("pages") or []) if _as_int(p)],
                    }
                    for item in selected_parents
                ]
                selected_chunks_by_doc[doc_ref] = [str(chunk.get("chunk_id")) for chunk in selected_chunks]
                emit_observer_event(
                    "retrieval_evidence_snippets_selected",
                    {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "doc_ref": doc_ref,
                        "selected_pages": selected_pages,
                        "snippet_count": len(snippet_items),
                        "snippets": [
                            {
                                "page": _as_int(item.get("page")),
                                "content_type": str(item.get("content_type") or "TEXT").strip().upper() or "TEXT",
                                "element_id": str(item.get("element_id") or ""),
                                "node_id": str(item.get("node_id") or ""),
                                "chunk_id": str(item.get("chunk_id") or ""),
                                "score": round(_safe_float(item.get("score")), 4),
                                "text_preview": _truncate_text(item.get("text", ""), 180),
                            }
                            for item in snippet_items
                        ],
                    },
                )

                for item in snippet_items:
                    all_page_content.append(dict(item))

                trace.page_result = PageContentResult(
                    requested_pages="; ".join(f"{ref}:{val}" for ref, val in requested_by_doc.items()),
                    returned_pages="; ".join(f"{ref}:{val}" for ref, val in returned_by_doc.items()),
                    content=all_page_content,
                    raw={
                        "success": True,
                        "doc_count": len(doc_refs),
                        "retrieval_mode": "hierarchical_parent_child",
                        "requested_pages_by_doc": requested_by_doc,
                        "returned_pages_by_doc": returned_by_doc,
                        "selected_parent_nodes": selected_parents_by_doc,
                        "selected_child_chunks": selected_chunks_by_doc,
                        "content": all_page_content,
                    },
                )
                emit(
                    "page_content",
                    {
                        "params": {"pages": pages, "doc_name": doc_name, "doc_ref": doc_ref, "doc_id": doc_id},
                        "page_result": trace.page_result,
                    },
                )

            trace.tree_search = TreeSearchResult(
                thinking="\n".join(collected_tree_thinking).strip(),
                node_list=collected_tree_node_ids,
                raw="\n\n".join(tree_raw_parts).strip(),
            )
            emit_observer_event(
                "retrieval_completed",
                {
                    "doc_refs": doc_refs,
                    "selected_parent_nodes_by_doc": selected_parents_by_doc,
                    "selected_child_chunks_by_doc": selected_chunks_by_doc,
                    "evidence_snippet_count": len(all_page_content),
                    "retrieval_mode": "hierarchical_parent_child",
                },
            )

            evidence_context = build_evidence_context(
                page_content=trace.page_result.content if trace.page_result else [],
                max_chars_per_page=self.config.max_page_evidence_chars,
            )
            # page fetch tokens: tokens for the evidence context provided to the answer prompt (informational)
            token_summary["page_fetch_tokens"] = _count_tokens(evidence_context)
            answer_prompt = build_answer_prompt(
                query=query,
                evidence_context=evidence_context,
                doc_refs=doc_refs,
                conversation_context=answer_context,
            )
            emit("answer_start", {})
            emit_observer_event(
                "answer_generation_started",
                {
                    "doc_refs": doc_refs,
                    "query_chars": len(query or ""),
                    "retrieval_context_chars": len(evidence_context or ""),
                },
            )

            answer_start = time.perf_counter()
            # count input tokens for the answer prompt
            answer_prompt_tokens = _count_tokens(answer_prompt)
            answer_query_tokens = _count_tokens(query)
            answer_retrieval_context_tokens = _count_tokens(evidence_context)
            answer_session_memory_tokens = _count_tokens(answer_context)
            answer_allowed_refs_tokens = _count_tokens(", ".join(doc_refs) if doc_refs else "")
            answer_instruction_tokens = max(
                0,
                answer_prompt_tokens
                - (
                    answer_query_tokens
                    + answer_retrieval_context_tokens
                    + answer_session_memory_tokens
                    + answer_allowed_refs_tokens
                ),
            )

            token_summary["answer_input_tokens"] += answer_prompt_tokens
            token_summary["answer_query_tokens"] = answer_query_tokens
            token_summary["answer_retrieval_context_tokens"] = answer_retrieval_context_tokens
            token_summary["answer_allowed_refs_tokens"] = answer_allowed_refs_tokens
            token_summary["answer_instruction_tokens"] = answer_instruction_tokens
            streamed_parts: List[str] = []
            last_stream_emit = 0.0
            last_emitted_length = 0

            def on_answer_text(text: str) -> None:
                nonlocal last_stream_emit, last_emitted_length
                streamed_parts.append(text)
                current_text = "".join(streamed_parts)
                now = time.perf_counter()
                if (now - last_stream_emit) >= 0.2 or (len(current_text) - last_emitted_length) >= 240:
                    emit("answer_stream", {"text": current_text})
                    last_stream_emit = now
                    last_emitted_length = len(current_text)

            answer_raw = ""
            provider_usage: Dict[str, Any] = {}
            provider_model = str(getattr(self.llm_client, "deployment_name", "unknown_model"))
            if hasattr(self.llm_client, "stream_complete_with_usage"):
                answer_response = self.llm_client.stream_complete_with_usage(
                    answer_prompt,
                    temperature=0.0,
                    max_tokens=self.config.answer_max_tokens,
                    on_text=on_answer_text,
                )
                answer_raw = str(answer_response.get("text", "")).strip()
                provider_usage = dict(answer_response.get("usage") or {})
                provider_model = str(answer_response.get("model") or provider_model)
            elif hasattr(self.llm_client, "stream_complete"):
                answer_raw = self.llm_client.stream_complete(
                    answer_prompt,
                    temperature=0.0,
                    max_tokens=self.config.answer_max_tokens,
                    on_text=on_answer_text,
                )
            elif hasattr(self.llm_client, "complete_with_usage"):
                answer_response = self.llm_client.complete_with_usage(
                    answer_prompt,
                    temperature=0.0,
                    max_tokens=self.config.answer_max_tokens,
                )
                answer_raw = str(answer_response.get("text", "")).strip()
                provider_usage = dict(answer_response.get("usage") or {})
                provider_model = str(answer_response.get("model") or provider_model)
            else:
                answer_raw = self.llm_client.complete(
                    answer_prompt,
                    temperature=0.0,
                    max_tokens=self.config.answer_max_tokens,
                )
            trace.timings["answer_sec"] = time.perf_counter() - answer_start
            # count output tokens for the answer
            token_summary["answer_output_tokens"] += _count_tokens(answer_raw)

            provider_prompt_tokens = _as_int(provider_usage.get("prompt_tokens")) or 0
            provider_completion_tokens = _as_int(provider_usage.get("completion_tokens")) or 0
            provider_total_tokens = _as_int(provider_usage.get("total_tokens")) or 0
            provider_cached_tokens = _as_int(provider_usage.get("cached_tokens")) or 0
            provider_reasoning_tokens = _as_int(provider_usage.get("reasoning_tokens")) or 0

            if provider_prompt_tokens > 0:
                token_summary["answer_input_tokens"] = provider_prompt_tokens
            if provider_completion_tokens > 0:
                token_summary["answer_output_tokens"] = provider_completion_tokens

            token_summary["answer_prompt_tokens_provider"] = provider_prompt_tokens
            token_summary["answer_completion_tokens_provider"] = provider_completion_tokens
            token_summary["answer_total_tokens_provider"] = provider_total_tokens
            token_summary["answer_cached_tokens_provider"] = provider_cached_tokens
            token_summary["answer_reasoning_tokens_provider"] = provider_reasoning_tokens

            answer_token_breakdown = {
                "llm_call": "answer_generation",
                "query_tokens": answer_query_tokens,
                "retrieval_context_tokens": answer_retrieval_context_tokens,
                "session_memory_tokens": answer_session_memory_tokens,
                "allowed_refs_tokens": answer_allowed_refs_tokens,
                "instruction_tokens": answer_instruction_tokens,
                "prompt_tokens": int(token_summary.get("answer_input_tokens", 0)),
                "completion_tokens": int(token_summary.get("answer_output_tokens", 0)),
                "provider_prompt_tokens": provider_prompt_tokens,
                "provider_completion_tokens": provider_completion_tokens,
                "provider_total_tokens": provider_total_tokens,
                "provider_cached_tokens": provider_cached_tokens,
                "provider_reasoning_tokens": provider_reasoning_tokens,
            }

            if self.observer is not None:
                self.observer.generation(
                    observer_trace,
                    "answer_generation",
                    model=provider_model,
                    input_text=answer_prompt,
                    output_text=answer_raw,
                    metadata={
                        "doc_refs": doc_refs,
                        "token_breakdown": answer_token_breakdown,
                    },
                    usage={
                        "input": int(token_summary.get("answer_input_tokens", 0)),
                        "output": int(token_summary.get("answer_output_tokens", 0)),
                        "total": int(
                            token_summary.get("answer_input_tokens", 0)
                            + token_summary.get("answer_output_tokens", 0)
                        ),
                    },
                )

            available_pages = [
                _as_int(item.get("page")) for item in (trace.page_result.content if trace.page_result else [])
            ]
            trace.answer_result = parse_answer_result(
                raw=answer_raw,
                doc_name=primary_doc_name,
                available_pages=[p for p in available_pages if p is not None],
                page_content=(trace.page_result.content if trace.page_result else []),
                query=query,
                allowed_doc_refs=doc_refs,
                fallback_doc_ref=(doc_refs[0] if doc_refs else primary_doc_name),
            )
            emit_observer_event(
                "answer_result_finalized",
                {
                    "answer_type": getattr(trace.answer_result, "answer_type", None),
                    "confidence": getattr(trace.answer_result, "confidence", None),
                    "cited_pages": getattr(trace.answer_result, "cited_pages", []) or [],
                    "cited_source_count": len(getattr(trace.answer_result, "cited_sources", []) or []),
                },
            )
            emit("final_answer", {"answer_result": trace.answer_result})

        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            trace.errors.append(message)
            emit("error", {"error": message})
            if self.logger:
                self.logger.exception("run_query_failed trace_id=%s", trace.trace_id)
            if self.observer is not None:
                self.observer.end_trace(observer_trace, error=message)

        trace.timings["total_sec"] = time.perf_counter() - total_start
        # totals for cost calculation: sum of tokens sent to LLMs and returned from LLMs
        token_summary["total_input_tokens"] = (
            token_summary.get("tree_search_input_tokens", 0)
            + token_summary.get("answer_input_tokens", 0)
        )
        token_summary["total_output_tokens"] = (
            token_summary.get("tree_search_output_tokens", 0)
            + token_summary.get("answer_output_tokens", 0)
        )
        trace.token_usage = {k: int(v) for k, v in token_summary.items()}
        trace.cache_stats = {
            "prepared_doc_cache": _PREPARED_DOC_CACHE.snapshot(),
            "metadata_cache": self.api_client.cache_stats() if hasattr(self.api_client, "cache_stats") else {},
        }
        if self.observer is not None and not trace.errors:
            self.observer.end_trace(
                observer_trace,
                output_payload={
                    "answer_type": getattr(trace.answer_result, "answer_type", None),
                    "confidence": getattr(trace.answer_result, "confidence", None),
                },
                metadata={
                    "timings": trace.timings,
                    "token_usage": trace.token_usage,
                    "memory": trace.memory,
                    "routing": trace.routing,
                    "cache_stats": trace.cache_stats,
                },
            )
        return trace


def run_query(
    query: str,
    doc_id: str,
    doc_name: str,
    tree: Sequence[Dict[str, Any]],
    *,
    api_client: Any,
    llm_client: Any,
    config: AppConfig,
    logger: Optional[Any] = None,
    stage_callback: Optional[StageCallback] = None,
) -> QueryTrace:
    pipeline = RAGPipeline(
        api_client=api_client,
        llm_client=llm_client,
        config=config,
        logger=logger,
    )
    return pipeline.run_query(
        query=query,
        doc_id=doc_id,
        doc_name=doc_name,
        tree=tree,
        stage_callback=stage_callback,
    )
