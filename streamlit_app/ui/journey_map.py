from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

from streamlit_app.rag.types import QueryTrace


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _short(value: Any, max_chars: int = 44) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _dot_escape(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    return text


def _doc_ref(doc_id: Any, doc_name: Any) -> str:
    safe_name = re.sub(r"[<>;]+", "_", str(doc_name or "unknown.pdf")).strip() or "unknown.pdf"
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "", str(doc_id or ""))[:8]
    return f"{safe_name}#{safe_id}" if safe_id else safe_name


@dataclass(frozen=True)
class JourneyNode:
    node_id: str
    label: str
    node_type: str
    lane: int
    highlighted: bool = False
    detail: str = ""


@dataclass(frozen=True)
class JourneyEdge:
    source_id: str
    target_id: str
    edge_type: str
    highlight_kind: str = "none"  # none | selected | cited


@dataclass(frozen=True)
class JourneyGraph:
    nodes: List[JourneyNode]
    edges: List[JourneyEdge]
    summary: Dict[str, Any]


def _cache_key(
    trace: QueryTrace,
    *,
    compact: bool,
    max_sections_per_doc: int,
    max_evidence_nodes: int,
) -> str:
    return "|".join(
        [
            str(getattr(trace, "trace_id", "")),
            str(compact),
            str(max_sections_per_doc),
            str(max_evidence_nodes),
        ]
    )


def _get_or_build_graph(
    trace: QueryTrace,
    *,
    compact: bool,
    max_sections_per_doc: int,
    max_evidence_nodes: int,
) -> JourneyGraph:
    cache = st.session_state.setdefault("_retrieval_journey_cache", {})
    key = _cache_key(
        trace,
        compact=compact,
        max_sections_per_doc=max_sections_per_doc,
        max_evidence_nodes=max_evidence_nodes,
    )
    cached = cache.get(key)
    if isinstance(cached, JourneyGraph):
        return cached
    graph = build_retrieval_journey_graph(
        trace,
        compact=compact,
        max_sections_per_doc=max_sections_per_doc,
        max_evidence_nodes=max_evidence_nodes,
    )
    cache[key] = graph
    if len(cache) > 80:
        # Keep cache bounded in long chat sessions.
        stale_keys = list(cache.keys())[:-40]
        for stale_key in stale_keys:
            cache.pop(stale_key, None)
    return graph


def build_retrieval_journey_graph(
    trace: QueryTrace,
    *,
    compact: bool,
    max_sections_per_doc: int,
    max_evidence_nodes: int,
) -> JourneyGraph:
    nodes: List[JourneyNode] = []
    edges: List[JourneyEdge] = []
    node_ids = set()
    edge_ids = set()

    def add_node(node: JourneyNode) -> None:
        if node.node_id in node_ids:
            return
        node_ids.add(node.node_id)
        nodes.append(node)

    def add_edge(edge: JourneyEdge) -> None:
        key = (edge.source_id, edge.target_id, edge.edge_type)
        if key in edge_ids:
            return
        edge_ids.add(key)
        edges.append(edge)

    query_text = _short(getattr(trace, "query", ""), 80)
    query_node_id = "journey::query"
    add_node(
        JourneyNode(
            node_id=query_node_id,
            label=f"Query\n{query_text or '(empty)'}",
            node_type="query",
            lane=0,
            highlighted=True,
        )
    )

    answer = getattr(trace, "answer_result", None)
    page_result = getattr(trace, "page_result", None)
    routing = getattr(trace, "routing", {}) or {}
    raw = (getattr(page_result, "raw", {}) if page_result else {}) or {}
    content = list((getattr(page_result, "content", []) if page_result else []) or [])

    selected_parents_by_doc = raw.get("selected_parent_nodes")
    if not isinstance(selected_parents_by_doc, dict):
        selected_parents_by_doc = {}

    cited_sources = list((getattr(answer, "cited_sources", []) if answer else []) or [])
    cited_exact = {
        (
            str(item.get("doc_ref") or "").strip(),
            _as_int(item.get("page")),
            str(item.get("element_id") or "").strip(),
        )
        for item in cited_sources
        if str(item.get("doc_ref") or "").strip() and _as_int(item.get("page"))
    }
    cited_pages = {
        (str(item.get("doc_ref") or "").strip(), _as_int(item.get("page")))
        for item in cited_sources
        if str(item.get("doc_ref") or "").strip() and _as_int(item.get("page"))
    }

    route_doc_ids = list(routing.get("route_doc_ids") or [])
    route_doc_names = list(routing.get("route_doc_names") or [])

    doc_refs: List[str] = []
    for index, doc_name in enumerate(route_doc_names):
        doc_id = route_doc_ids[index] if index < len(route_doc_ids) else ""
        doc_ref = _doc_ref(doc_id, doc_name)
        if doc_ref and doc_ref not in doc_refs:
            doc_refs.append(doc_ref)

    for doc_ref in selected_parents_by_doc.keys():
        clean_ref = str(doc_ref or "").strip()
        if clean_ref and clean_ref not in doc_refs:
            doc_refs.append(clean_ref)

    for item in content:
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or "").strip()
        if doc_ref and doc_ref not in doc_refs:
            doc_refs.append(doc_ref)

    for item in cited_sources:
        doc_ref = str(item.get("doc_ref") or "").strip()
        if doc_ref and doc_ref not in doc_refs:
            doc_refs.append(doc_ref)

    if not doc_refs:
        primary_ref = _doc_ref(getattr(trace, "doc_id", ""), getattr(trace, "doc_name", ""))
        if primary_ref:
            doc_refs.append(primary_ref)

    doc_node_map: Dict[str, str] = {}
    for doc_ref in doc_refs:
        doc_node_id = f"journey::doc::{doc_ref}"
        doc_highlight = any(
            item[0] == doc_ref for item in cited_pages
        ) or bool(selected_parents_by_doc.get(doc_ref))
        doc_node_map[doc_ref] = doc_node_id
        add_node(
            JourneyNode(
                node_id=doc_node_id,
                label=f"Document\n{_short(doc_ref, 58)}",
                node_type="document",
                lane=1,
                highlighted=doc_highlight,
            )
        )
        add_edge(
            JourneyEdge(
                source_id=query_node_id,
                target_id=doc_node_id,
                edge_type="routed_to",
                highlight_kind="selected" if doc_highlight else "none",
            )
        )

    section_node_map: Dict[Tuple[str, str], str] = {}
    selected_section_pairs = set()
    for doc_ref, sections in selected_parents_by_doc.items():
        clean_doc_ref = str(doc_ref or "").strip()
        if clean_doc_ref not in doc_node_map:
            continue
        section_items = list(sections or [])
        # Defensive: filter out any non-dict items to prevent AttributeError
        section_items = [item for item in section_items if isinstance(item, dict)]
        if not section_items:
            continue
        section_items.sort(key=lambda item: _safe_float(item.get("score")), reverse=True)
        if compact:
            section_items = section_items[: max(1, max_sections_per_doc)]
        for index, section in enumerate(section_items, start=1):
            node_id = str(section.get("node_id") or "").strip()
            if not node_id:
                continue
            title = _short(section.get("title") or "(untitled)", 48)
            score = _safe_float(section.get("score"))
            pages = ",".join(str(p) for p in (section.get("pages") or []) if _as_int(p))
            detail = f"rank={index} score={score:.2f}"
            if pages:
                detail += f" pages={pages}"
            section_graph_id = f"journey::section::{clean_doc_ref}::{node_id}"
            section_node_map[(clean_doc_ref, node_id)] = section_graph_id
            selected_section_pairs.add((clean_doc_ref, node_id))
            add_node(
                JourneyNode(
                    node_id=section_graph_id,
                    label=f"Section\n{title}",
                    node_type="section",
                    lane=2,
                    highlighted=True,
                    detail=detail,
                )
            )
            add_edge(
                JourneyEdge(
                    source_id=doc_node_map[clean_doc_ref],
                    target_id=section_graph_id,
                    edge_type="selected_parent",
                    highlight_kind="selected",
                )
            )

    evidence_candidates: List[Dict[str, Any]] = []
    for item in content:
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or "").strip()
        page = _as_int(item.get("page"))
        if not doc_ref or not page:
            continue
        element_id = str(item.get("element_id") or "unknown").strip() or "unknown"
        node_id = str(item.get("node_id") or "").strip()
        score = _safe_float(item.get("score"))
        content_type = str(item.get("content_type") or "TEXT").strip().upper() or "TEXT"
        chunk_id = str(item.get("chunk_id") or "").strip()
        is_cited = (doc_ref, page, element_id) in cited_exact or (doc_ref, page) in cited_pages
        evidence_candidates.append(
            {
                "doc_ref": doc_ref,
                "page": page,
                "element_id": element_id,
                "node_id": node_id,
                "score": score,
                "content_type": content_type,
                "chunk_id": chunk_id,
                "is_cited": is_cited,
                "text": _short(item.get("text") or "", 120),
            }
        )

    evidence_candidates.sort(
        key=lambda item: (
            0 if item.get("is_cited") else 1,
            -_safe_float(item.get("score")),
            _as_int(item.get("page")) or 0,
        )
    )

    if compact:
        forced = [item for item in evidence_candidates if item.get("is_cited")]
        optional = [item for item in evidence_candidates if not item.get("is_cited")]
        evidence_candidates = forced + optional[: max(0, max_evidence_nodes - len(forced))]
    else:
        evidence_candidates = evidence_candidates[: max(1, max_evidence_nodes * 2)]

    evidence_node_ids_by_key: Dict[Tuple[str, int, str], str] = {}
    evidence_node_ids_by_page: Dict[Tuple[str, int], List[str]] = {}
    evidence_count = 0
    for evidence in evidence_candidates:
        doc_ref = str(evidence["doc_ref"])
        page = int(evidence["page"])
        element_id = str(evidence["element_id"])
        node_key = (doc_ref, page, element_id)
        if node_key in evidence_node_ids_by_key:
            continue

        content_type = str(evidence["content_type"])
        score = _safe_float(evidence.get("score"))
        node_id = (
            f"journey::evidence::{doc_ref}::p{page}::{element_id}"
            if element_id
            else f"journey::evidence::{doc_ref}::p{page}"
        )
        detail = f"{content_type} p.{page}"
        if score > 0:
            detail += f" score={score:.2f}"
        if evidence.get("text"):
            detail += f" sample={evidence['text']}"
        add_node(
            JourneyNode(
                node_id=node_id,
                label=f"Evidence\np.{page} {content_type}",
                node_type="evidence",
                lane=3,
                highlighted=bool(evidence.get("is_cited")),
                detail=detail,
            )
        )
        evidence_node_ids_by_key[node_key] = node_id
        evidence_node_ids_by_page.setdefault((doc_ref, page), []).append(node_id)
        evidence_count += 1

        section_node_id = section_node_map.get((doc_ref, str(evidence.get("node_id") or "")))
        if section_node_id:
            add_edge(
                JourneyEdge(
                    source_id=section_node_id,
                    target_id=node_id,
                    edge_type="selected_chunk",
                    highlight_kind="selected",
                )
            )
        elif doc_ref in doc_node_map:
            add_edge(
                JourneyEdge(
                    source_id=doc_node_map[doc_ref],
                    target_id=node_id,
                    edge_type="evidence_fallback",
                    highlight_kind="selected",
                )
            )

    citation_count = 0
    citation_node_ids: List[str] = []
    for source in cited_sources:
        doc_ref = str(source.get("doc_ref") or "").strip()
        page = _as_int(source.get("page"))
        if not doc_ref or not page:
            continue
        content_type = str(source.get("content_type") or "TEXT").strip().upper() or "TEXT"
        element_id = str(source.get("element_id") or "unknown").strip() or "unknown"
        citation_id = str(source.get("citation_id") or "").strip()
        citation_node_id = (
            f"journey::citation::{citation_id}"
            if citation_id
            else f"journey::citation::{doc_ref}::p{page}::{content_type}::{element_id}"
        )
        add_node(
            JourneyNode(
                node_id=citation_node_id,
                label=f"Citation\np.{page} {content_type}",
                node_type="citation",
                lane=4,
                highlighted=True,
                detail=f"{_short(doc_ref, 56)} element={_short(element_id, 28)}",
            )
        )
        citation_node_ids.append(citation_node_id)
        citation_count += 1

        evidence_node_id = evidence_node_ids_by_key.get((doc_ref, page, element_id))
        if evidence_node_id is None:
            same_page = evidence_node_ids_by_page.get((doc_ref, page), [])
            evidence_node_id = same_page[0] if same_page else None
        if evidence_node_id is not None:
            add_edge(
                JourneyEdge(
                    source_id=evidence_node_id,
                    target_id=citation_node_id,
                    edge_type="cited_in_answer",
                    highlight_kind="cited",
                )
            )
        elif doc_ref in doc_node_map:
            add_edge(
                JourneyEdge(
                    source_id=doc_node_map[doc_ref],
                    target_id=citation_node_id,
                    edge_type="citation_fallback",
                    highlight_kind="cited",
                )
            )

    answer_node_id = "journey::answer"
    answer_type = str(getattr(answer, "answer_type", "DIRECT") or "DIRECT")
    confidence = str(getattr(answer, "confidence", "MEDIUM") or "MEDIUM")
    add_node(
        JourneyNode(
            node_id=answer_node_id,
            label=f"Final Answer\n{answer_type} / {confidence}",
            node_type="answer",
            lane=5,
            highlighted=True,
        )
    )

    if citation_node_ids:
        for citation_node_id in citation_node_ids:
            add_edge(
                JourneyEdge(
                    source_id=citation_node_id,
                    target_id=answer_node_id,
                    edge_type="supports",
                    highlight_kind="cited",
                )
            )
    else:
        add_edge(
            JourneyEdge(
                source_id=query_node_id,
                target_id=answer_node_id,
                edge_type="direct_answer",
                highlight_kind="selected",
            )
        )

    return JourneyGraph(
        nodes=nodes,
        edges=edges,
        summary={
            "documents": len(doc_refs),
            "sections": len(section_node_map),
            "evidence": evidence_count,
            "citations": citation_count,
            "answer_type": answer_type,
            "confidence": confidence,
        },
    )


def _node_style(node: JourneyNode) -> Dict[str, str]:
    base = {
        "query": ("ellipse", "#0f6ab4", "#e8f2fd"),
        "document": ("box", "#2f4a6d", "#eef3fb"),
        "section": ("box", "#2a6f65", "#eaf7f4"),
        "evidence": ("note", "#6e5f22", "#fdf6e5"),
        "citation": ("diamond", "#4c5f82", "#edf1f8"),
        "answer": ("box", "#0d4f8b", "#e6f2fd"),
    }
    shape, border, fill = base.get(node.node_type, ("box", "#556070", "#f4f7fc"))
    if not node.highlighted:
        border = "#aab7cc"
        fill = "#f4f6fa"
    return {
        "shape": shape,
        "color": border,
        "fillcolor": fill,
        "fontcolor": "#122032",
        "style": "filled,rounded",
        "penwidth": "2.1" if node.highlighted else "1.0",
    }


def _edge_style(edge: JourneyEdge) -> Dict[str, str]:
    if edge.highlight_kind == "cited":
        return {
            "color": "#1d7c57",
            "penwidth": "2.6",
            "style": "solid",
        }
    if edge.highlight_kind == "selected":
        return {
            "color": "#0f6ab4",
            "penwidth": "2.2",
            "style": "solid",
        }
    return {
        "color": "#bdc8d8",
        "penwidth": "1.0",
        "style": "dashed",
    }


def journey_graph_to_dot(graph: JourneyGraph) -> str:
    lines = [
        "digraph RetrievalJourney {",
        "rankdir=LR;",
        'graph [fontsize=11, fontname="Helvetica", bgcolor="white", splines=true, overlap=false];',
        'node [fontsize=10, fontname="Helvetica"];',
        'edge [fontsize=9, fontname="Helvetica"];',
    ]

    lanes: Dict[int, List[str]] = {}
    for node in graph.nodes:
        attrs = _node_style(node)
        attrs["label"] = _dot_escape(node.label)
        if node.detail:
            attrs["tooltip"] = _dot_escape(node.detail)
        attr_text = ", ".join(f'{key}="{value}"' for key, value in attrs.items())
        lines.append(f'"{_dot_escape(node.node_id)}" [{attr_text}];')
        lanes.setdefault(int(node.lane), []).append(node.node_id)

    for lane in sorted(lanes.keys()):
        lines.append(f"subgraph rank_group_{lane} {{")
        lines.append("rank=same;")
        for node_id in lanes[lane]:
            lines.append(f'"{_dot_escape(node_id)}";')
        lines.append("}")

    for edge in graph.edges:
        attrs = _edge_style(edge)
        attrs["tooltip"] = _dot_escape(edge.edge_type)
        attr_text = ", ".join(f'{key}="{value}"' for key, value in attrs.items())
        lines.append(
            f'"{_dot_escape(edge.source_id)}" -> "{_dot_escape(edge.target_id)}" [{attr_text}];'
        )

    lines.append("}")
    return "\n".join(lines)


def _render_fallback_list(graph: JourneyGraph) -> None:
    highlighted_edges = [edge for edge in graph.edges if edge.highlight_kind in {"selected", "cited"}]
    node_lookup = {node.node_id: node for node in graph.nodes}
    if not highlighted_edges:
        st.info("No highlighted traversal edges were available for a fallback view.")
        return
    st.markdown("**Fallback Journey Path**")
    for edge in highlighted_edges:
        source = node_lookup.get(edge.source_id)
        target = node_lookup.get(edge.target_id)
        if not source or not target:
            continue
        prefix = "[CITED]" if edge.highlight_kind == "cited" else "[PATH]"
        st.markdown(
            f"- {prefix} {source.node_type}: {_short(source.label, 58)} -> {target.node_type}: {_short(target.label, 58)}"
        )


def render_retrieval_journey_map(trace: Optional[QueryTrace]) -> None:
    if trace is None:
        st.info("Retrieval journey map is not available for this response.")
        return

    raw = (getattr(trace.page_result, "raw", {}) if trace.page_result else {}) or {}
    has_parent_data = bool(raw.get("selected_parent_nodes"))
    has_evidence = bool((getattr(trace.page_result, "content", None) if trace.page_result else None))
    if not has_parent_data and not has_evidence:
        st.info("No traversal metadata was captured for this trace.")
        return

    map_key = re.sub(r"[^0-9a-zA-Z_]+", "_", str(getattr(trace, "trace_id", "journey")))
    compact = st.toggle("Compact view", value=True, key=f"journey_compact_{map_key}")
    max_sections_per_doc = st.slider(
        "Sections per document",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        key=f"journey_sections_{map_key}",
    )
    max_evidence_nodes = st.slider(
        "Evidence nodes",
        min_value=6,
        max_value=36,
        value=18,
        step=2,
        key=f"journey_evidence_{map_key}",
    )

    graph = _get_or_build_graph(
        trace,
        compact=compact,
        max_sections_per_doc=max_sections_per_doc,
        max_evidence_nodes=max_evidence_nodes,
    )
    dot = journey_graph_to_dot(graph)

    st.caption(
        "Blue edges show selected retrieval traversal; green edges show citations that directly support the final answer."
    )
    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception:
        st.warning("Interactive graph rendering is unavailable in this environment. Showing a text fallback.")
        _render_fallback_list(graph)

    summary = graph.summary
    st.caption(
        " | ".join(
            [
                f"docs: {summary.get('documents', 0)}",
                f"sections: {summary.get('sections', 0)}",
                f"evidence: {summary.get('evidence', 0)}",
                f"citations: {summary.get('citations', 0)}",
                f"answer: {summary.get('answer_type', 'n/a')}/{summary.get('confidence', 'n/a')}",
            ]
        )
    )
