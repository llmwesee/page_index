from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import streamlit as st

from streamlit_app.rag.types import AnswerResult, PageContentResult, QueryTrace, RetrievedNode
from streamlit_app.ui.journey_map import render_retrieval_journey_map


_IMAGE_TEXT_PATTERN = re.compile(r"Image\s*\(([^)]+)\)\s*:", re.IGNORECASE)
_INLINE_CITATION_PATTERN = re.compile(
    r"<doc=([^;>]+);page=(\d+)(?:;type=([^;>]+))?(?:;element=([^;>]+))?>",
    re.IGNORECASE,
)


def _extract_image_file_ref(item: Dict) -> str:
    direct = str(item.get("image_file") or item.get("file") or item.get("image_path") or "").strip()
    if direct:
        return direct
    text = str(item.get("text") or "")
    match = _IMAGE_TEXT_PATTERN.search(text)
    if match:
        return (match.group(1) or "").strip()
    return ""


def _candidate_image_paths(image_ref: str) -> List[Path]:
    if not image_ref:
        return []
    ref = image_ref.replace("\\", "/").strip()
    if not ref:
        return []

    ref_path = Path(ref)
    roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[2] / "pageindex_backend" / "data",
        Path(__file__).resolve().parents[3],
        # Path(__file__).resolve().parents[3] / "doctor_bot_v2" / "outputs" / "images",
        # Path(__file__).resolve().parents[3] / "doctor_bot_v2" / "outputs" / "images" / "files",
        Path(__file__).resolve().parents[3] / "doctor_bot_v2" / "outputs" / "images_ppt",
        Path(__file__).resolve().parents[3] / "doctor_bot_v2" / "outputs" / "images_ppt" / "files",
    ]

    candidates: List[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        for root in roots:
            candidates.append(root / ref_path)
            if ref.startswith("files/"):
                candidates.append(root / ref[len("files/"):])
            else:
                candidates.append(root / "files" / ref)
    return candidates


def _resolve_image_path(item: Dict) -> Optional[Path]:
    image_ref = _extract_image_file_ref(item)
    for path in _candidate_image_paths(image_ref):
        if path.exists() and path.is_file():
            return path
    return None


def _find_image_paths_for_pages(pages: Sequence[int]) -> List[Path]:
    valid_pages = sorted({int(p) for p in pages if isinstance(p, int) and p > 0})
    if not valid_pages:
        return []

    search_dirs = [
        Path(__file__).resolve().parents[2] / "pageindex_backend" / "data" / "files",
        # Path(__file__).resolve().parents[3] / "doctor_bot_v2" / "outputs" / "images" / "files",
        Path(__file__).resolve().parents[3] / "doctor_bot_v2" / "outputs" / "images_ppt" / "files",
    ]

    found: List[Path] = []
    seen = set()
    for page in valid_pages:
        pattern = f"page_{page:04d}_region_*.png"
        for directory in search_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for path in sorted(directory.glob(pattern)):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(path)
    return found


def _extract_image_citations_from_markdown(markdown: str) -> List[Dict[str, object]]:
    citations: List[Dict[str, object]] = []
    for doc_ref, page_raw, content_type_raw, element_raw in _INLINE_CITATION_PATTERN.findall(markdown or ""):
        content_type = (content_type_raw or "TEXT").strip().upper()
        if content_type != "IMAGE":
            continue
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            continue
        citations.append(
            {
                "doc_ref": (doc_ref or "").strip(),
                "page": page,
                "element_id": (element_raw or "").strip(),
                "content_type": "IMAGE",
            }
        )
    return citations


def _extract_inline_citations_from_markdown(markdown: str) -> List[Dict[str, object]]:
    citations: List[Dict[str, object]] = []
    for doc_ref, page_raw, content_type_raw, element_raw in _INLINE_CITATION_PATTERN.findall(markdown or ""):
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            continue
        citations.append(
            {
                "doc_ref": (doc_ref or "").strip(),
                "page": page,
                "content_type": (content_type_raw or "TEXT").strip().upper(),
                "element_id": (element_raw or "").strip(),
            }
        )
    return citations


def shorten_text(value: str, max_chars: int) -> str:
    text = " ".join((value or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _answer_tone(answer_type: str) -> str:
    mapping = {
        "DIRECT": "Grounded answer from cited evidence",
        "PARTIAL": "Supported answer with a stated limitation",
        "INSUFFICIENT": "Evidence gap identified clearly",
        "DRAFT": "Drafting a grounded answer in real time",
    }
    return mapping.get((answer_type or "").upper(), "Grounded answer from cited evidence")


def _strip_leading_heading(markdown: str) -> str:
    text = (markdown or "").strip()
    return re.sub(r"^\s{0,3}##\s+Final Answer\s*", "", text, count=1, flags=re.IGNORECASE).strip()


def _strip_sources_section(markdown: str) -> str:
    text = (markdown or "").strip()
    return re.sub(r"\n#{1,6}\s*Sources\b[\s\S]*$", "", text, flags=re.IGNORECASE).strip()


def _render_answer_hero(answer_result: AnswerResult) -> None:
    raw_type = (answer_result.answer_type or "DIRECT").upper()
    raw_confidence = (answer_result.confidence or "MEDIUM").upper()
    answer_type = "STREAMING" if raw_type == "DRAFT" else raw_type
    confidence = "LIVE" if raw_confidence == "LIVE" else raw_confidence
    tone = html.escape(_answer_tone(answer_result.answer_type))
    st.markdown(
        f"""
<div class="answer-hero">
  <div class="answer-hero-topline">Assistant Response</div>
  <div class="answer-hero-title">Final Answer</div>
  <div class="answer-hero-subtitle">{tone}</div>
  <div class="answer-pill-row">
    <span class="answer-pill answer-pill-type">{answer_type}</span>
    <span class="answer-pill answer-pill-confidence">{confidence} confidence</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_answer_stats(answer_result: AnswerResult) -> None:
    cited_count = len(answer_result.cited_sources or [])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
<div class="answer-stat-card">
  <div class="answer-stat-label">Mode</div>
  <div class="answer-stat-value">{html.escape(answer_result.answer_type or 'DIRECT')}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="answer-stat-card">
  <div class="answer-stat-label">Confidence</div>
  <div class="answer-stat-value">{html.escape(answer_result.confidence or 'MEDIUM')}</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
<div class="answer-stat-card">
  <div class="answer-stat-label">Evidence Anchors</div>
  <div class="answer-stat-value">{cited_count}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def _render_source_pills(cited_sources: Sequence[Dict]) -> None:
    if not cited_sources:
        return
    chips: List[str] = []
    seen = set()
    for item in cited_sources:
        doc_ref = str(item.get("doc_ref") or "").strip()
        page = item.get("page")
        content_type = str(item.get("content_type") or "TEXT").strip().upper() or "TEXT"
        element_id = str(item.get("element_id") or "unknown").strip() or "unknown"
        key = (doc_ref, page, content_type, element_id)
        if not doc_ref or not page or key in seen:
            continue
        seen.add(key)
        chips.append(
            f'<span class="answer-source-pill"><span class="answer-source-doc">{html.escape(shorten_text(doc_ref, 54))}</span><span class="answer-source-page">p.{page}</span><span class="answer-source-page">{html.escape(content_type)}</span><span class="answer-source-page">{html.escape(shorten_text(element_id, 24))}</span></span>'
        )
    if chips:
        st.markdown(
            "<div class=\"answer-source-strip\">" + "".join(chips) + "</div>",
            unsafe_allow_html=True,
        )


def _citation_key(doc_ref: str, page: int, content_type: str, element_id: str) -> Tuple[str, int, str, str]:
    return (
        doc_ref.strip(),
        int(page),
        (content_type or "TEXT").strip().upper(),
        (element_id or "").strip(),
    )


def _infer_content_type_from_element_id(raw_element_id: object) -> str:
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


def _infer_content_type_from_text_context(raw_text: object) -> str:
    text = f" {str(raw_text or '').lower()} "
    table_tokens = (
        " table ",
        " tabular ",
        " grid ",
        " rows ",
        " columns ",
    )
    image_tokens = (
        " image ",
        " images ",
        " figure ",
        " figures ",
        " fig ",
        " map ",
        " maps ",
        " diagram ",
        " chart ",
        " visual ",
    )
    formula_tokens = (
        " formula ",
        " equation ",
        " math ",
    )
    if any(token in text for token in image_tokens):
        return "IMAGE"
    if any(token in text for token in table_tokens):
        return "TABLE"
    if any(token in text for token in formula_tokens):
        return "FORMULA"
    return ""


def _format_answer_with_chatgpt_citations(
    markdown: str,
    cited_sources: Sequence[Dict],
) -> Tuple[str, List[Dict[str, object]]]:
    allowed_content_types = {"TEXT", "TABLE", "IMAGE", "FORMULA"}
    order: List[Dict[str, object]] = []
    seen: Dict[Tuple[str, int, str, str], int] = {}
    source_type_lookup: Dict[Tuple[str, int, str], str] = {}

    for item in cited_sources or []:
        doc_ref = str(item.get("doc_ref") or "").strip()
        page_raw = item.get("page")
        element_id = str(item.get("element_id") or "").strip()
        content_type = str(item.get("content_type") or "").strip().upper()
        if content_type not in allowed_content_types:
            content_type = ""
        inferred = _infer_content_type_from_element_id(element_id)
        if not content_type and inferred:
            content_type = inferred
        elif content_type == "TEXT" and inferred in {"TABLE", "IMAGE", "FORMULA"}:
            content_type = inferred
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            continue
        if not doc_ref or page <= 0 or not element_id:
            continue
        source_type_lookup[(doc_ref, page, element_id)] = content_type or "TEXT"

    def register(doc_ref: str, page: int, content_type: str, element_id: str) -> int:
        key = _citation_key(doc_ref, page, content_type, element_id)
        existing = seen.get(key)
        if existing is not None:
            return existing
        idx = len(order) + 1
        seen[key] = idx
        order.append(
            {
                "index": idx,
                "doc_ref": doc_ref,
                "page": int(page),
                "content_type": content_type,
                "element_id": element_id,
            }
        )
        return idx

    def replace_inline(match: re.Match[str]) -> str:
        full_text = markdown or ""
        doc_ref = (match.group(1) or "").strip()
        page_raw = match.group(2)
        raw_type = (match.group(3) or "").strip().upper()
        element_id = (match.group(4) or "").strip()
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            return ""
        if not doc_ref or page <= 0:
            return ""

        content_type = raw_type if raw_type in allowed_content_types else ""
        inferred = _infer_content_type_from_element_id(element_id)
        if not content_type and element_id:
            content_type = source_type_lookup.get((doc_ref, page, element_id), "")
        if not content_type and inferred:
            content_type = inferred
        elif content_type == "TEXT" and inferred in {"TABLE", "IMAGE", "FORMULA"}:
            content_type = inferred

        context_start = max(0, match.start() - 140)
        context_end = min(len(full_text), match.end() + 48)
        context_type = _infer_content_type_from_text_context(full_text[context_start:context_end])
        if not content_type and context_type:
            content_type = context_type
        elif content_type == "TEXT" and context_type in {"TABLE", "IMAGE", "FORMULA"}:
            content_type = context_type

        if not content_type:
            content_type = "TEXT"

        idx = register(doc_ref, page, content_type, element_id)
        return f" [\\[{idx}\\]](#citation-{idx})"

    transformed = _INLINE_CITATION_PATTERN.sub(replace_inline, markdown or "")

    for item in cited_sources or []:
        doc_ref = str(item.get("doc_ref") or "").strip()
        page_raw = item.get("page")
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            continue
        if not doc_ref or page <= 0:
            continue
        content_type = str(item.get("content_type") or "TEXT").strip().upper()
        element_id = str(item.get("element_id") or "").strip()
        register(doc_ref, page, content_type, element_id)

    # Normalize horizontal spacing without collapsing markdown line breaks.
    transformed = re.sub(r"[ \t]{2,}", " ", transformed)
    transformed = re.sub(r"\n{3,}", "\n\n", transformed)
    return transformed.strip(), order


def _render_chatgpt_source_chips(citations: Sequence[Dict[str, object]]) -> None:
    if not citations:
        return
    type_class_map = {
        "TEXT": "chatgpt-citation-chip-text",
        "TABLE": "chatgpt-citation-chip-table",
        "IMAGE": "chatgpt-citation-chip-image",
        "FORMULA": "chatgpt-citation-chip-formula",
    }
    chips: List[str] = []
    visible = list(citations[:8])
    for item in visible:
        index = int(item.get("index", 0))
        doc_ref = shorten_text(str(item.get("doc_ref") or "source"), 32)
        page = item.get("page", "?")
        content_type = str(item.get("content_type") or "TEXT").strip().upper()
        type_class = type_class_map.get(content_type, "chatgpt-citation-chip-text")
        chips.append(
            "".join(
                [
                    f'<span id="citation-{index}" class="chatgpt-citation-chip {type_class}">',
                    f'<span class="chatgpt-citation-index">[{index}]</span>',
                    f'<span class="chatgpt-citation-meta">{html.escape(doc_ref)} p.{html.escape(str(page))}</span>',
                    "</span>",
                ]
            )
        )
    extra = len(citations) - len(visible)
    if extra > 0:
        chips.append(f'<span class="chatgpt-citation-chip">+{extra} more</span>')
    st.markdown(
        "<div class=\"chatgpt-citation-row\">" + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def _render_thought_panel(
    thinking_steps: Sequence[str],
    thought_seconds: Optional[int],
    collapse: bool,
) -> None:
    steps = [str(item).strip() for item in thinking_steps if str(item).strip()]
    if not steps:
        return
    label = f"Thought for {max(1, int(thought_seconds or 0))}s"
    open_attr = "" if collapse else " open"
    items_html = "".join(f"<li>{html.escape(item)}</li>" for item in steps[-4:])
    st.markdown(
        f"""
<details class="chatgpt-thought"{open_attr}>
  <summary>
    <span class="chatgpt-thought-label">{html.escape(label)}</span>
    <span class="chatgpt-thought-chevron">›</span>
  </summary>
  <div class="chatgpt-thought-content">
    <ul>{items_html}</ul>
  </div>
</details>
        """,
        unsafe_allow_html=True,
    )


def _chunked(items: Sequence[Dict], chunk_size: int) -> List[List[Dict]]:
    rows: List[List[Dict]] = []
    size = max(1, int(chunk_size))
    for idx in range(0, len(items), size):
        rows.append(list(items[idx: idx + size]))
    return rows


def _ensure_square_image_styles() -> None:
    style_key = "_square_image_styles_injected"
    if st.session_state.get(style_key):
        return
    st.session_state[style_key] = True
    st.markdown(
        """
<style>
.fixed-square-image-box {
  width: min(100%, var(--square-size, 280px));
  aspect-ratio: 1 / 1;
  margin: 0 auto 0.45rem auto;
  border: 1px solid rgba(18, 32, 50, 0.15);
  border-radius: 0.6rem;
  overflow: hidden;
  background: #eef3f8;
}
.fixed-square-image-box > img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.fixed-square-image-caption {
  font-size: 0.82rem;
  line-height: 1.4;
  color: var(--muted);
  word-break: break-word;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _image_to_data_uri(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    mime = mime_type or "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_square_image_preview(path: Path, caption: str, size_px: int = 280) -> None:
    data_uri = _image_to_data_uri(path)
    if not data_uri:
        st.image(str(path), caption=caption, use_container_width=True)
        return
    safe_caption = html.escape(caption)
    st.markdown(
        f"""
<div class="fixed-square-image-box" style="--square-size: {int(size_px)}px;">
  <img src="{data_uri}" alt="{safe_caption}" loading="lazy" />
</div>
<div class="fixed-square-image-caption">{safe_caption}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_cited_figure_cards(image_entries: Sequence[Dict]) -> None:
    entries = list(image_entries or [])
    if not entries:
        return

    _ensure_square_image_styles()

    if len(entries) == 1:
        entry = entries[0]
        left, middle, right = st.columns([1, 2, 1])
        with middle:
            with st.container(border=True):
                _render_square_image_preview(Path(entry["path"]), str(entry["caption"]), size_px=320)
                with st.expander("Open full-size image", expanded=False):
                    st.image(str(entry["path"]), caption=entry["caption"], use_container_width=True)
        return

    max_cols = 3
    for row in _chunked(entries, max_cols):
        cols = st.columns(len(row), gap="medium")
        for col, entry in zip(cols, row):
            with col:
                with st.container(border=True):
                    _render_square_image_preview(Path(entry["path"]), str(entry["caption"]), size_px=280)
                    with st.expander("Open full-size image", expanded=False):
                        st.image(str(entry["path"]), caption=entry["caption"], use_container_width=True)


def _render_cited_images(
    answer_result: AnswerResult,
    page_result: Optional[PageContentResult],
) -> None:
    if not page_result:
        return

    page_content = page_result.content or []

    cited_set = {
        (
            str(item.get("doc_ref") or "").strip(),
            item.get("page"),
            str(item.get("element_id") or "").strip(),
        )
        for item in answer_result.cited_sources
        if str(item.get("content_type") or "").strip().upper() == "IMAGE"
    }

    markdown_citations = _extract_image_citations_from_markdown(answer_result.answer_markdown)
    all_markdown_citations = _extract_inline_citations_from_markdown(answer_result.answer_markdown)
    cited_set.update(
        (
            str(item.get("doc_ref") or "").strip(),
            item.get("page"),
            str(item.get("element_id") or "").strip(),
        )
        for item in markdown_citations
    )

    page_items = [
        item for item in page_content
        if str(item.get("content_type") or "").strip().upper() == "IMAGE"
    ]

    matched: List[Dict] = []
    matched_ids = set()

    def _add_if_new(candidate: Dict) -> None:
        cid = id(candidate)
        if cid not in matched_ids:
            matched_ids.add(cid)
            matched.append(candidate)

    # Collect pages explicitly cited as IMAGE in answer markdown/sources.
    explicit_image_pages = sorted(
        {
            int(item.get("page"))
            for item in markdown_citations
            if isinstance(item.get("page"), int) and int(item.get("page")) > 0
        }
    )
    if not explicit_image_pages:
        explicit_image_pages = sorted(
            {
                int(item.get("page"))
                for item in (answer_result.cited_sources or [])
                if str(item.get("content_type") or "").strip().upper() == "IMAGE"
                and isinstance(item.get("page"), int)
                and int(item.get("page")) > 0
            }
        )

    if cited_set:
        # 1) exact doc_ref + page + element_id for explicit IMAGE citations
        for item in page_items:
            key = (
                str(item.get("doc_ref") or item.get("doc_name") or "").strip(),
                item.get("page"),
                str(item.get("element_id") or "").strip(),
            )
            if key in cited_set:
                _add_if_new(item)

    # 2) fallback by explicit IMAGE cited pages only (never mix unrelated page here)
    if not matched and explicit_image_pages:
        for item in page_items:
            item_page = item.get("page")
            if isinstance(item_page, int) and item_page in explicit_image_pages:
                _add_if_new(item)

    # Auto-helpful-image mode: use all text/image cited pages for related figures.
    cited_pages = sorted(
        {
            int(item.get("page"))
            for item in all_markdown_citations
            if isinstance(item.get("page"), int) and int(item.get("page")) > 0
        }
    )
    if not cited_pages:
        cited_pages = sorted(
            {
                int(item.get("page"))
                for item in (answer_result.cited_sources or [])
                if isinstance(item.get("page"), int) and int(item.get("page")) > 0
            }
        )

    if not matched and cited_pages:
        for item in page_items:
            if isinstance(item.get("page"), int) and int(item.get("page")) in cited_pages:
                _add_if_new(item)

    # Build deterministic sections: cited figures first, then related figures.
    explicit_set = set(explicit_image_pages)
    related_pages = [page for page in cited_pages if page not in explicit_set]

    resolved_entries: List[Dict] = []
    seen_paths = set()
    for item in matched:
        image_path = _resolve_image_path(item)
        if not image_path:
            continue
        path_key = str(image_path.resolve())
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        doc_ref = str(item.get("doc_ref") or item.get("doc_name") or "n/a")
        page = item.get("page", "?")
        element_id = str(item.get("element_id") or "image")
        resolved_entries.append(
            {
                "path": image_path,
                "caption": f"{shorten_text(doc_ref, 54)} | p.{page} | {element_id}",
                "page": page if isinstance(page, int) else None,
            }
        )

    if not resolved_entries:
        # Try explicit cited pages first for accuracy.
        if explicit_image_pages:
            for path in _find_image_paths_for_pages(explicit_image_pages):
                path_key = str(path.resolve())
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                resolved_entries.append(
                    {
                        "path": path,
                        "caption": f"Figure from cited image page(s): {', '.join(str(p) for p in explicit_image_pages)}",
                        "page": None,
                    }
                )
        if not resolved_entries and cited_pages:
            for path in _find_image_paths_for_pages(cited_pages):
                path_key = str(path.resolve())
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                resolved_entries.append(
                    {
                        "path": path,
                        "caption": f"Figure from cited page(s): {', '.join(str(p) for p in cited_pages)}",
                        "page": None,
                    }
                )

    if not resolved_entries:
        return

    cited_entries = [
        entry for entry in resolved_entries
        if isinstance(entry.get("page"), int) and int(entry.get("page")) in explicit_set
    ]
    related_entries = [entry for entry in resolved_entries if entry not in cited_entries]

    # If explicit pages exist but nothing could be page-tagged from matched items,
    # populate cited entries directly from explicit page file lookup.
    if explicit_image_pages and not cited_entries:
        for path in _find_image_paths_for_pages(explicit_image_pages):
            path_key = str(path.resolve())
            if any(str(item.get("path", "")) == str(path) for item in cited_entries):
                continue
            if path_key in seen_paths:
                # already shown in related; move semantic priority to cited section
                pass
            cited_entries.append(
                {
                    "path": path,
                    "caption": f"Figure from cited image page(s): {', '.join(str(p) for p in explicit_image_pages)}",
                    "page": None,
                }
            )
        # Remove duplicates from related section.
        cited_paths = {str(item.get("path")) for item in cited_entries}
        related_entries = [item for item in related_entries if str(item.get("path")) not in cited_paths]

    if cited_entries:
        st.markdown("**Cited Figures**")
        _render_cited_figure_cards(cited_entries)

    if related_entries or related_pages:
        if not related_entries and related_pages:
            for path in _find_image_paths_for_pages(related_pages):
                related_entries.append(
                    {
                        "path": path,
                        "caption": f"Figure from related cited page(s): {', '.join(str(p) for p in related_pages)}",
                        "page": None,
                    }
                )
        if related_entries:
            st.markdown("**Related Figures**")
            _render_cited_figure_cards(related_entries)


def render_streaming_answer(
    partial_text: str,
    thinking_steps: Optional[Sequence[str]] = None,
    status: str = "Thinking through relevant evidence...",
    thought_seconds: Optional[int] = None,
    collapse_thinking: bool = False,
) -> None:
    preview = partial_text.strip()
    steps = [str(item).strip() for item in (thinking_steps or []) if str(item).strip()]

    if steps:
        _render_thought_panel(steps, thought_seconds, collapse=collapse_thinking)
    else:
        st.caption(status)

    if preview:
        st.markdown(_strip_sources_section(preview) + "\n\n▌")
    else:
        st.caption("Drafting answer...")


def render_thought_summary(
    thinking_steps: Optional[Sequence[str]],
    thought_seconds: Optional[int],
) -> None:
    _render_thought_panel(thinking_steps or [], thought_seconds, collapse=True)


def _tree_lines(nodes: Sequence[Dict], depth: int, max_summary_chars: int) -> List[str]:
    lines: List[str] = []
    for node in nodes:
        node_id = node.get("node_id", "----")
        title = node.get("title", "(untitled)")
        page = node.get("page_index", node.get("start_index", "?"))
        indent = "  " * depth
        lines.append(f"{indent}- `[{node_id}]` **{title}** (page: `{page}`)")

        summary = node.get("summary") or node.get("prefix_summary")
        if summary:
            lines.append(f"{indent}  - {shorten_text(str(summary), max_summary_chars)}")

        children = node.get("nodes") or []
        if children:
            lines.extend(_tree_lines(children, depth + 1, max_summary_chars))
    return lines


def render_tree_section(tree: Sequence[Dict], max_summary_chars: int = 220) -> None:
    if not tree:
        st.info("No tree available for this document yet.")
        return
    st.markdown("\n".join(_tree_lines(tree, depth=0, max_summary_chars=max_summary_chars)))


def render_retrieved_nodes(nodes: Sequence[RetrievedNode]) -> None:
    if not nodes:
        st.warning("No nodes were retrieved.")
        return
    top_nodes = list(nodes)[:3]
    table_rows = [
        {
            "rank": idx + 1,
            "node_id": n.node_id,
            "page": n.page,
            "title": n.title,
            "source": shorten_text(n.doc_ref or n.doc_name or "n/a", 42),
        }
        for idx, n in enumerate(top_nodes)
    ]
    st.caption("Showing top 3 most relevant nodes.")
    st.dataframe(table_rows, width="stretch", hide_index=True)


def render_tree_search_reasoning(reasoning_by_doc: Optional[Dict[str, str]]) -> None:
    with st.expander("Tree Search Reasoning", expanded=False):
        if not reasoning_by_doc:
            st.write("No reasoning returned.")
            return
        for doc_ref, reasoning in reasoning_by_doc.items():
            st.markdown(f"- **{doc_ref}**: {reasoning}")


def render_retrieved_nodes_section(nodes: Sequence[RetrievedNode]) -> None:
    with st.expander("Retrieved Nodes", expanded=False):
        render_retrieved_nodes(nodes)


def render_get_page_content(
    page_result: Optional[PageContentResult],
    params: Dict[str, str],
    max_preview_chars: int,
) -> None:
    with st.expander("Get page content", expanded=False):
        if not page_result:
            st.warning("No page content found.")
            return

        st.markdown("**Parameters**")
        st.json(params)

        # Allow the user to control the preview length for this panel.
        # Build a stable, unique key for the widget to avoid duplicate element IDs
        raw_doc = str(params.get("doc_name", "")) + "_" + str(params.get("pages", ""))
        safe_key = re.sub(r"[^0-9a-zA-Z_]+", "_", raw_doc)
        widget_key = f"preview_length_{safe_key}"

        preview_chars = st.number_input(
            "Preview length (chars)",
            min_value=50,
            max_value=5000,
            value=max(50, min(5000, int(max_preview_chars))),
            step=10,
            key=widget_key,
        )

        raw = page_result.raw
        st.markdown("**Debug Summary**")
        if "doc_count" in raw:
            st.markdown(
                "\n".join(
                    [
                        f"- **Documents in scope:** `{raw.get('doc_count', 0)}`",
                        f"- **Requested pages:** `{page_result.requested_pages}`",
                        f"- **Returned pages:** `{page_result.returned_pages}`",
                        f"- **Success:** `{raw.get('success', False)}`",
                    ]
                )
            )
        else:
            st.markdown(
                "\n".join(
                    [
                        f"- **Doc name:** `{raw.get('doc_name', 'n/a')}`",
                        f"- **Requested pages:** `{page_result.requested_pages}`",
                        f"- **Returned pages:** `{page_result.returned_pages}`",
                        f"- **Total pages:** `{raw.get('total_pages', 'n/a')}`",
                        f"- **Success:** `{raw.get('success', False)}`",
                    ]
                )
            )

        st.markdown(f"**Evidence Preview ({len(page_result.content)} snippet(s))**")
        if not page_result.content:
            st.write("No page content returned.")
        else:
            for item in page_result.content:
                page = item.get("page", "?")
                doc_ref = item.get("doc_ref") or item.get("doc_name") or "n/a"
                text = shorten_text(str(item.get("text", "")), max_preview_chars)
                st.markdown(f"- **{doc_ref} | Page {page}:** {text}")

        options = (raw.get("next_steps") or {}).get("options") or []
        if options:
            st.markdown("**Suggested Next Steps**")
            for idx, option in enumerate(options, start=1):
                st.markdown(f"{idx}. {option}")

        st.markdown("**Raw Result JSON**")
        st.json(raw)


def render_answer_section(
    answer_result: Optional[AnswerResult],
    page_result: Optional[PageContentResult],
    max_preview_chars: int,
    trace: Optional[QueryTrace] = None,
) -> None:
    if not answer_result:
        st.error("Answer generation failed. Check the error panel for details.")
        return

    cleaned_answer = _strip_sources_section(_strip_leading_heading(answer_result.answer_markdown))
    formatted_answer, formatted_citations = _format_answer_with_chatgpt_citations(
        cleaned_answer,
        answer_result.cited_sources,
    )
    st.markdown(formatted_answer)
    _render_chatgpt_source_chips(formatted_citations)
    _render_cited_images(answer_result, page_result)

    if trace is not None:
        with st.expander("Retrieval Journey Map", expanded=False):
            render_retrieval_journey_map(trace)


def render_trace(trace: QueryTrace, max_preview_chars: int) -> None:
    render_answer_section(
        answer_result=trace.answer_result,
        page_result=trace.page_result,
        max_preview_chars=max_preview_chars,
        trace=trace,
    )

    if trace.errors:
        st.error(" | ".join(trace.errors))
