from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


REFERENTIAL_TERMS = {
    "it",
    "its",
    "they",
    "them",
    "that",
    "this",
    "those",
    "these",
    "there",
    "above",
    "previous",
    "prior",
    "same",
    "clause",
    "section",
    "provision",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _strip_markdown(value: str) -> str:
    text = re.sub(r"<doc=[^>]+>", "", value or "")
    text = re.sub(r"[`#>*_\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(value: str, max_chars: int) -> str:
    text = _clean_text(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _assistant_turn_summary(message: Dict[str, Any], max_chars: int) -> Optional[str]:
    trace = message.get("trace")
    if not trace:
        return None
    answer_result = getattr(trace, "answer_result", None)
    answer_markdown = getattr(answer_result, "answer_markdown", "") if answer_result else ""
    if answer_markdown:
        return _truncate(_strip_markdown(answer_markdown), max_chars)
    errors = getattr(trace, "errors", None) or []
    if errors:
        return _truncate("; ".join(str(item) for item in errors), max_chars)
    return None


def build_session_memory(
    chat_history: Sequence[Dict[str, Any]],
    *,
    selected_docs: Optional[Sequence[Dict[str, Any]]] = None,
    max_turns: int = 6,
    max_chars: int = 900,
) -> Dict[str, Any]:
    selected_doc_names = [str(doc.get("name", "unknown.pdf")) for doc in (selected_docs or []) if doc.get("name")]

    turns: List[Dict[str, str]] = []
    for message in reversed(list(chat_history)):
        if len(turns) >= max(1, max_turns):
            break
        role = str(message.get("role") or "").strip().lower()
        if role == "user":
            content = _truncate(_clean_text(message.get("content", "")), max(120, max_chars // 3))
        elif role == "assistant":
            content = _assistant_turn_summary(message, max(120, max_chars // 3))
        else:
            content = None
        if not content:
            continue
        turns.append({"role": role, "content": content})

    turns.reverse()
    lines: List[str] = []
    if selected_doc_names:
        lines.append("Selected docs: " + ", ".join(selected_doc_names[:4]))
    for turn in turns:
        label = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{label}: {turn['content']}")

    context_text = _truncate("\n".join(lines), max_chars)
    return {
        "selected_doc_names": selected_doc_names,
        "recent_turns": turns,
        "context_text": context_text,
        "turn_count": len(turns),
        "max_turns": max_turns,
        "char_budget": max_chars,
    }


def resolve_query_with_session_memory(query: str, session_memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned_query = _clean_text(query)
    if not session_memory:
        return {
            "resolved_query": cleaned_query,
            "answer_context": "",
            "used": False,
            "reason": "none",
            "context_chars": 0,
        }

    context_text = _clean_text(session_memory.get("context_text", ""))
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9']+", cleaned_query)]
    has_reference = any(token in REFERENTIAL_TERMS for token in tokens)
    is_short_follow_up = len(tokens) <= 8 and bool(context_text)
    use_memory = bool(context_text) and (has_reference or is_short_follow_up)
    if not use_memory:
        return {
            "resolved_query": cleaned_query,
            "answer_context": "",
            "used": False,
            "reason": "not_needed",
            "context_chars": 0,
        }

    context_block = _truncate(context_text, int(session_memory.get("char_budget", 900) or 900))
    return {
        "resolved_query": f"{cleaned_query}\n\nConversation context:\n{context_block}".strip(),
        "answer_context": context_block,
        "used": True,
        "reason": "referential_follow_up" if has_reference else "short_follow_up",
        "context_chars": len(context_block),
    }
