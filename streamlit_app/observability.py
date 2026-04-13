from __future__ import annotations

import json
import logging
from typing import Any, Dict

from streamlit_app.rag.types import QueryTrace


def get_logger(name: str = "streamlit_app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_query_trace(logger: logging.Logger, trace: QueryTrace) -> None:
    payload: Dict[str, Any] = {
        "trace_id": trace.trace_id,
        "query": trace.query,
        "doc_id": trace.doc_id,
        "doc_name": trace.doc_name,
        "selected_doc_ids": trace.selected_doc_ids,
        "selected_doc_names": trace.selected_doc_names,
        "retrieved_node_ids": [n.node_id for n in trace.retrieved_nodes],
        "page_request": trace.page_result.requested_pages if trace.page_result else None,
        "page_returned": trace.page_result.returned_pages if trace.page_result else None,
        "cited_pages": trace.answer_result.cited_pages if trace.answer_result else [],
        "timings": trace.timings,
        "token_usage": getattr(trace, "token_usage", {}),
        "memory": getattr(trace, "memory", {}),
        "routing": getattr(trace, "routing", {}),
        "cache_stats": getattr(trace, "cache_stats", {}),
        "errors": trace.errors,
    }
    logger.info("query_trace=%s", json.dumps(payload, ensure_ascii=False))
